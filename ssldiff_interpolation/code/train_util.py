import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
from torch.optim import AdamW

from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import time


# Initial logarithmic loss scale used for mixed-precision training.
# The actual loss-scaling factor is calculated as 2 ** lg_loss_scale.
INITIAL_LOG_LOSS_SCALE = 20.0


# Directory used to save the trained diffusion-model checkpoints.
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


class TrainLoop:
    """
    Manage the training procedure of the conditional diffusion model.

    The training loop receives two laterally shifted seismic patches from
    datasets.py. These patches are passed to the diffusion loss function to
    construct the self-supervised conditional diffusion training task.

    This class also handles optimizer updates, exponential moving average
    parameters, optional FP16 training, checkpoint loading and saving,
    learning-rate annealing, and training-log recording.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        """
        Initialize the diffusion-model training loop.

        Parameters
        ----------
        model
            Conditional U-Net used to predict the clean seismic patch.

        diffusion
            Diffusion object defining the forward noising process and the
            diffusion-model training loss.

        data
            Infinite generator returning self-supervised seismic patch pairs.

        batch_size : int
            Number of seismic patch pairs in each training batch.

        lr : float
            Initial learning rate of the AdamW optimizer.

        ema_rate
            Exponential moving average rate or comma-separated EMA rates.

        log_interval : int
            Number of iterations between logging operations.

        save_interval : int
            Number of iterations between checkpoint-saving operations.

        resume_checkpoint : str
            Path to a previously saved model checkpoint.

        use_fp16 : bool, optional
            Whether to perform mixed-precision training.

        fp16_scale_growth : float, optional
            Growth rate of the logarithmic FP16 loss scale.

        schedule_sampler
            Sampler used to select diffusion timesteps during training.

        weight_decay : float, optional
            Weight-decay coefficient used by AdamW.

        lr_anneal_steps : int, optional
            Total number of iterations used for linear learning-rate
            annealing. A value of zero disables learning-rate annealing.
        """

        # Store the conditional diffusion network.
        self.model = model

        # Determine the computing device from the model parameters.
        self.device = next(model.parameters()).device

        # Store the diffusion-process implementation.
        self.diffusion = diffusion

        # Store the infinite seismic-data generator.
        self.data = data

        # Store the number of samples in each training batch.
        self.batch_size = batch_size

        # Store the initial optimizer learning rate.
        self.lr = lr

        # Convert the EMA configuration into a list so that one or multiple
        # EMA parameter sets can be maintained.
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        # Store the training-log output interval.
        self.log_interval = log_interval

        # Store the model-checkpoint saving interval.
        self.save_interval = save_interval

        # Store the checkpoint path used to resume training.
        self.resume_checkpoint = resume_checkpoint

        # Store the mixed-precision training option.
        self.use_fp16 = use_fp16

        # Store the growth rate of the FP16 logarithmic loss scale.
        self.fp16_scale_growth = fp16_scale_growth

        # Uniformly sample diffusion timesteps when no other timestep sampler
        # is explicitly provided.
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

        # Store the AdamW weight-decay coefficient.
        self.weight_decay = weight_decay

        # Store the total number of learning-rate annealing steps.
        self.lr_anneal_steps = lr_anneal_steps

        # Number of iterations completed during the current training run.
        self.step = 0

        # Number of iterations completed before resuming from a checkpoint.
        self.resume_step = 0

        # Effective number of samples processed at each training iteration.
        self.global_batch = self.batch_size

        # Collect all trainable parameters of the conditional U-Net.
        self.model_params = list(self.model.parameters())

        # In standard-precision training, the model parameters are directly
        # used as the master parameters.
        self.master_params = self.model_params

        # Initialize the logarithmic dynamic loss-scaling factor.
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        # Record whether CUDA is available.
        self.sync_cuda = th.cuda.is_available()

        # Load network parameters when training resumes from a checkpoint.
        self._load_and_sync_parameters()

        # Create full-precision master parameters and convert the model to
        # FP16 when mixed-precision training is enabled.
        if self.use_fp16:
            self._setup_fp16()

        # Construct the AdamW optimizer used to train the diffusion model.
        self.opt = AdamW(
            self.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        if self.resume_step:
            # Restore the optimizer state when training resumes.
            self._load_optimizer_state()

            # Restore the EMA parameter sets associated with the checkpoint.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Initialize the EMA parameters from the current model parameters
            # at the beginning of a new training run.
            self.ema_params = [
                copy.deepcopy(self.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        """
        Load model parameters from the specified checkpoint.

        The training iteration is extracted from the checkpoint filename and
        stored in resume_step.
        """

        # Prefer an automatically discovered checkpoint when available.
        # Otherwise, use the explicitly provided checkpoint.
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # Extract the completed training iteration from the filename.
            self.resume_step = parse_resume_step_from_filename(
                resume_checkpoint
            )

            logger.log(
                f"loading model from checkpoint: {resume_checkpoint}..."
            )

            # Restore the network parameters on the current computing device.
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint,
                    map_location=self.device,
                )
            )

    def _load_ema_parameters(self, rate):
        """
        Load an EMA parameter set corresponding to a resumed checkpoint.

        Parameters
        ----------
        rate : float
            Exponential moving average rate.

        Returns
        -------
        list
            EMA parameters in the same representation as master_params.
        """

        # Use the current parameters as the default EMA initialization.
        ema_params = copy.deepcopy(self.master_params)

        # Locate the main checkpoint used to resume training.
        main_checkpoint = (
            find_resume_checkpoint() or self.resume_checkpoint
        )

        # Locate the EMA checkpoint associated with the same iteration.
        ema_checkpoint = find_ema_checkpoint(
            main_checkpoint,
            self.resume_step,
            rate,
        )

        if ema_checkpoint:
            logger.log(
                f"loading EMA from checkpoint: {ema_checkpoint}..."
            )

            # Load the saved EMA state dictionary.
            state_dict = th.load_state_dict(
                ema_checkpoint,
                map_location=self.device,
            )

            # Convert the state dictionary to the master-parameter format.
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """
        Load the AdamW optimizer state associated with a resumed model.

        The expected optimizer-checkpoint filename is optNNNNNN.pt, where
        NNNNNN denotes the number of completed training iterations.
        """

        # Locate the main checkpoint used to resume training.
        main_checkpoint = (
            find_resume_checkpoint() or self.resume_checkpoint
        )

        # Construct the expected optimizer-checkpoint path.
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint),
            f"opt{self.resume_step:06}.pt",
        )

        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: "
                f"{opt_checkpoint}"
            )

            # Load the serialized optimizer state.
            state_dict = th.load_state_dict(
                opt_checkpoint,
                map_location=self.device,
            )

            # Restore the AdamW optimizer state.
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Prepare the model and master parameters for FP16 training.

        Full-precision master parameters are retained for stable parameter
        updates, while the network used in forward propagation is converted
        to half precision.
        """

        # Create full-precision master parameters.
        self.master_params = make_master_params(self.model_params)

        # Convert the conditional U-Net to half precision.
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        Execute the iterative conditional diffusion training procedure.

        At every iteration, one batch of self-supervised seismic patch pairs
        is loaded, the diffusion loss is evaluated, the model is updated, and
        the EMA parameters are refreshed.
        """

        # Continue indefinitely when lr_anneal_steps is zero. Otherwise, stop
        # after reaching the specified number of annealing iterations.
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # Start timing after the first iteration to reduce the influence
            # of initial data-loading and device-initialization overhead.
            if self.step == 1:
                start = time.time()

            # Obtain a batch of one-trace-shifted seismic patches and optional
            # conditioning information from the data generator.
            batch_shot_near, batch_shot_far, cond = next(self.data)

            # Perform one complete model-training iteration.
            self.run_step(
                batch_shot_near,
                batch_shot_far,
                cond,
            )

            # Output the accumulated training statistics.
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # Save model parameters at the specified interval.
            if self.step % self.save_interval == 0:
                self.save()

                # Terminate early when running an integration test.
                if (
                    os.environ.get("DIFFUSION_TRAINING_TEST", "")
                    and self.step > 0
                ):
                    return

            # Increment the current training iteration.
            self.step += 1

            # Report the runtime for 100 training iterations.
            if self.step == 101:
                end = time.time()
                print(f'time cost {end - start} s')

        # Save the final model if the last iteration does not coincide with
        # the regular checkpoint-saving interval.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(
        self,
        batch_shot_near,
        batch_shot_far,
        cond,
    ):
        """
        Perform one complete training iteration.

        Parameters
        ----------
        batch_shot_near
            First seismic patch returned by the dataset.

        batch_shot_far
            Adjacent one-trace-shifted seismic patch returned by the dataset.

        cond : dict
            Optional model-conditioning information.
        """

        # Evaluate the diffusion loss and calculate parameter gradients.
        self.forward_backward(
            batch_shot_near,
            batch_shot_far,
            cond,
        )

        # Update model parameters using the selected numerical precision.
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()

        # Record the current training iteration and processed sample count.
        self.log_step()

    def forward_backward(
        self,
        batch_shot_near,
        batch_shot_far,
        cond,
    ):
        """
        Evaluate the diffusion training loss and backpropagate gradients.

        A diffusion timestep is sampled for every seismic patch in the batch.
        The diffusion implementation generates the noisy target and evaluates
        the conditional clean-data prediction loss.
        """

        # Clear gradients left from the previous training iteration.
        zero_grad(self.model_params)

        # Move the two seismic patch batches to the model device.
        batch_shot_near = batch_shot_near.to(self.device)
        batch_shot_far = batch_shot_far.to(self.device)

        # Sample one diffusion timestep and importance weight for each item.
        t, weights = self.schedule_sampler.sample(
            batch_shot_near.shape[0],
            self.device,
        )

        # Prepare the diffusion training-loss function using the current model,
        # the self-supervised seismic patch pair, and sampled timesteps.
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch_shot_near,
            batch_shot_far,
            t,
            model_kwargs=cond,
        )

        # Evaluate all loss components returned by the diffusion process.
        losses = compute_losses()

        # Update the adaptive timestep sampler when loss-aware sampling is used.
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t,
                losses["loss"].detach(),
            )

        # Apply the timestep-sampling weights and average over the batch.
        loss = (losses["loss"] * weights).mean()

        # Log the weighted loss values over the full diffusion schedule.
        log_loss_dict(
            self.diffusion,
            t,
            {
                k: v * weights
                for k, v in losses.items()
            },
        )

        if self.use_fp16:
            # Scale the loss to reduce gradient underflow during FP16 training.
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            # Directly backpropagate the loss in standard precision.
            loss.backward()

    def optimize_fp16(self):
        """
        Update the model using mixed-precision optimization.

        If non-finite gradients are detected, the optimization step is skipped
        and the dynamic loss scale is reduced.
        """

        # Check whether any model gradient contains NaN or infinite values.
        if any(
            not th.isfinite(p.grad).all()
            for p in self.model_params
        ):
            self.lg_loss_scale -= 1

            logger.log(
                f"Found NaN, decreased lg_loss_scale to "
                f"{self.lg_loss_scale}"
            )

            return

        # Copy gradients from the FP16 model parameters to the full-precision
        # master parameters.
        model_grads_to_master_grads(
            self.model_params,
            self.master_params,
        )

        # Remove the dynamic loss-scaling factor from the master gradients.
        self.master_params[0].grad.mul_(
            1.0 / (2 ** self.lg_loss_scale)
        )

        # Record the gradient norm before updating the model.
        self._log_grad_norm()

        # Apply linear learning-rate annealing when enabled.
        self._anneal_lr()

        # Update the full-precision master parameters using AdamW.
        self.opt.step()

        # Update every EMA parameter set.
        for rate, params in zip(
            self.ema_rate,
            self.ema_params,
        ):
            update_ema(
                params,
                self.master_params,
                rate=rate,
            )

        # Copy the optimized master parameters back to the FP16 model.
        master_params_to_model_params(
            self.model_params,
            self.master_params,
        )

        # Increase the logarithmic loss scale after a successful update.
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Update the model and EMA parameters using standard precision.
        """

        # Record the gradient norm before the optimizer update.
        self._log_grad_norm()

        # Apply linear learning-rate annealing when enabled.
        self._anneal_lr()

        # Update the conditional diffusion model parameters.
        self.opt.step()

        # Update each EMA parameter set after the optimizer step.
        for rate, params in zip(
            self.ema_rate,
            self.ema_params,
        ):
            update_ema(
                params,
                self.master_params,
                rate=rate,
            )

    def _log_grad_norm(self):
        """
        Calculate and log the L2 norm of all parameter gradients.
        """

        # Accumulate the squared L2 norm over all master parameters.
        sqsum = 0.0

        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()

        # Log the complete gradient norm.
        logger.logkv_mean(
            "grad_norm",
            np.sqrt(sqsum),
        )

    def _anneal_lr(self):
        """
        Linearly reduce the learning rate over lr_anneal_steps.
        """

        # Keep the original learning rate when annealing is disabled.
        if not self.lr_anneal_steps:
            return

        # Calculate the completed fraction of the annealing schedule.
        frac_done = (
            self.step + self.resume_step
        ) / self.lr_anneal_steps

        # Linearly reduce the learning rate toward zero.
        lr = self.lr * (1 - frac_done)

        # Apply the updated learning rate to all parameter groups.
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        Record the current training iteration and processed sample count.
        """

        # Log the absolute iteration, including resumed iterations.
        logger.logkv(
            "step",
            self.step + self.resume_step,
        )

        # Log the cumulative number of processed seismic patch samples.
        logger.logkv(
            "samples",
            (
                self.step
                + self.resume_step
                + 1
            ) * self.global_batch,
        )

        # Log the current dynamic loss scale during FP16 training.
        if self.use_fp16:
            logger.logkv(
                "lg_loss_scale",
                self.lg_loss_scale,
            )

    def save(self):
        """
        Save the current EMA model parameters.

        Saving the non-EMA parameters and optimizer state is disabled in the
        current implementation but retained as commented code.
        """

        def save_checkpoint(rate, params):
            """
            Save one set of model parameters as a checkpoint.

            Parameters
            ----------
            rate
                EMA rate associated with the saved parameter set.

            params
                Model or EMA parameters to be serialized.
            """

            # Convert the master parameters to a model-compatible state
            # dictionary while preserving all original parameter names.
            state_dict = self._master_params_to_state_dict(params)

            logger.log(f"saving model {rate}...")

            # Construct the checkpoint filename.
            if not rate:
                filename = (
                    f"model"
                    f"{(self.step+self.resume_step):06d}.pt"
                )
            else:
                filename = (
                    f"ema_{rate}_"
                    f"{(self.step+self.resume_step):06d}.pt"
                )

            # Save the state dictionary to the checkpoint directory.
            with bf.BlobFile(
                bf.join(dir_checkpoints, filename),
                "wb",
            ) as f:
                th.save(state_dict, f)

        # Saving the non-EMA model parameters is disabled.
        # save_checkpoint(0, self.master_params)

        # Save one checkpoint for each configured EMA rate.
        for rate, params in zip(
            self.ema_rate,
            self.ema_params,
        ):
            save_checkpoint(rate, params)

        # Saving the optimizer state is disabled.
        # with bf.BlobFile(
        #     bf.join(
        #         dir_checkpoints,
        #         f"opt{(self.step+self.resume_step):06d}.pt",
        #     ),
        #     "wb",
        # ) as f:
        #     th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master parameters into a model-compatible state dictionary.

        The original model parameter names are preserved so that the saved
        checkpoint can be loaded into the same network architecture.
        """

        # Recover the individual parameter tensors when flattened FP16 master
        # parameters are used.
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(),
                master_params,
            )

        # Obtain the model state dictionary, including registered buffers.
        state_dict = self.model.state_dict()

        # Replace each trainable model parameter while preserving its name.
        for i, (name, _value) in enumerate(
            self.model.named_parameters()
        ):
            assert name in state_dict
            state_dict[name] = master_params[i]

        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Convert a model state dictionary into master parameters.

        Parameters are collected in the exact order returned by
        model.named_parameters().
        """

        # Extract the saved tensors according to the model parameter names.
        params = [
            state_dict[name]
            for name, _ in self.model.named_parameters()
        ]

        # Construct full-precision master parameters for FP16 training.
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse the training iteration from a model-checkpoint filename.

    The expected filename format is path/to/modelNNNNNN.pt, where NNNNNN
    denotes the number of completed training iterations.
    """

    # Separate the substring following the final occurrence of "model".
    split = filename.split("model")

    if len(split) < 2:
        return 0

    # Remove the filename extension.
    split1 = split[-1].split(".")[0]

    try:
        return int(split1)
    except ValueError:
        return 0


def parse_dataname_from_filename(filename):
    """
    Parse the substring following "gaussian5" from a filename.
    """

    # Split the filename using the dataset-specific identifier.
    split = filename.split("gaussian5")

    if len(split) < 2:
        return 0

    # Remove the filename extension from the parsed substring.
    split1 = split[-1].split(".")[0]

    try:
        return split1
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Return the directory used for blob-based training logs.

    DIFFUSION_BLOB_LOGDIR takes precedence over the directory configured by
    the logger.
    """

    return os.environ.get(
        "DIFFUSION_BLOB_LOGDIR",
        logger.get_dir(),
    )


def find_resume_checkpoint():
    """
    Return an automatically discovered checkpoint path.

    Automatic checkpoint discovery is disabled in the current implementation.
    """

    # This function can be adapted to locate the latest checkpoint on local
    # or remote storage automatically.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Locate an EMA checkpoint associated with a main checkpoint.

    Parameters
    ----------
    main_checkpoint : str
        Path to the main model checkpoint.

    step : int
        Training iteration associated with the checkpoint.

    rate : float
        EMA rate used in the checkpoint filename.

    Returns
    -------
    str or None
        EMA checkpoint path when the file exists; otherwise, None.
    """

    # An EMA checkpoint cannot be located without a main-checkpoint path.
    if main_checkpoint is None:
        return None

    # Construct the EMA-checkpoint filename.
    filename = f"ema_{rate}_{(step):06d}.pt"

    # Search in the directory containing the main checkpoint.
    path = bf.join(
        bf.dirname(main_checkpoint),
        filename,
    )

    if bf.exists(path):
        return path

    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log the overall diffusion loss and losses over timestep quartiles.

    Parameters
    ----------
    diffusion
        Diffusion object containing the total number of timesteps.

    ts
        Diffusion timesteps sampled for the current batch.

    losses : dict
        Dictionary containing per-sample diffusion loss values.
    """

    # Process every loss component returned by training_losses().
    for key, values in losses.items():
        # Log the mean value over the complete batch.
        logger.logkv_mean(
            key,
            values.mean().item(),
        )

        # Divide the diffusion schedule into four timestep quartiles and log
        # their losses separately.
        for sub_t, sub_loss in zip(
            ts.cpu().numpy(),
            values.detach().cpu().numpy(),
        ):
            quartile = int(
                4 * sub_t / diffusion.num_timesteps
            )

            logger.logkv_mean(
                f"{key}_q{quartile}",
                sub_loss,
            )