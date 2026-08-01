"""
Train the self-supervised conditional diffusion model for near-offset
reconstruction of towed-streamer seismic data.
"""

import argparse

from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import torch as th


def main():
    """
    Configure and execute the conditional diffusion-model training workflow.

    The workflow consists of the following main steps:

    1. Parse model, diffusion, dataset, and optimization arguments.
    2. Construct the conditional U-Net and Gaussian diffusion process.
    3. Move the model to the GPU.
    4. Create the diffusion-timestep sampler.
    5. Construct the self-supervised seismic patch data generator.
    6. Start the iterative diffusion-model training loop.
    """

    # Parse command-line arguments using the default configurations defined
    # in create_argparser().
    args = create_argparser().parse_args()

    # Initialize the training logger and create the output log directory.
    logger.configure()

    # Use a CUDA-enabled GPU for model training.
    device = th.device('cuda')

    logger.log("creating model and diffusion...")

    # Construct the conditional U-Net and the corresponding Gaussian
    # diffusion process using the command-line configurations.
    #
    # With the default settings:
    # - the U-Net input contains two channels formed by concatenating the
    #   noisy target patch x_t and the adjacent conditioning patch shot_far;
    # - the U-Net directly predicts the clean target patch x_0;
    # - the original diffusion process contains 1000 timesteps.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(
            args,
            model_and_diffusion_defaults().keys()
        )
    )

    # Optional initialization from a previously trained model.
    #
    # The parameter names and network architecture of the pretrained model
    # must exactly match those of the current UNetModel.
    #
    # pretrained_dict = th.load(
    #     'pretrained_model.pt',
    #     map_location=device
    # )
    # model.load_state_dict(pretrained_dict)

    # Transfer all model parameters and buffers to the selected GPU.
    model.to(device)

    # Construct the diffusion-timestep sampling strategy used during training.
    #
    # The default "uniform" sampler assigns equal probability to every
    # diffusion timestep.
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion
    )

    logger.log("creating data loader...")

    # Create an infinite self-supervised seismic-data generator.
    #
    # For each selected shot gather, datasets.py extracts two overlapping
    # patches with a one-trace lateral shift:
    #
    # - one patch is used as the diffusion target;
    # - the adjacent patch is used as the conditioning context shot_far.
    #
    # width_size defines the lateral patch width W described in the manuscript.
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        width_size=args.width_size,
        device=device,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    # Create and run the complete diffusion-model training loop.
    #
    # TrainLoop handles:
    # - forward diffusion and conditional clean-patch prediction;
    # - loss calculation and backward propagation;
    # - AdamW optimization;
    # - exponential moving average parameter updates;
    # - training-log recording;
    # - periodic model-checkpoint saving.
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """
    Create the command-line argument parser for model training.

    The parser combines training-specific arguments defined in this file with
    the model and diffusion defaults returned by
    model_and_diffusion_defaults().

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line argument parser.
    """

    # Define dataset and optimization defaults.
    defaults = dict(
        # Directory containing MATLAB files for the synthetic SEAM training
        # shot gathers.
        data_dir="../dataset/seam/train/",

        # Diffusion-timestep sampling strategy used during training.
        schedule_sampler="uniform",

        # Initial learning rate of the AdamW optimizer.
        lr=1e-4,

        # Weight-decay coefficient of the AdamW optimizer.
        weight_decay=0.0,

        # Number of steps used for linear learning-rate annealing.
        # A value of zero disables learning-rate annealing.
        lr_anneal_steps=0,

        # Number of self-supervised seismic patch pairs in each training batch.
        batch_size=16,

        # Exponential moving average rate used to maintain a smoothed model.
        # Multiple rates may be provided as a comma-separated string.
        ema_rate="0.999",

        # Number of training iterations between log outputs.
        log_interval=100,

        # Number of training iterations between model-checkpoint saves.
        save_interval=2000,

        # Path to a checkpoint used to resume training.
        # An empty string starts training from the current initialization.
        resume_checkpoint="",

        # Whether to use mixed-precision FP16 training.
        use_fp16=False,

        # Growth rate of the logarithmic dynamic loss scale during FP16
        # training.
        fp16_scale_growth=1e-3,

        # Lateral width W of the target and conditioning seismic patches.
        width_size=32,
    )

    # Add the default conditional U-Net and diffusion-process configurations,
    # including channel numbers, attention resolutions, diffusion steps, and
    # the noise schedule.
    defaults.update(model_and_diffusion_defaults())

    # Construct the command-line parser.
    parser = argparse.ArgumentParser()

    # Register every default dictionary entry as a command-line argument.
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # Execute model training only when this file is run directly.
    main()
