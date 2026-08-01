from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a diffusion-timestep sampler from a predefined sampler name.

    The sampler determines how diffusion timesteps are selected during
    training. The returned importance weights ensure that the expected
    training objective remains unbiased when nonuniform sampling is used.

    Parameters
    ----------
    name : str
        Name of the timestep-sampling strategy.

        Supported values are:

        - "uniform":
          Sample every diffusion timestep with equal probability.

        - "loss-second-moment":
          Adaptively sample timesteps according to the second moment of their
          recent training losses.

    diffusion
        Diffusion-process object containing the total number of timesteps.

    Returns
    -------
    ScheduleSampler
        Configured diffusion-timestep sampler.

    Raises
    ------
    NotImplementedError
        If the requested sampler name is not supported.
    """

    # Uniformly sample all diffusion timesteps.
    if name == "uniform":
        return UniformSampler(diffusion)

    # Adaptively emphasize timesteps with large recent loss magnitudes.
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)

    # Reject unsupported timestep-sampling strategies.
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for diffusion-timestep sampling distributions.

    A ScheduleSampler assigns a positive sampling weight to every timestep in
    the diffusion process. These weights are normalized into probabilities
    when sample() is called.

    By default, the sampled training losses are multiplied by inverse-
    probability importance weights. This preserves the expected value of the
    original uniformly averaged diffusion objective even when timesteps are
    sampled nonuniformly.
    """

    @abstractmethod
    def weights(self):
        """
        Return an unnormalized positive weight for every diffusion timestep.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with length diffusion.num_timesteps.

            The values do not need to sum to one, but every value must be
            positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample diffusion timesteps for one training batch.

        Parameters
        ----------
        batch_size : int
            Number of diffusion timesteps to sample. Normally, this equals the
            number of seismic patch pairs in the current training batch.

        device : torch.device
            Device on which the returned timestep and importance-weight
            tensors are placed.

        Returns
        -------
        tuple
            A pair containing:

            timesteps:
                Integer tensor of shape [batch_size] containing the sampled
                diffusion timestep indices.

            weights:
                Floating-point tensor of shape [batch_size] containing the
                inverse-probability importance weights applied to the
                corresponding per-sample diffusion losses.
        """

        # Obtain an unnormalized positive weight for each diffusion timestep.
        w = self.weights()

        # Normalize the timestep weights into a probability distribution.
        p = w / np.sum(w)

        # Independently sample one diffusion timestep for each training sample.
        indices_np = np.random.choice(
            len(p),
            size=(batch_size,),
            p=p
        )

        # Convert the sampled timestep indices to a PyTorch integer tensor.
        indices = th.from_numpy(indices_np).long().to(device)

        # Compute inverse-probability importance weights:
        #
        #     1 / (T * p_t)
        #
        # where T is the total number of available diffusion timesteps and
        # p_t is the probability of sampling timestep t.
        weights_np = 1 / (len(p) * p[indices_np])

        # Convert the importance weights to a PyTorch floating-point tensor.
        weights = th.from_numpy(weights_np).float().to(device)

        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniformly sample all timesteps in the diffusion process.

    Every timestep receives the same sampling weight. Therefore, sample()
    produces a uniform timestep distribution and returns importance weights
    equal to one.
    """

    def __init__(self, diffusion):
        # Store the diffusion process to access its total number of timesteps.
        self.diffusion = diffusion

        # Assign an identical unnormalized weight to every timestep.
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """
        Return equal weights for all diffusion timesteps.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of ones with length
            diffusion.num_timesteps.
        """

        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Base class for adaptive timestep samplers driven by training losses.

    Subclasses update their timestep-sampling weights from recent loss values.
    In distributed training, losses are gathered from all workers before the
    sampler state is updated, ensuring that every worker maintains the same
    timestep distribution.
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Synchronize local timestep losses and update the sampler state.

        Each distributed worker calls this method using the timesteps and
        losses computed from its local mini-batch. The method gathers data from
        every worker and then calls update_with_all_losses() with identical
        global timestep-loss pairs on every rank.

        Parameters
        ----------
        local_ts : torch.Tensor
            One-dimensional integer tensor containing the diffusion timesteps
            sampled on the current distributed worker.

        local_losses : torch.Tensor
            One-dimensional tensor containing the corresponding per-sample
            diffusion losses on the current worker.
        """

        # Allocate one scalar tensor per distributed worker to collect the
        # local batch size from every rank.
        batch_sizes = [
            th.tensor(
                [0],
                dtype=th.int32,
                device=local_ts.device
            )
            for _ in range(dist.get_world_size())
        ]

        # Gather the local batch size from every distributed worker.
        dist.all_gather(
            batch_sizes,
            th.tensor(
                [len(local_ts)],
                dtype=th.int32,
                device=local_ts.device
            ),
        )

        # Convert gathered scalar tensors into Python integers.
        batch_sizes = [x.item() for x in batch_sizes]

        # Determine the largest local batch size across all workers.
        max_bs = max(batch_sizes)

        # Allocate padded timestep buffers for every distributed worker.
        #
        # Padding is required because dist.all_gather expects tensors with
        # identical shapes on every rank.
        timestep_batches = [
            th.zeros(max_bs).to(local_ts)
            for bs in batch_sizes
        ]

        # Allocate padded loss buffers for every distributed worker.
        loss_batches = [
            th.zeros(max_bs).to(local_losses)
            for bs in batch_sizes
        ]

        # Gather the local timestep tensors from every worker.
        dist.all_gather(timestep_batches, local_ts)

        # Gather the local loss tensors from every worker.
        dist.all_gather(loss_batches, local_losses)

        # Remove padded entries and flatten all gathered timestep values into
        # one Python list.
        timesteps = [
            x.item()
            for y, bs in zip(timestep_batches, batch_sizes)
            for x in y[:bs]
        ]

        # Remove padded entries and flatten all gathered loss values into one
        # Python list.
        losses = [
            x.item()
            for y, bs in zip(loss_batches, batch_sizes)
            for x in y[:bs]
        ]

        # Update the adaptive sampler using the globally gathered data.
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update adaptive timestep weights using globally gathered losses.

        Subclasses must implement this method. It is called on every
        distributed worker with identical inputs, so its behavior should be
        deterministic to keep the sampler state synchronized across workers.

        Parameters
        ----------
        ts : list of int
            Diffusion timestep indices associated with the gathered losses.

        losses : list of float
            Per-sample diffusion loss values corresponding to ts.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Adaptively sample timesteps using the second moment of recent losses.

    For every diffusion timestep, the sampler stores a fixed-length history of
    recent loss values. Once all timestep histories are populated, the sampling
    weight for timestep t is calculated as:

        sqrt(mean(loss_t ** 2))

    Timesteps with larger recent loss magnitudes are therefore sampled more
    frequently. A small uniform probability is retained so that every
    diffusion timestep continues to have a nonzero sampling probability.

    Parameters
    ----------
    diffusion
        Diffusion-process object containing the total number of timesteps.

    history_per_term : int, optional
        Number of recent losses retained for each diffusion timestep.

    uniform_prob : float, optional
        Fraction of the sampling distribution reserved for uniform sampling.
    """

    def __init__(
        self,
        diffusion,
        history_per_term=10,
        uniform_prob=0.001
    ):
        # Store the diffusion process.
        self.diffusion = diffusion

        # Number of recent losses maintained for each timestep.
        self.history_per_term = history_per_term

        # Small uniform component that prevents any timestep from receiving a
        # zero sampling probability.
        self.uniform_prob = uniform_prob

        # Loss-history matrix with shape:
        #
        #     [number of diffusion timesteps, history_per_term]
        #
        # Each row stores recent loss values for one diffusion timestep.
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term],
            dtype=np.float64
        )

        # Number of valid loss-history entries currently stored for each
        # diffusion timestep.
        self._loss_counts = np.zeros(
            [diffusion.num_timesteps],
            dtype=np.int
        )

    def weights(self):
        """
        Compute adaptive sampling weights for all diffusion timesteps.

        Before every timestep has accumulated history_per_term observations,
        uniform weights are returned. After warm-up, weights are proportional
        to the square root of the mean squared recent loss at each timestep.

        Returns
        -------
        numpy.ndarray
            Positive sampling weight for every diffusion timestep.
        """

        # Use uniform timestep sampling until every timestep has accumulated a
        # complete loss history.
        if not self._warmed_up():
            return np.ones(
                [self.diffusion.num_timesteps],
                dtype=np.float64
            )

        # Estimate the square root of the second moment of recent losses for
        # each diffusion timestep.
        weights = np.sqrt(
            np.mean(self._loss_history ** 2, axis=-1)
        )

        # Normalize the adaptive weights into a probability distribution.
        weights /= np.sum(weights)

        # Reserve most of the probability mass for loss-aware sampling.
        weights *= 1 - self.uniform_prob

        # Add a small uniform component so every timestep remains sampleable.
        weights += self.uniform_prob / len(weights)

        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Insert newly observed losses into each timestep's history.

        Once the history for a timestep is full, the oldest loss is removed
        and the newest loss is appended.

        Parameters
        ----------
        ts : list of int
            Diffusion timestep indices.

        losses : list of float
            Loss values corresponding to the timestep indices in ts.
        """

        # Process each globally gathered timestep-loss pair.
        for t, loss in zip(ts, losses):
            # If the history for timestep t is already full, discard its
            # oldest entry and append the newest loss.
            if self._loss_counts[t] == self.history_per_term:
                # Shift all stored losses one position toward the beginning.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]

                # Store the newest loss in the final history position.
                self._loss_history[t, -1] = loss

            else:
                # Add the new loss to the next unused history position.
                self._loss_history[t, self._loss_counts[t]] = loss

                # Increase the number of valid stored losses for timestep t.
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check whether every timestep has a complete loss history.

        Returns
        -------
        bool
            True when every diffusion timestep has accumulated exactly
            history_per_term loss observations.
        """

        return (
            self._loss_counts == self.history_per_term
        ).all()