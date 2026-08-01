import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Select a subset of timesteps from an original diffusion schedule.

    This function supports two timestep-selection strategies:

    1. Section-based selection:
       The original diffusion schedule is divided into approximately equal
       sections, and a specified number of uniformly spaced timesteps is
       selected from each section.

    2. DDIM-style selection:
       If section_counts is a string beginning with "ddim", a fixed integer
       stride is searched for so that exactly the requested number of
       timesteps is selected from the original diffusion schedule.

    For example, if the original diffusion process contains 300 timesteps and
    section_counts is [10, 15, 20], the 300 timesteps are divided into three
    sections. The function then selects 10 timesteps from the first section,
    15 from the second section, and 20 from the final section.

    Parameters
    ----------
    num_timesteps : int
        Total number of timesteps in the original diffusion process.

    section_counts : list, tuple, or str
        Number of timesteps to retain from each section.

        A comma-separated string such as "10,15,20" is converted into a list
        of section counts.

        A string such as "ddim10" requests exactly 10 DDIM sampling timesteps
        selected using a fixed integer stride.

    Returns
    -------
    set
        Set containing the selected timestep indices from the original
        diffusion process.

    Raises
    ------
    ValueError
        If the requested DDIM timestep count cannot be obtained using an
        integer stride, or if a section contains fewer original timesteps than
        the requested number of retained timesteps.
    """

    # Parse string-based timestep specifications.
    if isinstance(section_counts, str):
        # Handle the special DDIM timestep-respacing format, such as "ddim10".
        if section_counts.startswith("ddim"):
            # Extract the requested number of DDIM sampling steps.
            desired_count = int(section_counts[len("ddim") :])

            # Search for an integer stride that produces exactly the requested
            # number of retained timesteps.
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))

            # Report an error when no integer stride produces the exact count.
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )

        # Convert a comma-separated section specification into integer counts.
        section_counts = [int(x) for x in section_counts.split(",")]

    # Compute the base number of original timesteps assigned to each section.
    size_per = num_timesteps // len(section_counts)

    # Distribute any remaining timesteps among the first few sections.
    extra = num_timesteps % len(section_counts)

    # Starting timestep index of the current section.
    start_idx = 0

    # Store all selected timestep indices.
    all_steps = []

    # Process each section independently.
    for i, section_count in enumerate(section_counts):
        # Sections at the beginning receive one additional timestep when the
        # total number of timesteps is not divisible by the number of sections.
        size = size_per + (1 if i < extra else 0)

        # A section cannot provide more retained timesteps than it contains.
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )

        # When only one timestep is requested, no interpolation is required.
        if section_count <= 1:
            frac_stride = 1
        else:
            # Fractional stride used to select approximately uniformly spaced
            # timesteps, including both ends of the section.
            frac_stride = (size - 1) / (section_count - 1)

        # Current floating-point position within the section.
        cur_idx = 0.0

        # Store selected timesteps from the current section.
        taken_steps = []

        # Select the requested number of timesteps from the section.
        for _ in range(section_count):
            # Round the fractional position to the nearest original timestep.
            taken_steps.append(start_idx + round(cur_idx))

            # Advance to the next selected position.
            cur_idx += frac_stride

        # Add the current section's selected timesteps to the complete list.
        all_steps += taken_steps

        # Move the section starting index to the next section.
        start_idx += size

    # Return unique selected timesteps.
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    Gaussian diffusion process defined on a selected subset of timesteps.

    SpacedDiffusion constructs a shortened diffusion process from an original
    full diffusion schedule. The selected timesteps are specified by
    use_timesteps, while the corresponding effective beta values are derived
    so that the cumulative alpha products at retained timesteps match those of
    the original process.

    This mechanism allows the model to be trained with an original schedule,
    such as 1000 diffusion steps, while sampling with a much smaller subset,
    such as the DDIM timestep schedule used during inference.

    Parameters
    ----------
    use_timesteps
        Sequence or set containing the timestep indices retained from the
        original diffusion process.

    **kwargs
        Keyword arguments used to construct the original GaussianDiffusion
        process, including betas, model mean type, variance type, loss type,
        and timestep-rescaling configuration.
    """

    def __init__(self, use_timesteps, **kwargs):
        # Store the retained original timestep indices as a set for efficient
        # membership testing.
        self.use_timesteps = set(use_timesteps)

        # Map each timestep in the shortened diffusion process back to its
        # corresponding timestep in the original diffusion schedule.
        self.timestep_map = []

        # Record the total number of timesteps in the original beta schedule.
        self.original_num_steps = len(kwargs["betas"])

        # Construct the complete original diffusion process. Its cumulative
        # alpha products are used to derive the shortened beta schedule.
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa

        # Cumulative alpha product at the previously retained timestep.
        last_alpha_cumprod = 1.0

        # Store the effective beta values for the shortened diffusion process.
        new_betas = []

        # Traverse all timesteps in the original diffusion process.
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            # Retain only timesteps selected by use_timesteps.
            if i in self.use_timesteps:
                # Compute an effective beta value so that the cumulative alpha
                # product of the shortened process matches the original process
                # at this retained timestep.
                new_betas.append(
                    1 - alpha_cumprod / last_alpha_cumprod
                )

                # Update the previous retained cumulative alpha product.
                last_alpha_cumprod = alpha_cumprod

                # Record the original timestep represented by this shortened
                # diffusion timestep.
                self.timestep_map.append(i)

        # Replace the original beta schedule with the derived shortened
        # diffusion beta schedule.
        kwargs["betas"] = np.array(new_betas)

        # Initialize GaussianDiffusion using the shortened beta schedule.
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        """
        Evaluate the reverse-process mean and variance on respaced timesteps.

        The supplied model is wrapped so that timestep indices from the
        shortened process are mapped back to the corresponding indices in the
        original diffusion schedule before being passed to the U-Net.
        """

        return super().p_mean_variance(
            self._wrap_model(model),
            *args,
            **kwargs
        )

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        """
        Compute diffusion training losses using mapped original timesteps.

        The model wrapper converts each timestep from the shortened schedule
        into the corresponding original timestep before evaluating the
        conditional U-Net.
        """

        return super().training_losses(
            self._wrap_model(model),
            *args,
            **kwargs
        )

    def _wrap_model(self, model):
        """
        Wrap a model so that respaced timesteps are mapped to original steps.

        Parameters
        ----------
        model
            Conditional U-Net used by the diffusion process.

        Returns
        -------
        _WrappedModel
            Callable wrapper that maps shortened timestep indices to their
            corresponding original timestep values.
        """

        # Avoid wrapping a model more than once.
        if isinstance(model, _WrappedModel):
            return model

        # Construct the timestep-mapping wrapper.
        return _WrappedModel(
            model,
            self.timestep_map,
            self.rescale_timesteps,
            self.original_num_steps
        )

    def _scale_timesteps(self, t):
        """
        Return shortened diffusion timesteps without additional scaling.

        Timestep mapping and optional rescaling are performed inside
        _WrappedModel before the timesteps are passed to the U-Net.
        """

        # Scaling is performed by the wrapped model.
        return t


class _WrappedModel:
    """
    Callable wrapper that maps respaced timesteps to original timesteps.

    The SpacedDiffusion process internally uses timestep indices ranging from
    zero to the number of retained timesteps minus one. Before the conditional
    U-Net is evaluated, this wrapper converts those shortened indices back to
    the corresponding timestep indices in the original diffusion schedule.

    Parameters
    ----------
    model
        Original conditional U-Net.

    timestep_map : list
        Mapping from shortened diffusion timestep indices to original
        diffusion timestep indices.

    rescale_timesteps : bool
        Whether mapped original timesteps should be rescaled to the standard
        0--1000 range before being passed into the timestep embedding.

    original_num_steps : int
        Number of timesteps in the original diffusion schedule.
    """

    def __init__(
        self,
        model,
        timestep_map,
        rescale_timesteps,
        original_num_steps
    ):
        # Store the original conditional U-Net.
        self.model = model

        # Store the shortened-to-original timestep mapping.
        self.timestep_map = timestep_map

        # Store whether timestep values should be rescaled.
        self.rescale_timesteps = rescale_timesteps

        # Store the number of timesteps in the original diffusion process.
        self.original_num_steps = original_num_steps

    def __call__(self, x, shot_far, ts, **kwargs):
        """
        Evaluate the conditional U-Net using mapped original timesteps.

        Parameters
        ----------
        x
            Noisy target seismic patch x_t at the current shortened diffusion
            timestep.

        shot_far
            Adjacent conditioning seismic patch y. It provides the recorded
            offset context used by the U-Net to predict the clean target patch.

        ts
            Batch of timestep indices in the shortened diffusion process.

        **kwargs
            Additional keyword arguments passed directly to the U-Net, such
            as optional class labels.

        Returns
        -------
        torch.Tensor
            Output predicted by the conditional U-Net.
        """

        # Convert the timestep mapping into a tensor on the same device and
        # with the same integer dtype as the shortened timestep batch.
        map_tensor = th.tensor(
            self.timestep_map,
            device=ts.device,
            dtype=ts.dtype
        )

        # Map each shortened timestep index to its corresponding timestep in
        # the original diffusion schedule.
        new_ts = map_tensor[ts]

        # Optionally rescale original timesteps to the standard 0--1000 range
        # expected by the timestep-embedding implementation.
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (
                1000.0 / self.original_num_steps
            )

        # Evaluate the original conditional U-Net using the noisy target patch,
        # adjacent conditioning patch, and mapped timestep values.
        return self.model(
            x,
            shot_far,
            new_ts,
            **kwargs
        )