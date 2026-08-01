import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel


# Number of classes retained for compatibility with the original
# class-conditional diffusion implementation. Class conditioning is disabled
# by default in the current seismic near-offset reconstruction framework.
NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Return the default configurations of the conditional U-Net and diffusion
    process used for self-supervised near-offset reconstruction.

    The default network follows the architecture described in the manuscript:
    five resolution scales with channel multipliers (1, 2, 4, 8, 16), two
    residual blocks per scale, a base channel width of 64, and self-attention
    at downsampling factors 8 and 16.

    Returns
    -------
    dict
        Default model and diffusion hyperparameters.
    """
    return dict(
        # The model receives a two-channel input formed by concatenating:
        # 1) the noisy target patch x_t, and
        # 2) the adjacent conditioning patch shot_far.
        in_channels=2,

        # Base number of feature channels in the U-Net.
        num_channels=64,

        # The network directly predicts one clean seismic target channel x_0.
        out_channels=1,

        # Channel multipliers for the five U-Net resolution scales:
        # 64, 128, 256, 512, and 1024 channels.
        channel_mult=(1, 2, 4, 8, 16),

        # Number of residual blocks at each U-Net resolution scale.
        num_res_blocks=2,

        # Number of attention heads in each self-attention block.
        num_heads=4,

        # A value of -1 makes the decoder use the same number of attention
        # heads as the encoder.
        num_heads_upsample=-1,

        # Apply attention at downsampling factors of 8 and 16.
        attention_resolutions=(8, 16),

        # Dropout probability in the residual blocks.
        dropout=0.0,

        # The model does not predict the reverse-process variance.
        learn_sigma=False,

        # Use the fixed-large variance when learn_sigma is False.
        sigma_small=False,

        # Class conditioning is not used for seismic reconstruction.
        class_cond=False,

        # Total number of diffusion timesteps used during training.
        diffusion_steps=1000,

        # Cosine noise schedule used to define the forward diffusion process.
        noise_schedule="cosine",

        # Default timestep subset used during DDIM inference.
        timestep_respacing="ddim10",

        # Use the MSE training objective rather than the KL objective.
        use_kl=False,

        # Directly predict the clean target patch x_0 instead of noise epsilon.
        predict_xstart=True,

        # Rescale model timesteps to match the original 0--1000 convention.
        rescale_timesteps=True,

        # Do not apply the rescaled MSE treatment for learned variances.
        rescale_learned_sigmas=False,

        # Gradient checkpointing is disabled by default.
        use_checkpoint=False,

        # Use timestep-dependent scale and shift parameters in residual-block
        # normalization, corresponding to adaptive group normalization.
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    """
    Construct the conditional U-Net and its Gaussian diffusion process.

    This function provides a unified interface for creating the two principal
    components of the proposed framework:

    1. A conditional U-Net that receives the noisy target patch x_t and the
       adjacent conditioning patch shot_far.
    2. A Gaussian diffusion process that defines forward noising, training
       objectives, and the selected reverse-sampling timestep schedule.

    Parameters
    ----------
    class_cond : bool
        Whether to enable class-conditional embeddings. This is False in the
        current seismic reconstruction framework.

    learn_sigma : bool
        Whether the U-Net additionally predicts the reverse-process variance.

    sigma_small : bool
        Whether to use the fixed-small variance when learn_sigma is False.

    in_channels : int
        Number of input channels passed into UNetModel. In the current
        framework, this is 2 because x_t and shot_far are concatenated.

    num_channels : int
        Base number of U-Net feature channels.

    out_channels : int
        Number of output channels predicted by the U-Net.

    channel_mult : tuple
        Channel multiplier at each U-Net resolution scale.

    num_res_blocks : int
        Number of residual blocks per resolution scale.

    num_heads : int
        Number of attention heads in encoder and middle attention blocks.

    num_heads_upsample : int
        Number of attention heads in decoder attention blocks.

    attention_resolutions : tuple
        Downsampling factors at which self-attention is applied.

    dropout : float
        Dropout probability in residual blocks.

    diffusion_steps : int
        Number of timesteps in the original diffusion schedule.

    noise_schedule : str
        Name of the beta schedule, such as "linear" or "cosine".

    timestep_respacing
        Specification of the timestep subset used by SpacedDiffusion. For
        example, "ddim10" selects 10 DDIM sampling timesteps.

    use_kl : bool
        Whether to optimize a rescaled variational lower-bound objective.

    predict_xstart : bool
        Whether the model directly predicts the clean target patch x_0.

    rescale_timesteps : bool
        Whether model timesteps are rescaled to the original 0--1000 range.

    rescale_learned_sigmas : bool
        Whether to use the rescaled MSE objective when variance is learned.

    use_checkpoint : bool
        Whether to enable gradient checkpointing in the U-Net.

    use_scale_shift_norm : bool
        Whether timestep embeddings control scale and shift parameters in
        residual-block normalization.

    Returns
    -------
    tuple
        The constructed UNetModel and SpacedDiffusion objects.
    """

    # Construct the conditional U-Net used to predict the clean seismic patch.
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )

    # Construct the diffusion process used for training and sampling.
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Construct the conditional U-Net used by the diffusion model.

    The U-Net receives the channel-wise concatenation of the noisy target patch
    x_t and the adjacent conditioning patch shot_far. With the default
    configuration, in_channels is therefore 2 and out_channels is 1.

    Parameters
    ----------
    in_channels : int
        Number of channels in the concatenated U-Net input.

    num_channels : int
        Base feature-channel width of the U-Net.

    out_channels : int
        Number of output channels predicted by the U-Net.

    channel_mult : tuple
        Feature-channel multipliers at successive resolution scales.

    num_res_blocks : int
        Number of residual blocks at each resolution scale.

    learn_sigma : bool
        Retained for consistency with the diffusion configuration. The output
        channel count is supplied separately through out_channels.

    class_cond : bool
        Whether to enable class-label conditioning.

    use_checkpoint : bool
        Whether to use gradient checkpointing.

    attention_resolutions : tuple
        Downsampling factors at which self-attention is inserted.

    num_heads : int
        Number of attention heads in encoder and middle blocks.

    num_heads_upsample : int
        Number of attention heads in decoder blocks.

    use_scale_shift_norm : bool
        Whether timestep embeddings adaptively control normalization through
        learned scale and shift parameters.

    dropout : float
        Dropout probability in residual blocks.

    Returns
    -------
    UNetModel
        Configured conditional U-Net.
    """

    return UNetModel(
        # Total number of channels after concatenating x_t and shot_far.
        in_channels=in_channels,

        # Base number of U-Net feature channels.
        model_channels=num_channels,

        # Number of predicted channels, normally one clean seismic channel.
        out_channels=out_channels,

        # Number of residual blocks in each resolution scale.
        num_res_blocks=num_res_blocks,

        # Resolution scales at which global spatial attention is applied.
        attention_resolutions=attention_resolutions,

        # Dropout probability used inside residual blocks.
        dropout=dropout,

        # Feature-channel multipliers defining the encoder and decoder scales.
        channel_mult=channel_mult,

        # Enable class embeddings only when class_cond is True.
        num_classes=(NUM_CLASSES if class_cond else None),

        # Enable or disable gradient checkpointing.
        use_checkpoint=use_checkpoint,

        # Number of attention heads in encoder and middle blocks.
        num_heads=num_heads,

        # Number of attention heads in decoder blocks.
        num_heads_upsample=num_heads_upsample,

        # Use timestep-conditioned scale-shift normalization.
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Construct the Gaussian diffusion process used for training and sampling.

    The function first generates the full beta schedule and then creates a
    SpacedDiffusion object containing the subset of timesteps specified by
    timestep_respacing. The original training schedule may contain 1000
    timesteps, while DDIM inference can use a much smaller selected subset.

    Parameters
    ----------
    steps : int, optional
        Number of timesteps in the original forward diffusion schedule.

    learn_sigma : bool, optional
        Whether the model predicts values used to determine reverse-process
        variance.

    sigma_small : bool, optional
        Whether to use the fixed-small reverse variance when learn_sigma is
        False.

    noise_schedule : str, optional
        Name of the beta schedule used in the forward diffusion process.

    use_kl : bool, optional
        Whether to use the rescaled variational lower-bound loss.

    predict_xstart : bool, optional
        Whether the network predicts the clean seismic target x_0 directly.
        When False, the network predicts the added Gaussian noise epsilon.

    rescale_timesteps : bool, optional
        Whether timesteps passed to the model are rescaled to a 0--1000 range.

    rescale_learned_sigmas : bool, optional
        Whether to use rescaled MSE when the variance is learned.

    timestep_respacing
        Timestep-subset configuration used by SpacedDiffusion. Examples include
        an empty value for the complete schedule and "ddim10" for ten DDIM
        sampling steps.

    Returns
    -------
    SpacedDiffusion
        Configured diffusion process.
    """

    # Generate beta_t for every timestep in the original diffusion schedule.
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Select the training objective.
    if use_kl:
        # Optimize a rescaled variational lower-bound objective.
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        # Use rescaled MSE when learning the reverse-process variance.
        loss_type = gd.LossType.RESCALED_MSE
    else:
        # Use the standard MSE objective adopted in the manuscript.
        loss_type = gd.LossType.MSE

    # Use every original diffusion timestep when no respacing is specified.
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        # Select the subset of original diffusion timesteps used by the
        # respaced diffusion process.
        use_timesteps=space_timesteps(steps, timestep_respacing),

        # Full beta schedule defining the original forward diffusion process.
        betas=betas,

        # Directly predict x_0 when predict_xstart is True; otherwise predict
        # the Gaussian noise epsilon.
        model_mean_type=(
            gd.ModelMeanType.EPSILON
            if not predict_xstart
            else gd.ModelMeanType.START_X
        ),

        # Select either a fixed reverse variance or a learned variance range.
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),

        # Loss function used to train the diffusion model.
        loss_type=loss_type,

        # Control whether the timestep indices are rescaled before being
        # passed into the U-Net timestep-embedding module.
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add every entry in a default-configuration dictionary to an argument
    parser as a command-line option.

    Boolean values are parsed using str2bool because argparse does not
    correctly interpret strings such as "False" through the built-in bool
    constructor.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to which the options are added.

    default_dict : dict
        Mapping from argument names to their default values.
    """

    for k, v in default_dict.items():
        # Infer the command-line conversion function from the default value.
        v_type = type(v)

        # Arguments with a None default are interpreted as strings.
        if v is None:
            v_type = str

        # Boolean strings are converted using the custom str2bool function.
        elif isinstance(v, bool):
            v_type = str2bool

        # Register the command-line argument using the dictionary key.
        parser.add_argument(
            f"--{k}",
            default=v,
            type=v_type,
        )


def args_to_dict(args, keys):
    """
    Extract selected command-line arguments into a dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    keys
        Names of the arguments to extract.

    Returns
    -------
    dict
        Dictionary containing the selected argument values.
    """

    return {
        k: getattr(args, k)
        for k in keys
    }


def str2bool(v):
    """
    Convert common command-line Boolean strings to Python Boolean values.

    Accepted true values are:
    "yes", "true", "t", "y", and "1".

    Accepted false values are:
    "no", "false", "f", "n", and "0".

    Parameters
    ----------
    v
        Boolean value or string representation of a Boolean value.

    Returns
    -------
    bool
        Parsed Boolean value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the supplied value is not a recognized Boolean representation.

    Reference
    ---------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    # Return existing Boolean values without additional conversion.
    if isinstance(v, bool):
        return v

    # Parse commonly used true-value strings.
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    # Parse commonly used false-value strings.
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    # Reject unrecognized command-line values.
    else:
        raise argparse.ArgumentTypeError("boolean value expected")