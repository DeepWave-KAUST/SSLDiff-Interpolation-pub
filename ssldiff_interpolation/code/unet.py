"""
U-Net backbone for the conditional seismic diffusion model.

The network predicts a clean target seismic patch from two inputs:
1. ``inp``: the noisy target patch x_t at the current diffusion timestep.
2. ``shot_far``: the adjacent conditioning patch y extracted with a one-trace shift.

The two patches are concatenated along the channel dimension before entering
the encoder. Diffusion-timestep embeddings condition every residual block.
All original variable, module, and parameter names are preserved so that
existing checkpoints remain compatible.
"""

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)

class TimeEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        # Keep the sinusoidal embedding on the same device as the timesteps.
        device = x.device

        # Split the embedding dimension equally between sine and cosine terms.
        half_dim = self.dim // 2
        # Construct the logarithmically spaced frequencies used by the
        # sinusoidal diffusion-timestep embedding.
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)

        # Form one frequency response for every timestep in the batch.
        emb = th.outer(x * self.scale, emb)

        # Concatenate sine and cosine components to obtain shape (N, dim).
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepBlock(nn.Module):
    """
    Base class for modules whose forward method also receives a timestep embedding.

    The current conditional seismic U-Net uses only the diffusion-timestep
    embedding ``time_emb`` in these blocks.
    """

    @abstractmethod
    def forward(self, x, time_emb):
        """
        Apply the module to ``x`` conditioned on ``time_emb``.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential container that forwards ``time_emb`` to child TimestepBlock modules.

    Standard layers receive only the feature tensor ``x``.
    """

    def forward(self, x, time_emb):
        # Pass the timestep embedding only to layers that explicitly support it.
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param time_emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        time_emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.time_emb_channels = time_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # time embedding
        self.time_emb_layers = nn.Sequential(
            SiLU(),
            linear(
                time_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, time_emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param time_emb: an [N x time_emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, time_emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, time_emb):
        # Transform the input feature map through normalization, activation,
        # and convolution.
        h = self.in_layers(x)

        # Project the timestep embedding to the channel dimension required by
        # this residual block and match the feature dtype.
        time_emb_out = self.time_emb_layers(time_emb).type(h.dtype)
        # Add singleton spatial dimensions so that the timestep embedding
        # broadcasts over time and offset coordinates.
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # Adaptive scale-shift normalization injects the diffusion
            # timestep into the residual feature transformation.
            scale, shift = th.chunk(time_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + time_emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        # Flatten all spatial dimensions so attention operates over every
        # time-offset position in the seismic feature map.
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        # Generate query, key, and value tensors and distribute the channel
        # dimension over the configured attention heads.
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    Conditional U-Net with self-attention and diffusion-timestep embedding.

    ``in_channels`` refers to the channel count after concatenating ``inp`` and
    ``shot_far``. In the manuscript configuration, the noisy target and the
    conditioning patch each contain one channel, yielding a two-channel input.

    :param in_channels: channels in the concatenated input tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        time_emb_scale=1.0,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # Spatial dimensions must be divisible by this factor after all
        # encoder downsampling operations.
        self.padder_size = 2 ** len(channel_mult)

        # Project the scalar diffusion timestep to the conditioning dimension
        # used by all residual blocks.
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Initial convolution maps the concatenated noisy-target and
        # conditioning channels into the base feature dimension.
        self.inp = conv_nd(dims, in_channels, model_channels, 3, padding=1)

        # Encoder blocks progressively reduce spatial resolution while
        # increasing the number of feature channels.
        self.downs = nn.ModuleList([])
        encoder_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.downs.append(TimestepEmbedSequential(*layers))
                encoder_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                encoder_channels.append(ch)
                ds *= 2

        # Bottleneck blocks operate at the lowest spatial resolution and
        # include self-attention for long-range seismic-event interactions.
        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Decoder blocks restore spatial resolution and fuse encoder
        # information through symmetric skip connections.
        self.ups = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + encoder_channels.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(*layers))

        # Final normalized projection maps decoder features to the requested
        # diffusion-model output channels.
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.downs.apply(convert_module_to_f16)
        self.middle.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.downs.apply(convert_module_to_f32)
        self.middle.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.downs.parameters()).dtype

    def forward(self, inp, shot_far, timesteps, y=None):
        """
        Predict the clean target seismic patch from a noisy target and condition.

        :param inp: an [N x C x H x W] tensor containing the noisy target
            patch x_t at the sampled diffusion timestep.
        :param shot_far: an [N x C x H x W] tensor containing the adjacent
            conditioning patch y. In this project, it provides the recorded
            offset context used to reconstruct the one-trace-shifted target.
        :param timesteps: an [N] tensor containing the diffusion timestep of
            each sample in the batch.
        :param y: an [N] tensor of class labels when class conditioning is
            enabled. It is unused in the current seismic reconstruction setup.
        :return: an [N x out_channels x H x W] tensor containing the model
            prediction, typically the clean target patch x_0.
        """
        # Preserve the original target-patch size so padding can be removed
        # from the network output.
        b, c, h, w = inp.shape

        # Concatenate the noisy target patch x_t and adjacent conditioning
        # patch shot_far along the channel dimension.
        x = th.cat([inp, shot_far], dim=1)

        # Replicate-pad the time and offset dimensions when required by the
        # sequence of encoder downsampling operations.
        x = self.check_image_size(x)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Encode the sampled diffusion timestep for residual-block conditioning.
        time_emb = self.time_embed(timesteps)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            time_emb = time_emb + self.label_emb(y)

        # Store encoder features for the symmetric decoder skip connections.
        skips = []

        # Match the precision used by the U-Net body, then apply the input
        # projection to the concatenated seismic patches.
        x = x.type(self.inner_dtype)
        x = self.inp(x)
        skips.append(x)

        # Run the encoder and retain every intermediate feature tensor needed
        # by the decoder.
        for module in self.downs:
            x = module(x, time_emb)
            skips.append(x)

        # Process the lowest-resolution representation.
        x = self.middle(x, time_emb)

        # Concatenate each decoder feature map with its matching encoder
        # feature map before applying the corresponding up block.
        for module in self.ups:
            cat_in = th.cat([x, skips.pop()], dim=1)
            x = module(cat_in, time_emb)

        # Restore the input precision, predict the diffusion target, and crop
        # away any padding introduced by check_image_size().
        x = x.type(inp.dtype)
        x = self.out(x)
        return x[:, :, :h, :w]

    def check_image_size(self, x):
        """
        Pad the time and offset dimensions to sizes compatible with the U-Net.

        Replicate padding is applied only to the bottom and right boundaries.
        The forward method crops the final prediction back to the original
        target-patch dimensions.
        """
        _, _, h, w = x.size()

        # Determine the minimum padding needed for repeated factor-of-two
        # downsampling and upsampling.
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size

        # Replicate boundary values instead of introducing artificial zeros.
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x