"""
## Diff and License
Almost all codes came from 'jik876/hifi-gan' (MIT License by Jungil Kong).
Plus, some modification for iSTFTNet from 'rishikksh20/iSTFTNet-pytorch' (Apache License 2.0).
Midification in:
  - conv_post
    - output channel size
    - reflection_pad    
  - STFT output
    - mag/phase with exp/sin
Other simplification from 'tarepan/iSTFTNet-pytorch' (MIT License by Tarepan)
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, ConvTranspose2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from einops import rearrange, reduce, repeat, unpack, pack
from einops.layers.torch import Rearrange, EinMix

from utils import Attend

LRELU_SLOPE = 0.1


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class ResBlock1(torch.nn.Module):
    """Big ResBlock."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        """
        Args:
            channels - Constant size of channel dimension (frequency dimension)
            kernel_size - Conv kernel size
            dilation - Dilation factor of Res1/Res2/Res3's 1st Conv (2nd Conv is dilation=1)
        """
        super().__init__()

        # 1st Conv of Res1/Res2/Res3
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        # 2nd Conv of Res1/Res2/Res3
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            # Inner-most Res block: Res(LReLU-Conv-LReLU-Conv)
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2D(torch.nn.Module):
    """Big ResBlock."""

    def __init__(self, channels, kernel_size=(3, 3), dilation=(1, 3, 5)):
        """
        Args:
            channels - Constant size of channel dimension (frequency dimension)
            kernel_size - Conv kernel size
            dilation - Dilation factor of Res1/Res2/Res3's 1st Conv (2nd Conv is dilation=1)
        """
        super().__init__()

        # 1st Conv of Res1/Res2/Res3
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        channels,
                        2 * channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        # 2nd Conv of Res1/Res2/Res3
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        2 * channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            # Inner-most Res block: Res(LReLU-Conv-LReLU-Conv)
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """Small ResBlock."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        """
        Args:
            channels - Constant size of channel dimension (frequency dimension)
            kernel_size - Conv kernel size
            dilation - Dilation factor of Res1/Res2's Conv
        """
        super().__init__()

        # The Conv of Res1/Res2
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            # Inner-most Res block : Res(LReLU-Conv)
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class ResBlock2D2(torch.nn.Module):
    """Small ResBlock."""

    def __init__(self, channels, kernel_size=(3, 3), dilation=(1, 3)):
        """
        Args:
            channels - Constant size of channel dimension (frequency dimension)
            kernel_size - Conv kernel size
            dilation - Dilation factor of Res1/Res2's Conv
        """
        super().__init__()

        # The Conv of Res1/Res2
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv2d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            # Inner-most Res block : Res(LReLU-Conv)
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, flash=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = Attend(flash=flash, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None, mask=None, rotary_emb=None, attn_bias=None):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask=mask, attn_bias=attn_bias)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        attn_flash=True,
        ff_dropout=0.0,
        conv_dropout=0.0,
        conv_causal=False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            flash=attn_flash,
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, rotary_emb=None, attn_bias=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask, rotary_emb=rotary_emb, attn_bias=attn_bias) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # PreConv
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # MainStack
        ## Upsampling
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # up = ConvTranspose1d(
            #     h.upsample_initial_channel // (2**i),
            #     h.upsample_initial_channel // (2 ** (i + 1)),
            #     k,
            #     u,
            #     padding=(k - u) // 2,
            # )
            up = ConvTranspose1d(
                h.upsample_initial_channel // (2**i),
                h.upsample_initial_channel // (2 ** (i + 1)),
                k,
                u,
                padding=(u // 2 + u % 2),
                output_padding=u % 2,
            )
            self.ups.append(weight_norm(up))
        self.ups.apply(init_weights)
        ## MRF
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # PostConv :: (B, F, T) -> (B, F=2+nfft, T)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(Conv1d(ch, h.gen_istft_n_fft + 2, 7, 1, padding=3))
        self.conv_post.apply(init_weights)
        self._center = h.gen_istft_n_fft // 2 + 1

    def forward(self, x):
        """
        Returns:
            spec  :: (B, F, T) - Linear amplitude (TODO: power? amplitude?)
            phase :: (B, F, T) - Phase
        """
        # PreConv
        x = self.conv_pre(x)

        # Stack of "UpSampling + MRF"
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # PostConv
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)

        # To STFT parameters :: (B, F=2f, T) -> (B, F=f, T)
        spec = torch.exp(x[:, : self._center, :])
        phase = torch.sin(x[:, self._center :, :])

        return spec, phase

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Generator2D(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels_time = len(h.resblock_time_kernel_sizes)
        self.num_kernels_freq = len(h.resblock_freq_kernel_sizes)
        self.num_upsamples_time = len(h.upsample_time_rates)
        self.num_upsamples_freq = len(h.upsample_freq_rates)

        # 1D #

        # PreConv
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel[0], 7, 1, padding=3)
        )

        # MainStack
        ## Upsampling
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(h.upsample_time_rates, h.upsample_time_kernel_sizes)
        ):
            up = ConvTranspose1d(
                h.upsample_initial_channel[0] // (2**i),
                h.upsample_initial_channel[0] // (2 ** (i + 1)),
                k,
                u,
                padding=(k - u) // 2,
            )
            self.ups.append(weight_norm(up))
        self.ups.apply(init_weights)
        ## MRF
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel[0] // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_time_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # PostConv :: (B, F, T) -> (B, F=2+nfft, T)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(Conv1d(ch, ch, 7, 1, padding=3))
        self.conv_post.apply(init_weights)
        self._center = h.gen_istft_n_fft // 2 + 1

        # 2D #

        # PreConv
        # self.conv_pre2D = weight_norm(
        #     Conv2d(ch // 2, h.upsample_initial_channel[1], (3, 3), 1, padding=3)
        # )
        self.conv_pre2D = weight_norm(
            Conv2d(ch // 2, h.upsample_initial_channel[1], (3, 3), padding=(4, 1))
        )

        # MainStack
        ## Upsampling
        self.ups2D = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(h.upsample_freq_rates, h.upsample_freq_kernel_sizes)
        ):
            up = ConvTranspose2d(
                h.upsample_initial_channel[1] // (2**i),
                h.upsample_initial_channel[1] // (2 ** (i + 1)),
                k,
                u,
                padding=(k[1] - u) // 2,
                output_padding=1,
            )
            self.ups2D.append(weight_norm(up))
        self.ups2D.apply(init_weights)
        ## MRF
        resblock2D = ResBlock2D if h.resblock == "1" else ResBlock2D2
        self.resblocks2D = nn.ModuleList()
        for i in range(len(self.ups)):
            # ch = h.upsample_initial_channel[1] // (2 ** (i + 1))
            ch = h.upsample_initial_channel[1]
            for j, (k, d) in enumerate(
                zip(h.resblock_freq_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks2D.append(resblock2D(ch, k, d))

    def forward(self, x):
        """
        Returns:
            spec  :: (B, F, T) - Linear amplitude (TODO: power? amplitude?)
            phase :: (B, F, T) - Phase
        """
        # 1D #
        # PreConv
        print("Mel spec", x.shape)
        x = self.conv_pre(x)
        print("Input 1D", x.shape)

        # Stack of "UpSampling + MRF"
        for i in range(self.num_upsamples_time):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels_time):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels_time + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels_time + j](x)
            x = xs / self.num_kernels_time
        print("Bloc 1D", x.shape)

        # PostConv
        x = F.leaky_relu(x)
        print("relu", x.shape)
        # x = self.reflection_pad(x)
        x = self.conv_post(x)
        print("conv post", x.shape)
        x = torch.reshape(x, [x.shape[0], x.shape[1] // 2, 2, x.shape[2]])
        print("1D 2D", x.shape)

        # 2D #
        # PreConv
        x = self.conv_pre2D(x)
        print("input 2D", x.shape)
        xs = None
        for j in range(self.num_kernels_freq):
            if xs is None:
                xs = self.resblocks2D[i * self.num_kernels_freq + j](x)
            else:
                xs += self.resblocks2D[i * self.num_kernels_freq + j](x)
        x = xs / self.num_kernels_freq
        print("block 2D", x.shape)

        for i in range(self.num_upsamples_freq):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups2D[i](x)
            print("upsample 2D", x.shape)

        print("upsample 2D", x.shape)

        # To STFT parameters :: (B, F=2f, T) -> (B, F=f, T)
        spec = torch.exp(x[:, 0, :, :])
        phase = torch.sin(x[:, 1, :, :])

        return spec, phase

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.ups2D:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.resblocks2D:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        """
        Args:
            y     - real
            y_hat - generated
        Returns:
            y_d_rs  :: List[] - MPD(y)
            y_d_gs  :: List[] - MPD(y_hat)
            fmap_rs :: List[] - feature map of MPD(y)
            fmap_gs :: List[] - feature map of MPD(y_hat)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Returns:
        loss - Sum of real loss and generated loss
    """
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss

    return loss


def generator_loss(disc_outputs):
    """
    Returns:
        loss - Sum of generator loss
    """
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)

    return loss
