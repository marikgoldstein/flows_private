"""
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# noinspection PyProtectedMember
from torch.nn.init import _calculate_fan_in_and_fan_out

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.sigmoid(x) * x


def group_norm(num_groups, out_ch):
    return nn.GroupNorm(num_groups=num_groups, num_channels=out_ch, eps=1e-6, affine=True)

def upsample(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module("up_nn", nn.Upsample(scale_factor=2, mode="nearest"))
    if with_conv:
        up.add_module("up_conv", conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up

def downsample(in_ch, with_conv):
    if with_conv:
        down = conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    else:
        down = nn.AvgPool2d(2, 2)
    return down

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        temb_ch,
        out_ch=None,
        conv_shortcut=False,
        dropout=0.0,
        normalize=group_norm,
        act=Swish(),
    ):
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act

        self.temb_proj = dense(temb_ch, out_ch)
        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, out_ch)
        self.norm2 = normalize(out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = conv2d(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, out_ch)
            else:
                self.shortcut = conv2d(in_ch, out_ch, kernel_size=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        # forward conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # add in timestep embedding
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        # forward conv2
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h


#class SelfAttention(nn.Module):
class AttnBlock(nn.Module):
    """
    copied modified from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py#L29
    copied modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66
    """

    def __init__(self, num_groups, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attn_k = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attn_v = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, init_scale=0.0
        )
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.GroupNorm(num_groups, in_channels)

    # noinspection PyUnusedLocal
    def forward(self, x, temp=None):
        """t is not used"""
        _, C, H, W = x.size()

        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, H * W)
        k = self.attn_k(h).view(-1, C, H * W)
        v = self.attn_v(h).view(-1, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(-1, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out", "fan_avg"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")


def dense(in_channels, out_channels, use_bias=True, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels, bias=use_bias)
    variance_scaling_init_(lin.weight, scale=init_scale)
    if use_bias:
        nn.init.zeros_(lin.bias)
    return lin



def conv2d(
    in_planes,
    out_planes,
    kernel_size=(3, 3),
    stride=1,
    dilation=1,
    padding=1,
    bias=True,
    padding_mode="zeros",
    init_scale=1.0,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode,
    )
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def get_sinusoidal_positional_embedding(
    timesteps: torch.LongTensor, embedding_dim: int
):
    """
    Copied and modified from
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    From Fairseq in
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.size()) == 1
    #timesteps = timesteps.to(torch.get_default_dtype())
    #device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float).type_as(timesteps) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [timesteps.size(0), embedding_dim]
    return emb



class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=Swish()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
            act,
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb



class Base2FourierFeaturesModule(nn.Module):
    def __init__(self,n_fourier):
        super().__init__()
        self.n_fourier = n_fourier
       
    def forward(self, x):
        
        B, C, H, W = x.shape

        fourier_features = []
        x_pi = x * np.pi
        freqs = [6.0, 7.0, 8.0]
        assert len(freqs)==self.n_fourier
        for n in freqs:
            x_2pown_pi = (x_pi * (2**n))
            s = x_2pown_pi.sin()
            c = x_2pown_pi.cos()
            fourier_features_n = torch.cat([s, c], dim=1)
            assert fourier_features_n.shape == (B, 2*C, H, W)
            fourier_features.append(fourier_features_n)
            
        fourier_features = torch.cat(fourier_features,dim=1)
        assert fourier_features.shape==(B, 2*C*self.n_fourier, H, W), fourier_features.shape

        x = torch.cat([x, fourier_features], dim=1)
        assert x.shape==(B, C + 2*C*self.n_fourier, H , W)
        
        return x 

def channels_torch_to_jax(x):
    return x.permute(0,2,3,1)
    
def channels_jax_to_torch(x):
    return x.permute(0,3,1,2)


class ResnetBlock(nn.Module):

    def __init__(self,in_ch, out_ch=None, temb_ch=32, num_groups = 32, pdrop=0.0):
        super().__init__()
        self.in_ch = in_ch
        out_ch = out_ch if out_ch is not None else in_ch
        self.out_ch = out_ch
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.normalize1 = nn.GroupNorm(self.num_groups, self.in_ch)
        self.normalize2 = nn.GroupNorm(self.num_groups, self.out_ch)
        self.conv1 = conv2d(in_ch, out_ch, 3, 1, init_scale=1.0)
        self.cond_proj = dense(temb_ch, out_ch, use_bias=False, init_scale=0.0)
        self.conv2 = conv2d(out_ch, out_ch, 3, 1, init_scale=0.0) 

        if self.in_ch != self.out_ch:
            self.nin_shortcut = dense(in_ch, out_ch, use_bias=True, init_scale=1.0)
   
    # TODO check normalize dims
    def forward(self, x, cond):

        B, C, _, _ = x.shape
        #assert C==self.in_ch # could be false cause of residual inputs in up block
        h = x
        h = Swish()(self.normalize1(h))
        h = self.conv1(h)

        if cond is not None:
            assert cond.shape[0] == B and len(cond.shape)==2
            h = h + self.cond_proj(cond)[:, :, None, None]

        h = Swish()(self.normalize2(h))
        h = nn.Dropout2d(p=self.pdrop)(h)
        h = self.conv2(h)

        if C != self.out_ch:
            x = channels_torch_to_jax(x)
            x = self.nin_shortcut(x)
            x = channels_jax_to_torch(x)

        assert x.shape == h.shape
        h = x + h
        return h


def get_scoreunet(conf):
 
    dset = conf.dataset
    assert dset == 'cifar'
    return ScoreUNet(
        n_embd=128,
        n_layers = 16,
        num_groups = 32,
        pdrop = conf.bigunet_dropout,
        use_fourier = conf.use_bigunet_fourier,
        with_attention = True,
        input_channels = conf.C,
        input_height = conf.H
    )

class ScoreUNet(nn.Module):
    def __init__(
        self, 
        n_embd=256,
        n_layers = 32,
        num_groups = 32,
        pdrop = .05,
        use_fourier = True,
        with_attention = True,
        input_channels = 3,
        input_height = 32,
    ):
        super().__init__()
        self.input_height = input_height
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.pdrop = pdrop
        self.use_fourier = use_fourier
        self.with_attention = with_attention
        self.num_groups = num_groups
        self.in_ch = input_channels 
        
        # FOURIER
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.n_fourier = 3
            self.base2fourier = Base2FourierFeaturesModule(n_fourier=self.n_fourier)
            self.in_ch_fourier = self.in_ch + self.in_ch * self.n_fourier * 2


        in_ch = self.in_ch_fourier if self.use_fourier else self.in_ch

        temb_ch = in_ch * 4

        # Timestep embedding
        self.temb_net = TimestepEmbedding(in_ch, temb_ch, temb_ch)
        
        # In Block
        self.conv_in = conv2d(in_ch, self.n_embd, kernel_size=(3,3), stride=(1,1))

        res_down_args = {'in_ch': self.n_embd, 'temb_ch': temb_ch, 'pdrop':pdrop, 'num_groups': self.num_groups}
        res_mid_args = res_down_args
        res_up_args = {'in_ch': self.n_embd * 2, 'temb_ch': temb_ch, 'pdrop':pdrop, 'num_groups': self.num_groups, 'out_ch': self.n_embd}
        #attn_args = { make num heads = 1}
        attn_down_args = {'num_groups': self.num_groups, "in_channels": self.n_embd}
        attn_mid_args = attn_down_args
        attn_up_args = {'num_groups': self.num_groups, "in_channels": self.n_embd}

        # Down and Up Blocks
        
        self.DownBlocks = nn.ModuleList([ResnetBlock(**res_down_args) for i in range(self.n_layers)])
        self.UpBlocks = nn.ModuleList([ResnetBlock(**res_up_args) for i in range(self.n_layers)])
        
        if self.with_attention:

            self.DownAttns = nn.ModuleList([AttnBlock(**attn_down_args) for i in range(self.n_layers)])
            self.UpAttns = nn.ModuleList([AttnBlock(**attn_up_args) for i in range(self.n_layers)])

        self.mid_block_1 = ResnetBlock(**res_mid_args)
        self.mid_block_2 = AttnBlock(**attn_mid_args)
        self.mid_block_3 = ResnetBlock(**res_mid_args)


        self.out_block = nn.Sequential(nn.GroupNorm(self.num_groups, self.n_embd), Swish(), conv2d(self.n_embd, self.in_ch, 3, 1, init_scale=0.0))


    def forward(self, u ,t):


        B, C, H, W = u.size()


        # embed t. could also optionally concat conditioning to t
        h = u
        cond = self.temb_net(t) 
               
        # in layer
        h = self.conv_in(self.base2fourier(h) if self.use_fourier else h)
        
        hs = [h]

        # Down
        for i_block in range(self.n_layers):
            h = self.DownBlocks[i_block](hs[-1], cond)
            if self.with_attention:
                h = self.DownAttns[i_block](h)
            hs.append(h)

        # Middle
        h = hs[-1]
        h = self.mid_block_1(h, cond)
        h = self.mid_block_2(h)
        h = self.mid_block_3(h, cond)
        
        # Up
        for i_block in range(0, self.n_layers):
            h1 = h
            h2 = hs[-(i_block+1)]
            h_in = torch.cat([h1,h2],dim=1)
            h = self.UpBlocks[i_block](h_in, cond)
            if self.with_attention:
                h = self.UpAttns[i_block](h)

        # out
        eps_pred = u + self.out_block(h)

        return eps_pred



