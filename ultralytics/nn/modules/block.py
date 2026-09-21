# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "SPPFCSPC",
    "CARAFE",
    "Dy_Sample",
    "SPPCSPC",
    "SPPFEC",
    "LightweightSPPFELAN",
    "SPPELAN_CM",
    # "EfficientEnhancedCSAttention",
    "HybridSimAMCSAttention"
    # "LSTMInspiredGatedCSAttention",
    # "BiLSTMGatedSCAttention",

)

from ..dysample import Dy_Sample


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class SPPFCSPC(nn.Module):
    # 本代码由YOLOAir目标检测交流群 心动 大佬贡献
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class CARAFE(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, in_tensor):
        if isinstance(in_tensor,list):
            in_tensor=torch.cat(in_tensor,dim=1)
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out = self.out(out_tensor)
        return out

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class SPPFEC(nn.Module):
    def __init__(self, c1, c2, e=0.25, k=5):
        """
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param e: 通道压缩比例
        :param k: SPP 的池化核大小
        """
        super(SPPFEC, self).__init__()
        c_ = int(c2 * e)  # reduced channels

        # SPPELAN 部分
        self.spp = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 多尺度池化
        self.depthwise = nn.Conv2d(c1, c_, kernel_size=3, stride=1, padding=1, groups=c1)  # 深度卷积
        self.pointwise = nn.Conv2d(c_, c_, kernel_size=1, stride=1)  # 逐点卷积

        # CSP 分支结构
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)  # 逐点卷积代替标准卷积
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_, c_, 3, 1, padding=1, groups=c_)  # 深度卷积
        self.cv4 = nn.Conv2d(c_, c_, 1, 1)

        # 多尺度融合
        self.fuse_conv = nn.Conv2d(4 * c_, c_, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(c_ + c_, c2, kernel_size=1, stride=1)  # 拼接后压缩通道数

    def forward(self, x):
        # SPPELAN 部分
        x_spp = self.spp(x)  # 多尺度池化
        x_depth = self.depthwise(x_spp)  # 深度卷积
        x_point = self.pointwise(x_depth)  # 逐点卷积

        # CSP 分支
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.spp(x1)
        x3 = self.spp(x2)
        x4 = self.spp(x3)

        # 特征融合
        y1 = self.fuse_conv(torch.cat((x1, x2, x3, x4), dim=1))  # 多尺度特征
        y2 = self.cv2(x)  # CSP 分支 2
        out = self.final_conv(torch.cat((y1, y2), dim=1))  # 融合后输出
        return out



import torch
import torch.nn as nn

class LightweightSPPFELAN(nn.Module):
    def __init__(self, c1, c2, e=0.25, k=3):
        """
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param e: 通道压缩比例
        :param k: SPP 的池化核大小
        """
        super(LightweightSPPFELAN, self).__init__()
        c_ = int(c2 * e)  # 压缩后的通道数

        # SPP 部分
        self.spp = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 简化多尺度池化
        self.depthwise = nn.Conv2d(c1, c_, kernel_size=3, stride=1, padding=1, groups=c1)  # 深度卷积
        self.pointwise = nn.Conv2d(c_, c_, kernel_size=1, stride=1)  # 逐点卷积

        # 简化的 CSP 分支结构
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)  # 减少通道数
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)  # 简化 CSP 的并行分支
        self.cv3 = nn.Conv2d(c_, c_, 1, 1)  # 将深度卷积替换为 1x1 卷积

        # 特征融合
        self.fuse_conv = nn.Conv2d(3 * c_, c2, kernel_size=1, stride=1)  # 简化融合层

    def forward(self, x):
        # SPP 部分
        x_spp = self.spp(x)  # 多尺度池化
        x_depth = self.depthwise(x_spp)  # 深度卷积
        x_point = self.pointwise(x_depth)  # 逐点卷积

        # CSP 分支
        x1 = self.cv1(x)
        x2 = self.spp(x1)
        x3 = self.cv3(x2)  # 简化后的分支

        # 特征融合
        out = self.fuse_conv(torch.cat((x_point, x1, x3), dim=1))  # 融合简化后的多分支
        return out


# class SimAM(torch.nn.Module):
#     def __init__(self, channels=None, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()  # Sigmoid 激活函数的实例
#         self.e_lambda = e_lambda  # 正则化项的参数 lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)  # 返回表示模型的字符串，包括 lambda 参数的值
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"  # 返回模块的名称，为 "simam"
#
#     def forward(self, x):
#         # 前向传播函数，对输入 x 进行处理并返回结果
#
#         b, c, h, w = x.size()    # 获取输入张量的形状信息
#         n          = w * h - 1   # 计算总像素数减1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # 计算每个像素与均值的差的平方
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         # 计算激活函数的输入 y，使用 SimAM 公式：x_minus_mu_square / (4 * (均值方差 + 正则化项)) + 0.5
#
#         return x * self.activaton(y)  # 返回经过激活函数后的结果与输入张量 x 的乘积
#

# class SPPELAN_CM(nn.Module):
#     """SPP-ELAN with SimAM Attention."""
#
#     def __init__(self, c1, c2, c3, k=5):
#         """Initializes SPP-ELAN block with convolution, max pooling layers, and SimAM attention."""
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#         self.simam = SimAM(c2)  # Add SimAM attention after convolution
#
#     def forward(self, x):
#         """Forward pass through SPPELAN with SimAM attention."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5
#         return self.simam(y)  # Apply SimAM attention after convolution

# class EfficientEnhancedCSAttention(nn.Module):
#     def __init__(self, c1, c2=None, groups=8, pool_size=4, **kwargs):
#         super().__init__()
#         if c1 % groups != 0:
#             original_groups = groups
#             for g in range(groups, 0, -1):
#                 if c1 % g == 0:
#                     groups = g
#                     break
#             if c1 % groups != 0:
#                 raise ValueError(
#                     f"c1({c1}) cannot be divided by any number from the original groups({original_groups}) down to 1.")
#
#         self.groups = groups
#         self.pool_size = pool_size
#
#         self.pool = nn.AvgPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
#         # 调整了GroupNorm的num_groups，使其在通道数较少时也能工作
#         # YOLO框架中常见的最小通道数是32的倍数，所以32通常是一个安全的选择
#         # 如果c1可能小于32，需要进一步调整
#         self.final_norm = nn.GroupNorm(num_groups=32 if c1 % 32 == 0 else groups, num_channels=c1, eps=1e-6)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # Grouped Channel Attention
#         x_grouped = x.view(b, self.groups, c // self.groups, h, w)
#         # 将维度转换为 (b*h*w*groups, c // groups, 1) 以进行批处理矩阵乘法
#         x_channel_seq = x_grouped.permute(0, 3, 4, 1, 2).reshape(b * h * w * self.groups, c // self.groups, 1)
#
#         q_c, k_c, v_c = x_channel_seq, x_channel_seq, x_channel_seq
#         attn_scores_c = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(c // self.groups)  # 缩放点积
#         attn_weights_c = F.softmax(attn_scores_c, dim=-1)
#         attended_c = torch.bmm(attn_weights_c, v_c)
#         # 将维度恢复为 (b, c, h, w)
#         attended_c = attended_c.reshape(b, h, w, self.groups, c // self.groups).permute(0, 3, 4, 1, 2).reshape(b, c, h, w)
#
#         x_after_channel = x + attended_c  # 残差连接
#
#         # Downsampled Spatial Attention
#         x_pooled = self.pool(x_after_channel)
#         ph, pw = x_pooled.shape[-2:]
#         # 将维度转换为 (b*c, ph*pw, 1)
#         x_spatial_seq = x_pooled.reshape(b * c, ph * pw, 1)
#         q_s, k_s, v_s = x_spatial_seq, x_spatial_seq, x_spatial_seq
#         attn_scores_s = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(ph * pw)  # 缩放点积
#         attn_weights_s = F.softmax(attn_scores_s, dim=-1)
#         attended_s_pooled = torch.bmm(attn_weights_s, v_s).reshape(b, c, ph, pw)
#         # 上采样回原始分辨率
#         attended_s = F.interpolate(attended_s_pooled, size=(h, w), mode='bilinear', align_corners=False)
#
#         x_after_spatial = x_after_channel + attended_s  # 残差连接
#
#         output = self.final_norm(x_after_spatial)  # 最终归一化
#         return output
#
# class SPPELAN_CM(nn.Module):
#     """SPP-ELAN with EfficientEnhancedCSAttention."""
#
#     def __init__(self, c1, c2, c3, k=5):
#         """Initializes SPP-ELAN block with convolution, max pooling layers, and EfficientEnhancedCSAttention."""
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#         # 将 SimAM 替换为 EfficientEnhancedCSAttention
#         # 注意：EfficientEnhancedCSAttention 的第一个参数是输入通道数，这里是 c2
#         self.attention = EfficientEnhancedCSAttention(c1=c2)
#
#     def forward(self, x):
#         """Forward pass through SPPELAN with EfficientEnhancedCSAttention."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5
#         return self.attention(y)  # Apply EfficientEnhancedCSAttention after convolution```


class HybridSimAMCSAttention(nn.Module):
    """
    一个混合注意力模块，结合了全局点积注意力和SimAM注意力。
    总权重由 α * W_Simam 和 β * W_全局 动态融合而成。
    α 和 β 是可学习的参数，在模型训练过程中通过反向传播自动优化。

    调用方式：[-1, 1, HybridSimAMCSAttention, [256]]
    """
    def __init__(self, c1, c2=None, groups=8, pool_size=4, **kwargs):
        """
        初始化函数。
        Args:
            c1 (int): 输入通道数。由框架根据上一层的输出自动提供。
            c2 (any): 兼容接口参数，通常是YAML中的输出通道数，但在此模块中未使用。
            groups (int): 通道分组数。
            pool_size (int): 空间注意力中的池化大小。
        """
        super().__init__()
        if c1 % groups != 0:
            original_groups = groups
            for g in range(groups, 0, -1):
                if c1 % g == 0:
                    groups = g
                    break
            if c1 % groups != 0:
                raise ValueError(
                    f"c1({c1}) cannot be divided by any number from the original groups({original_groups}) down to 1.")

        self.groups = groups
        self.pool_size = pool_size

        # --- 引入可学习的融合参数 α 和 β ---
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # SimAM中的超参数lambda
        self.simam_lambda = 1e-4

        self.pool = nn.AvgPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        num_gn_groups = 32 if c1 % 32 == 0 else groups
        self.final_norm = nn.GroupNorm(num_groups=num_gn_groups, num_channels=c1, eps=1e-6)

    def forward(self, x):
        b, c, h, w = x.shape

        # --- 1. 分组通道注意力 ---
        x_grouped = x.view(b, self.groups, c // self.groups, h, w)
        x_channel_seq = x_grouped.permute(0, 3, 4, 1, 2).reshape(b * h * w * self.groups, c // self.groups, 1)

        q_c, k_c, v_c = x_channel_seq, x_channel_seq, x_channel_seq

        global_scores_c = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(c // self.groups)

        mu_c = torch.mean(q_c, dim=1, keepdim=True)
        var_c = torch.var(q_c, dim=1, keepdim=True)
        e_c = (q_c - mu_c)**2 / (4 * (var_c + self.simam_lambda)) + 0.5
        simam_scores_c = -e_c.transpose(1, 2)

        total_scores_c = self.alpha * simam_scores_c + self.beta * global_scores_c

        attn_weights_c = F.softmax(total_scores_c, dim=-1)
        attended_c = torch.bmm(attn_weights_c, v_c)
        attended_c = attended_c.reshape(b, h, w, self.groups, c // self.groups).permute(0, 3, 4, 1, 2).reshape(b, c, h, w)

        x_after_channel = x + attended_c

        # --- 2. 降采样空间注意力 ---
        x_pooled = self.pool(x_after_channel)
        ph, pw = x_pooled.shape[-2:]
        x_spatial_seq = x_pooled.reshape(b * c, ph * pw, 1)

        q_s, k_s, v_s = x_spatial_seq, x_spatial_seq, x_spatial_seq

        global_scores_s = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(ph * pw)

        mu_s = torch.mean(q_s, dim=1, keepdim=True)
        var_s = torch.var(q_s, dim=1, keepdim=True)
        e_s = (q_s - mu_s)**2 / (4 * (var_s + self.simam_lambda)) + 0.5
        simam_scores_s = -e_s.transpose(1, 2)

        total_scores_s = self.alpha * simam_scores_s + self.beta * global_scores_s

        attn_weights_s = F.softmax(total_scores_s, dim=-1)
        attended_s_pooled = torch.bmm(attn_weights_s, v_s).reshape(b, c, ph, pw)
        attended_s = F.interpolate(attended_s_pooled, size=(h, w), mode='bilinear', align_corners=False)

        x_after_spatial = x_after_channel + attended_s

        output = self.final_norm(x_after_spatial)
        return output


class SPPELAN_CM(nn.Module):
    """
    SPP-ELAN with HybridSimAMCSAttention.
    """

    def __init__(self, c1, c2, c3, k=5):
        """
        Initializes SPP-ELAN block with convolution, max pooling layers,
        and the specified HybridSimAMCSAttention.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

        # ======================================================
        # <<< 核心修改点在这里 >>>
        # 将注意力模块替换为 HybridSimAMCSAttention
        # 注意：注意力模块的输入通道数是 c2
        self.attention = HybridSimAMCSAttention(c1=c2)
        # ======================================================

    def forward(self, x):
        """Forward pass through SPPELAN with HybridSimAMCSAttention."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5

        # 将SPP-ELAN的输出送入注意力模块
        return self.attention(y)


# class HybridSimAMSCAttention(nn.Module):
#     """
#     一个混合注意力模块，采用【先空间后通道】的串行结构。
#     结合了全局点积注意力和SimAM注意力。
#     总权重由 α * W_Simam 和 β * W_全局 动态融合而成。
#     α 和 β 是可学习的参数，在模型训练过程中通过反向传播自动优化。
#     """
#
#     def __init__(self, c1, c2=None, groups=8, pool_size=4, **kwargs):
#         super().__init__()
#         if c1 % groups != 0:
#             original_groups = groups
#             for g in range(groups, 0, -1):
#                 if c1 % g == 0:
#                     groups = g
#                     break
#             if c1 % groups != 0:
#                 raise ValueError(
#                     f"c1({c1}) cannot be divided by any number from the original groups({original_groups}) down to 1.")
#
#         self.groups = groups
#         self.pool_size = pool_size
#
#         # --- 引入可学习的融合参数 α 和 β (保持不变) ---
#         self.alpha = nn.Parameter(torch.tensor(1.0))
#         self.beta = nn.Parameter(torch.tensor(1.0))
#
#         # SimAM中的超参数lambda
#         self.simam_lambda = 1e-4
#
#         self.pool = nn.AvgPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
#         num_gn_groups = 32 if c1 % 32 == 0 else groups
#         self.final_norm = nn.GroupNorm(num_groups=num_gn_groups, num_channels=c1, eps=1e-6)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # --- 1. 降采样空间注意力 (Downsampled Spatial Attention) - 首先执行 ---
#         # 注意：输入是原始的 x
#         x_pooled = self.pool(x)
#         ph, pw = x_pooled.shape[-2:]
#         x_spatial_seq = x_pooled.reshape(b * c, ph * pw, 1)
#
#         q_s, k_s, v_s = x_spatial_seq, x_spatial_seq, x_spatial_seq
#
#         # a) 计算全局注意力分数
#         global_scores_s = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(ph * pw)
#
#         # b) 计算SimAM分数
#         mu_s = torch.mean(q_s, dim=1, keepdim=True)
#         var_s = torch.var(q_s, dim=1, keepdim=True)
#         e_s = (q_s - mu_s) ** 2 / (4 * (var_s + self.simam_lambda)) + 0.5
#         simam_scores_s = -e_s.transpose(1, 2)
#
#         # c) 融合分数
#         total_scores_s = self.alpha * simam_scores_s + self.beta * global_scores_s
#
#         # d) 计算最终权重和输出
#         attn_weights_s = F.softmax(total_scores_s, dim=-1)
#         attended_s_pooled = torch.bmm(attn_weights_s, v_s).reshape(b, c, ph, pw)
#         attended_s = F.interpolate(attended_s_pooled, size=(h, w), mode='bilinear', align_corners=False)
#
#         # 第一个残差连接：空间注意力的输出与原始输入 x 相加
#         x_after_spatial = x + attended_s
#
#         # --- 2. 分组通道注意力 (Grouped Channel Attention) - 其次执行 ---
#         # 注意：输入是上一步的输出 x_after_spatial
#         x_grouped = x_after_spatial.view(b, self.groups, c // self.groups, h, w)
#         x_channel_seq = x_grouped.permute(0, 3, 4, 1, 2).reshape(b * h * w * self.groups, c // self.groups, 1)
#
#         q_c, k_c, v_c = x_channel_seq, x_channel_seq, x_channel_seq
#
#         # a) 计算全局注意力分数
#         global_scores_c = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(c // self.groups)
#
#         # b) 计算SimAM分数
#         mu_c = torch.mean(q_c, dim=1, keepdim=True)
#         var_c = torch.var(q_c, dim=1, keepdim=True)
#         e_c = (q_c - mu_c) ** 2 / (4 * (var_c + self.simam_lambda)) + 0.5
#         simam_scores_c = -e_c.transpose(1, 2)
#
#         # c) 融合分数
#         total_scores_c = self.alpha * simam_scores_c + self.beta * global_scores_c
#
#         # d) 计算最终权重和输出
#         attn_weights_c = F.softmax(total_scores_c, dim=-1)
#         attended_c = torch.bmm(attn_weights_c, v_c)
#         attended_c = attended_c.reshape(b, h, w, self.groups, c // self.groups).permute(0, 3, 4, 1, 2).reshape(b, c, h,
#                                                                                                                w)
#
#         # 第二个残差连接：通道注意力的输出与 x_after_spatial 相加
#         x_after_sc = x_after_spatial + attended_c
#
#         # --- 最终归一化 ---
#         output = self.final_norm(x_after_sc)
#         return output
#
# class SPPELAN_CM(nn.Module):
#     """
#     SPP-ELAN with HybridSimAMSCAttention (the SC version).
#     """
#
#     def __init__(self, c1, c2, c3, k=5):
#         """
#         Initializes SPP-ELAN block with convolution, max pooling layers,
#         and the specified HybridSimAMSCAttention (SC version).
#         """
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#
#         # ======================================================
#         # <<< 核心修改点在这里 >>>
#         # 将注意力模块替换为性能更优的 SC (先空间后通道) 版本
#         # 注意：注意力模块的输入通道数是 c2
#         self.attention = HybridSimAMSCAttention(c1=c2)
#         # ======================================================
#
#     def forward(self, x):
#         """Forward pass through SPPELAN with HybridSimAMSCAttention."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5
#
#         # 将SPP-ELAN的输出送入注意力模块
#         return self.attention(y)

#
# class LSTMInspiredGatedCSAttention(nn.Module):
#     """
#     一个混合注意力模块，使用受 LSTM 启发的门控结构动态融合 SimAM 和全局权重。
#
#     核心创新:
#     1. 移除了静态的 alpha 和 beta 参数。
#     2. 引入了一个轻量级的门控网络 (GateMLP)，它根据输入内容动态生成权重。
#     3. 使用 Softmax 确保两个权重相加为1，强制模型进行权衡。
#
#     调用方式：[-1, 1, LSTMInspiredGatedCSAttention, [256]]
#     """
#
#     def __init__(self, c1, c2=None, groups=8, pool_size=4, **kwargs):
#         super().__init__()
#         if c1 % groups != 0:
#             original_groups = groups
#             for g in range(groups, 0, -1):
#                 if c1 % g == 0:
#                     groups = g
#                     break
#             if c1 % groups != 0:
#                 raise ValueError(
#                     f"c1({c1}) cannot be divided by any number from the original groups({original_groups}) down to 1.")
#
#         self.groups = groups
#         self.pool_size = pool_size
#         self.simam_lambda = 1e-4
#
#         # --- 创新点: "LSTM-inspired" 门控网络 ---
#         reduction = 4
#         self.gate_mlp = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(c1, c1 // reduction),
#             nn.ReLU(),
#             nn.Linear(c1 // reduction, 2),  # 输出2个logits: for SimAM and Global
#             nn.Softmax(dim=1)  # Softmax确保两者相加为1，形成权衡
#         )
#
#         self.pool = nn.AvgPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
#         num_gn_groups = 32 if c1 % 32 == 0 else groups
#         self.final_norm = nn.GroupNorm(num_groups=num_gn_groups, num_channels=c1, eps=1e-6)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # --- 步骤 1: 动态计算门控权重 ---
#         # gates: [B, 2]
#         gates = self.gate_mlp(x)
#         gate_simam = gates[:, 0]
#         gate_global = gates[:, 1]
#
#         # 【调试支持】暂存以便 trainer.py 打印
#         self.last_gate_simam = gate_simam.detach()
#         self.last_gate_global = gate_global.detach()
#
#         # --- 步骤 2: 分组通道注意力 ---
#         x_grouped = x.view(b, self.groups, c // self.groups, h, w)
#         x_channel_seq = x_grouped.permute(0, 3, 4, 1, 2).reshape(b * h * w * self.groups, c // self.groups, 1)
#
#         q_c, k_c, v_c = x_channel_seq, x_channel_seq, x_channel_seq
#
#         global_scores_c = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(c // self.groups)
#
#         mu_c = torch.mean(q_c, dim=1, keepdim=True)
#         var_c = torch.var(q_c, dim=1, keepdim=True)
#         e_c = (q_c - mu_c) ** 2 / (4 * (var_c + self.simam_lambda)) + 0.5
#         simam_scores_c = -e_c.transpose(1, 2)
#
#         # --- 使用动态门控进行融合 ---
#         # gate_simam/global: [B] -> [B, 1, 1, 1] -> 适配广播
#         # 注意：这里的广播需要小心处理，因为N维度是 b*h*w*groups
#         # 我们将 batch-wise 的门控扩展到所有像素
#         gate_simam_expanded_c = gate_simam.view(b, 1, 1, 1).repeat(1, self.groups, h, w).permute(0, 2, 3, 1).reshape(-1,
#                                                                                                                      1,
#                                                                                                                      1)
#         gate_global_expanded_c = gate_global.view(b, 1, 1, 1).repeat(1, self.groups, h, w).permute(0, 2, 3, 1).reshape(
#             -1, 1, 1)
#
#         total_scores_c = gate_simam_expanded_c * simam_scores_c + gate_global_expanded_c * global_scores_c
#
#         attn_weights_c = F.softmax(total_scores_c, dim=-1)
#         attended_c = torch.bmm(attn_weights_c, v_c)
#         attended_c = attended_c.reshape(b, h, w, self.groups, c // self.groups).permute(0, 3, 4, 1, 2).reshape(b, c, h,
#                                                                                                                w)
#
#         x_after_channel = x + attended_c
#
#         # --- 步骤 3: 降采样空间注意力 ---
#         x_pooled = self.pool(x_after_channel)
#         ph, pw = x_pooled.shape[-2:]
#         x_spatial_seq = x_pooled.reshape(b * c, ph * pw, 1)
#
#         q_s, k_s, v_s = x_spatial_seq, x_spatial_seq, x_spatial_seq
#
#         global_scores_s = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(ph * pw)
#
#         mu_s = torch.mean(q_s, dim=1, keepdim=True)
#         var_s = torch.var(q_s, dim=1, keepdim=True)
#         e_s = (q_s - mu_s) ** 2 / (4 * (var_s + self.simam_lambda)) + 0.5
#         simam_scores_s = -e_s.transpose(1, 2)
#
#         # --- 同样使用动态门控进行融合 ---
#         gate_simam_expanded_s = gate_simam.view(b, 1, 1, 1).repeat(1, c, 1, 1).reshape(-1, 1, 1)
#         gate_global_expanded_s = gate_global.view(b, 1, 1, 1).repeat(1, c, 1, 1).reshape(-1, 1, 1)
#
#         total_scores_s = gate_simam_expanded_s * simam_scores_s + gate_global_expanded_s * global_scores_s
#
#         attn_weights_s = F.softmax(total_scores_s, dim=-1)
#         attended_s_pooled = torch.bmm(attn_weights_s, v_s).reshape(b, c, ph, pw)
#         attended_s = F.interpolate(attended_s_pooled, size=(h, w), mode='bilinear', align_corners=False)
#
#         x_after_spatial = x_after_channel + attended_s
#
#         output = self.final_norm(x_after_spatial)
#         return output
#
# class SPPELAN_CM(nn.Module):
#     """
#     SPP-ELAN with LSTMInspiredGatedCSAttention.
#     """
#
#     def __init__(self, c1, c2, c3, k=5):
#         """
#         Initializes SPP-ELAN block with convolution, max pooling layers,
#         and the specified LSTMInspiredGatedCSAttention.
#         """
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#
#         # ======================================================
#         # <<< 核心修改点在这里 >>>
#         # 将注意力模块替换为 LSTMInspiredGatedCSAttention
#         # 注意：注意力模块的输入通道数是 c2
#         self.attention = LSTMInspiredGatedCSAttention(c1=c2)
#         # ======================================================
#
#     def forward(self, x):
#         """Forward pass through SPPELAN with LSTMInspiredGatedCSAttention."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5
#
#         # 将SPP-ELAN的输出送入注意力模块
#         return self.attention(y)

# class BiLSTMGatedSCAttention(nn.Module):
#     """
#     一个混合注意力模块，采用【先空间后通道】结构，并使用【BiLSTM门控】动态融合 SimAM 和全局权重。
#
#     核心创新:
#     1. 采用最优的 SC 串行结构。
#     2. 使用 BiLSTM 门控网络捕捉通道间依赖，根据输入内容动态生成 SimAM 和全局权重的门控值。
#     3. 使用 Softmax 确保门控值相加为1，强制模型进行权衡。
#
#     调用方式：[-1, 1, BiLSTMGatedSCAttention, [256]]
#     """
#
#     def __init__(self, c1, c2=None, groups=8, pool_size=4, **kwargs):
#         super().__init__()
#         # ... (分组检查代码不变) ...
#         if c1 % groups != 0:
#             original_groups = groups
#             for g in range(groups, 0, -1):
#                 if c1 % g == 0:
#                     groups = g
#                     break
#             if c1 % groups != 0:
#                 raise ValueError(
#                     f"c1({c1}) cannot be divided by any number from the original groups({original_groups}) down to 1.")
#
#         self.groups = groups
#         self.pool_size = pool_size
#         self.simam_lambda = 1e-4
#
#         # BiLSTM 门控网络
#         self.gate_avg_pool = nn.AdaptiveAvgPool2d(1)
#         bilstm_hidden_size = max(16, c1 // 8)
#         self.gate_bilstm = nn.LSTM(input_size=1,
#                                    hidden_size=bilstm_hidden_size,
#                                    num_layers=1,
#                                    bidirectional=True,
#                                    batch_first=True)
#         self.gate_fc = nn.Linear(bilstm_hidden_size * 2, 2)
#
#         self.pool = nn.AvgPool2d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
#         num_gn_groups = 32 if c1 % 32 == 0 else groups
#         self.final_norm = nn.GroupNorm(num_groups=num_gn_groups, num_channels=c1, eps=1e-6)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # --- 步骤 1: 动态计算 BiLSTM 门控权重 (基于原始输入 x) ---
#         context_vector = self.gate_avg_pool(x).squeeze(-1).squeeze(-1)
#         lstm_input_seq = context_vector.unsqueeze(-1)
#         lstm_output, _ = self.gate_bilstm(lstm_input_seq)
#         aggregated_vector = lstm_output[:, -1, :]
#         logits = self.gate_fc(aggregated_vector)
#         gates = F.softmax(logits, dim=1)
#
#         gate_simam = gates[:, 0]
#         gate_global = gates[:, 1]
#
#         self.last_gate_simam = gate_simam.detach()
#         self.last_gate_global = gate_global.detach()
#
#         # --- 步骤 2: 降采样空间注意力 (先执行, 作用于原始 x) ---
#         x_pooled = self.pool(x)
#         ph, pw = x_pooled.shape[-2:]
#         x_spatial_seq = x_pooled.reshape(b * c, ph * pw, 1)
#         q_s, k_s, v_s = x_spatial_seq, x_spatial_seq, x_spatial_seq
#         global_scores_s = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(ph * pw)
#         mu_s = torch.mean(q_s, dim=1, keepdim=True);
#         var_s = torch.var(q_s, dim=1, keepdim=True)
#         e_s = (q_s - mu_s) ** 2 / (4 * (var_s + self.simam_lambda)) + 0.5
#         simam_scores_s = -e_s.transpose(1, 2)
#
#         gate_simam_expanded_s = gate_simam.view(b, 1, 1, 1).repeat(1, c, 1, 1).reshape(-1, 1, 1)
#         gate_global_expanded_s = gate_global.view(b, 1, 1, 1).repeat(1, c, 1, 1).reshape(-1, 1, 1)
#         total_scores_s = gate_simam_expanded_s * simam_scores_s + gate_global_expanded_s * global_scores_s
#
#         attn_weights_s = F.softmax(total_scores_s, dim=-1)
#         attended_s_pooled = torch.bmm(attn_weights_s, v_s).reshape(b, c, ph, pw)
#         attended_s = F.interpolate(attended_s_pooled, size=(h, w), mode='bilinear', align_corners=False)
#         x_after_spatial = x + attended_s
#
#         # --- 步骤 3: 分组通道注意力 (后执行, 输入为 x_after_spatial) ---
#         x_grouped = x_after_spatial.view(b, self.groups, c // self.groups, h, w)
#         x_channel_seq = x_grouped.permute(0, 3, 4, 1, 2).reshape(b * h * w * self.groups, c // self.groups, 1)
#         q_c, k_c, v_c = x_channel_seq, x_channel_seq, x_channel_seq
#         global_scores_c = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(c // self.groups)
#         mu_c = torch.mean(q_c, dim=1, keepdim=True);
#         var_c = torch.var(q_c, dim=1, keepdim=True)
#         e_c = (q_c - mu_c) ** 2 / (4 * (var_c + self.simam_lambda)) + 0.5
#         simam_scores_c = -e_c.transpose(1, 2)
#
#         gate_simam_expanded_c = gate_simam.view(b, 1, 1, 1).repeat(1, self.groups, h, w).permute(0, 2, 3, 1).reshape(-1,
#                                                                                                                      1,
#                                                                                                                      1)
#         gate_global_expanded_c = gate_global.view(b, 1, 1, 1).repeat(1, self.groups, h, w).permute(0, 2, 3, 1).reshape(
#             -1, 1, 1)
#         total_scores_c = gate_simam_expanded_c * simam_scores_c + gate_global_expanded_c * global_scores_c
#
#         attn_weights_c = F.softmax(total_scores_c, dim=-1)
#         attended_c = torch.bmm(attn_weights_c, v_c)
#         attended_c = attended_c.reshape(b, h, w, self.groups, c // self.groups).permute(0, 3, 4, 1, 2).reshape(b, c, h,
#                                                                                                                w)
#         x_after_sc = x_after_spatial + attended_c
#
#         # --- 最终归一化 ---
#         output = self.final_norm(x_after_sc)
#         return output
#
# class SPPELAN_CM(nn.Module):
#     """
#     SPP-ELAN with BiLSTMGatedSCAttention.
#     """
#
#     def __init__(self, c1, c2, c3, k=5):
#         """
#         Initializes SPP-ELAN block with convolution, max pooling layers,
#         and the specified BiLSTMGatedSCAttention.
#         """
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#
#         # ======================================================
#         # <<< 核心修改点在这里 >>>
#         # 将注意力模块替换为 BiLSTMGatedSCAttention
#         # 注意：注意力模块的输入通道数是 c2
#         self.attention = BiLSTMGatedSCAttention(c1=c2)
#         # ======================================================
#
#     def forward(self, x):
#         """Forward pass through SPPELAN with BiLSTMGatedSCAttention."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         y = self.cv5(torch.cat(y, 1))  # Concatenate and pass through cv5
#
#         # 将SPP-ELAN的输出送入注意力模块
#         return self.attention(y)




