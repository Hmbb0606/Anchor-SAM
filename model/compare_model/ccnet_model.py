'''
@ARTICLE{CCNet,
  author={Yang, Zhigang and Zhang, Wei and Li, Qiang and Ni, Weiping and Wu, Junzheng and Wang, Qi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={C²Net: Road Extraction via Context Perception and Cross Spatial-Scale Feature Interaction},
  year={2024},
  volume={62},
  number={},
  pages={1-11}
}
'''

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu,inplace=True)
from torchvision import models
from torch import nn, einsum
from einops import rearrange, repeat
from torchvision.models import ResNet34_Weights
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)



class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1, h, w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=in_channels)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class Spatialaware(nn.Module):
    def __init__(self, inchannel):
        super(Spatialaware, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, 1, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(inchannel, 1, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(inchannel, 1, kernel_size=3, dilation=1, padding=1)
        self.conv4 = nn.Conv2d(inchannel, 1, kernel_size=3, dilation=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self,x1,x2,x3,x4):
        feature1 = self.conv1(x1)
        feature2 = self.conv2(x2)
        feature3 = self.conv3(x3)
        feature4 = self.conv4(x4)
        fuse_feature = torch.cat([feature1,feature2,feature3,feature4],dim=1)
        att = self.sig(fuse_feature)
        att1 = att[:,0,:,:].unsqueeze(1)
        att2 = att[:,1,:,:].unsqueeze(1)
        att3 = att[:,2,:,:].unsqueeze(1)
        att4 = att[:,3,:,:].unsqueeze(1)
        fea = x1*att1+x2*att2+x3*att3+x4*att4
        return  fea


class Semanticdependence(nn.Module):
    def __init__(self, inchannel):
        super(Semanticdependence, self).__init__()
 
        self.conv = nn.Conv1d(1, 1, kernel_size=11, dilation=1, padding=5)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sig = nn.Sigmoid()

    def forward(self,x1,x2,x3,x4):
        n,c,h,w = x1.shape
        feature1 = self.pool(x1)
        feature2 = self.pool(x2)
        feature3 = self.pool(x3)
        feature4 = self.pool(x4)
        fuse_feature = torch.cat([feature1,feature2,feature3,feature4],dim=1)
        feature_map = fuse_feature.view(n, 2048, 1).transpose(-1, -2)
        feature_map = self.conv(feature_map).transpose(-1, -2)
        feature_map = feature_map.unsqueeze(dim=3)
        att = self.sig(feature_map)
        att1 = feature_map[:, 0:512, :, :]
        att2 = feature_map[:, 512:1024, :, :]
        att3 = feature_map[:, 1024:1536, :, :]
        att4 = feature_map[:, 1536:2048, :, :]
        fea = x1*att1 + x2*att2 + x3*att3 + x4*att4
        return fea

class inter_attn(nn.Module):
    def __init__(self, f_dim, n_heads=8, d_q=None, d_v=None, dropout=0.3):
        super().__init__()
        self.build_inter_attn(f_dim, n_heads, d_q, d_v, dropout)

        for m in self.modules():
            weights_init(m)

    def build_inter_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)

    def inter_attn(self, Local, Fuse, mask_L2R=None, mask_R2L=None):
        b, c, h, w = Local.shape
        Lf = Local.view(b, c, -1).permute(0, 2, 1)
        Rf = Fuse.view(b, c, -1).permute(0, 2, 1)
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf2 = self.layer_norm1(Lf)
        Rf2 = self.layer_norm2(Rf)

        Lq = self.w_qs(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lk = self.w_ks(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lv = self.w_vs(Lf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        Rq = self.w_qs(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rk = self.w_ks(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rv = self.w_vs(Rf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn_R2L = torch.matmul(Lq, Rk.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask_R2L is not None:
            attn_R2L = attn_R2L.masked_fill(mask_R2L == 0, -1e9)

        attn_R2L = F.softmax(attn_R2L, dim=-1)  # bs, h, V, V

        attn_R2L = self.dropout1(attn_R2L)

        feat_R2L = torch.matmul(attn_R2L, Rv).transpose(1, 2).contiguous().view(BS, V, -1)

        feat_R2L = self.dropout2(self.fc(feat_R2L))

        Lf = Lf + feat_R2L

        Local = Lf.view(Local.size(0), Local.size(2), Local.size(3), Local.size(1)).permute(0, 3, 1, 2)

        return Local

    def forward(self, Lf, Rf, mask_L2R=None, mask_R2L=None):

        Lf = self.inter_attn(Lf, Rf, mask_L2R, mask_R2L)

        return Lf

class GLCPblock(nn.Module):
    def __init__(self, channel):
        super(GLCPblock, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.swin2 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                num_heads=8, head_dim=32,
                                  window_size=2, relative_pos_embedding=True)
        self.swin4 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                num_heads=8, head_dim=32,
                                  window_size=4, relative_pos_embedding=True)
        self.swin8 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                num_heads=8, head_dim=32,
                                  window_size=8, relative_pos_embedding=True)
        self.swin16 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                num_heads=8, head_dim=32,
                                  window_size=16, relative_pos_embedding=True)
        
        self.crossfuse = inter_attn(f_dim=512)

        self.conv = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=1, padding=0)

        self.act1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act4 = nn.Conv2d(channel, channel, kernel_size=1)

        self.SA = Spatialaware(inchannel=channel)
        self.SD = Semanticdependence(inchannel=channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate11_out = nonlinearity(self.swin2(dilate1_out))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate22_out = nonlinearity(self.swin4(dilate2_out))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate33_out = nonlinearity(self.swin8(dilate3_out))
        dilate4_out = nonlinearity(self.dilate4(x))
        dilate44_out = nonlinearity(self.swin16(dilate4_out))
        fuse1 = self.act1(dilate1_out+dilate11_out)
        fuse2 = self.act2(dilate2_out+dilate22_out)
        fuse3 = self.act3(dilate3_out+dilate33_out)
        fuse4 = self.act4(dilate4_out+dilate44_out)
        spatialfuse = self.SA(fuse1,fuse2,fuse3,fuse4)
        semanticfuse = self.SD(fuse1,fuse2,fuse3,fuse4)

        branch = self.conv(spatialfuse+semanticfuse)

        return branch


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class DWCONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=True
        )

    def forward(self, x):
        result = self.depthwise(x)
        return result

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock2,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels//2, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.relu2 = nonlinearity

        self.swin = StageModule(in_channels=in_channels, hidden_dimension=in_channels//2, layers=2,
                                 num_heads=8, head_dim=in_channels//8,
                                 window_size=8, relative_pos_embedding=True)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels // 2, 1)

        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x2= self.swin(x)

        x = self.deconv2(self.conv2(x+x2))
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x



class CSFIMblock(nn.Module):
    def __init__(self):
        super(CSFIMblock, self).__init__()
        self.block_layer = [1, 1, 1, 1]
        self.image_size = [256, 128, 64, 32]
        self.patchsize = [32, 16, 8, 4]
        self.channels = [64, 128, 256, 512]
        self.split_list = [32 * 32, 16 * 16, 8 * 8, 4 * 4]


        self.conv1 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[1], kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[1], kernel_size=3,
                               stride=1, padding=1)
        self.convFuse1 = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=(1, 129), dilation=7,
                                   padding=(0, 448))
        self.convFuse2 = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=(5, 1), dilation=2, padding=(4, 0))
        self.convFuse3 = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=3, dilation=1, padding=1)
        self.conv = nn.Conv2d(in_channels=self.channels[1] * 3, out_channels=self.channels[1], kernel_size=1,
                              stride=1, padding=0)

        self.conv_module = nn.ModuleList()
        for i in range(len(self.block_layer)):
            self.conv_module.append(
                nn.Conv2d(self.channels[1], self.channels[i], kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):

        feature0 = self.conv1(x[0])
        feature1 = self.conv2(x[1])
        feature2 = self.conv3(x[2])
        feature3 = self.conv4(x[3])
        b, c, h, w = feature0.shape

        feature0 = feature0.view(b, c, 8, self.patchsize[0], 8, self.patchsize[0]).permute(0, 2, 4, 1, 3,
                                                                                           5).contiguous()
        feature0 = feature0.view(b, 8, 8, c, self.patchsize[0] * self.patchsize[0]).permute(0, 3, 1, 2,
                                                                                            4).contiguous().view(b, c,
                                                                                                                 -1,
                                                                                                                 self.patchsize[
                                                                                                                     0] *
                                                                                                                 self.patchsize[
                                                                                                                     0])

        b, c, h, w = feature1.shape
        feature1 = feature1.view(b, c, 8, self.patchsize[1], 8, self.patchsize[1]).permute(0, 2, 4, 1, 3,
                                                                                           5).contiguous()
        feature1 = feature1.view(b, 8, 8, c, self.patchsize[1] * self.patchsize[1]).permute(0, 3, 1, 2,
                                                                                            4).contiguous().view(b, c,
                                                                                                                 -1,
                                                                                                                 self.patchsize[
                                                                                                                     1] *
                                                                                                                 self.patchsize[
                                                                                                                     1])
        b, c, h, w = feature2.shape
        feature2 = feature2.view(b, c, 8, self.patchsize[2], 8, self.patchsize[2]).permute(0, 2, 4, 1, 3,
                                                                                           5).contiguous()
        feature2 = feature2.view(b, 8, 8, c, self.patchsize[2] * self.patchsize[2]).permute(0, 3, 1, 2,
                                                                                            4).contiguous().view(b, c,
                                                                                                                 -1,
                                                                                                                 self.patchsize[
                                                                                                                     2] *
                                                                                                                 self.patchsize[
                                                                                                                     2])
        b, c, h, w = feature3.shape
        feature3 = feature3.view(b, c, 8, self.patchsize[3], 8, self.patchsize[3]).permute(0, 2, 4, 1, 3,
                                                                                           5).contiguous()
        feature3 = feature3.view(b, 8, 8, c, self.patchsize[3] * self.patchsize[3]).permute(0, 3, 1, 2,
                                                                                            4).contiguous().view(b, c,
                                                                                                                 -1,
                                                                                                                 self.patchsize[
                                                                                                                     3] *
                                                                                                                 self.patchsize[
                                                                                                                     3])

        Fusefeature = torch.cat([feature0, feature1, feature2, feature3], dim=-1)
        Fusefeature1 = self.convFuse1(Fusefeature) + Fusefeature
        Fusefeature2 = self.convFuse2(Fusefeature) + Fusefeature
        Fusefeature3 = self.convFuse3(Fusefeature) + Fusefeature

        Fusefeatureall = self.conv(torch.cat([Fusefeature1, Fusefeature2, Fusefeature3], dim=1))

        x = torch.split(Fusefeatureall, self.split_list, dim=-1)
        x = list(x)
        for j, item in enumerate(x):
            B, C, num_blocks, N = item.shape
            item = item.reshape(B, C, 8, 8, self.patchsize[j], self.patchsize[j]).permute(0, 1, 2, 4, 3,
                                                                                          5).contiguous().reshape(B, C,
                                                                                                                  8 *
                                                                                                                  self.patchsize[
                                                                                                                      j],
                                                                                                                  8 *
                                                                                                                  self.patchsize[
                                                                                                                      j])
            item = self.conv_module[j](item)
            x[j] = item
        return x



class CCNet(nn.Module):
    def __init__(self, n_classes=1):
        super(CCNet, self).__init__()
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./pretrain_weight/resnet34-b627a593.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.encoder_channels = [512, 256, 128, 64]

        self.GLCPblock =GLCPblock(channel=512)
        self.CSFIMblock = CSFIMblock()

        self.decoder4 = DecoderBlock(self.encoder_channels[0], self.encoder_channels[1])
        self.decoder3 = DecoderBlock(self.encoder_channels[1], self.encoder_channels[2])
        self.decoder2 = DecoderBlock(self.encoder_channels[2], self.encoder_channels[3])
        self.decoder1 = DecoderBlock(self.encoder_channels[3], self.encoder_channels[3])


        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=3,
        )
        self.decoder_final = DecoderBlock(64, 64)
        self.conv4_1 = ConvBNReLU(c_in= self.encoder_channels[0]*2, c_out = self.encoder_channels[0],kernel_size =1)
        self.conv3_1 = ConvBNReLU(c_in= self.encoder_channels[1]*2, c_out = self.encoder_channels[1],kernel_size =1)
        self.conv2_1 = ConvBNReLU(c_in= self.encoder_channels[2]*2, c_out = self.encoder_channels[2],kernel_size =1)
        self.conv1_1 = ConvBNReLU(c_in= self.encoder_channels[3]*2, c_out = self.encoder_channels[3],kernel_size =1)
        
        self.conv5 = nn.Conv2d(in_channels=self.encoder_channels[-4], out_channels=self.encoder_channels[-1], kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.encoder_channels[-3], out_channels=self.encoder_channels[-1], kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.encoder_channels[-2], out_channels=self.encoder_channels[-1], kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.encoder_channels[-1], out_channels=self.encoder_channels[-1], kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)


    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        features = []
        features.append(e1)
        features.append(e2)
        features.append(e3)
        features.append(e4)
        Fuse_features = self.CSFIMblock(features)

        d4 = self.GLCPblock(e4) + e4 + Fuse_features[-1]
        d3 = self.decoder4(d4) + e3 + Fuse_features[-2]
        d2 = self.decoder3(d3) + e2 + Fuse_features[-3]
        d1 = self.decoder2(d2) + e1 + Fuse_features[-4]
        d0 = self.decoder1(d1) + x
        x_final = self.decoder_final(d0 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d3)) + self.conv3(
            self.hd2_d0(d2)) + self.conv2(self.hd1_d0(d1)))
        logits = self.segmentation_head(x_final)
        return logits


if __name__ == '__main__':

    model = CCNet(n_classes=1)
    dummy_input = torch.randn(1, 3, 1024, 1024)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

