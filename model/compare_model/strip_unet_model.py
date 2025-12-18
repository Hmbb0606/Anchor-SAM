'''
@article{StripUnet,
  title={{StripUnet}: A Method for Dense Road Extraction from Remote Sensing Images},
  author={Ma, Xianzhi and Zhang, Xiaokai and Zhou, Daoxiang and Chen, Zehua},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  pages={7097--7109},
  volume={9},
  publisher={IEEE Transactions on Intelligent Vehicles}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from functools import partial
from timm.models.layers import DropPath
import warnings

warnings.filterwarnings("ignore")


# --- 从 DSConv.py 移植的代码 ---

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device=None):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu (set to None by default for better flexibility)
        """
        super(DSConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4 if out_ch > 1 else 1, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        if self.device is None:
            self.device = f.device

        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        # 校正变量赋值以匹配PyTorch的(H, W)约定
        self.height = input_shape[2]
        self.width = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope

        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width, device=self.device).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width).permute(1, 0).reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float().unsqueeze(0)

        x_center = torch.arange(0, self.height, device=self.device).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height).permute(0, 1).reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float().unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1, device=self.device)
            x = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points),
                               device=self.device)

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height]).reshape(
                [self.num_points, self.width, self.height]).unsqueeze(0)
            x_grid = x_spread.repeat([1, self.width * self.height]).reshape(
                [self.num_points, self.width, self.height]).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                y_offset_new[center] = 0
                for index in range(1, center + 1):
                    if (center + index) < self.num_points:
                        y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    if (center - index) >= 0:
                        y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height]).permute(0, 3, 1, 4,
                                                                                                         2).reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height])
            x_new = x_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height]).permute(0, 3, 1, 4,
                                                                                                         2).reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height])
            return y_new, x_new

        else:
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points),
                               device=self.device)
            x = torch.linspace(0, 0, 1, device=self.device)

            y, x = torch.meshgrid(y, x, indexing='ij')
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height]).reshape(
                [self.num_points, self.width, self.height]).unsqueeze(0)
            x_grid = x_spread.repeat([1, self.width * self.height]).reshape(
                [self.num_points, self.width, self.height]).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1).to(self.device)

            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center + 1):
                    if (center + index) < self.num_points:
                        x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    if (center - index) >= 0:
                        x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height]).permute(0, 3, 1, 4,
                                                                                                         2).reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height])
            x_new = x_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height]).permute(0, 3, 1, 4,
                                                                                                         2).reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height])
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        # y是宽度坐标, x是高度坐标
        y, x = y.reshape([-1]).float(), x.reshape([-1]).float()

        # Bug fix: 统一使用 long 类型进行索引计算, 避免潜在的类型转换和溢出问题
        zero = torch.zeros([]).long().to(self.device)
        max_y = torch.tensor(self.width - 1, device=self.device).long()
        max_x = torch.tensor(self.height - 1, device=self.device).long()

        y0, y1 = torch.floor(y).long(), torch.floor(y).long() + 1
        x0, x1 = torch.floor(x).long(), torch.floor(x).long() + 1

        y0, y1 = torch.clamp(y0, zero, max_y), torch.clamp(y1, zero, max_y)
        x0, x1 = torch.clamp(x0, zero, max_x), torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.permute(0, 2, 3, 1).contiguous().reshape(-1, self.num_channels)

        dimension = self.height * self.width
        points_per_sample = self.num_points * self.height * self.width
        batch_indices = torch.arange(self.num_batch, device=self.device).repeat_interleave(points_per_sample)
        base = (batch_indices * dimension).long()

        stride = self.width

        index_a0 = base + x0 * stride + y0
        index_c0 = base + x0 * stride + y1
        index_a1 = base + x1 * stride + y0
        index_c1 = base + x1 * stride + y1

        value_a0, value_c0 = input_feature_flat[index_a0], input_feature_flat[index_c0]
        value_a1, value_c1 = input_feature_flat[index_a1], input_feature_flat[index_c1]

        y0_f, y1_f = y0.float(), y1.float()
        x0_f, x1_f = x0.float(), x1.float()

        vol_a0 = ((y1_f - y) * (x1_f - x)).unsqueeze(-1)
        vol_c0 = ((y1_f - y) * (x - x0_f)).unsqueeze(-1)
        vol_a1 = ((y - y0_f) * (x1_f - x)).unsqueeze(-1)
        vol_c1 = ((y - y0_f) * (x - x0_f)).unsqueeze(-1)

        outputs = value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1

        if self.morph == 0:
            outputs = outputs.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height, self.num_channels]).permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height, self.num_channels]).permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


# --- 从 StripUNet.py 移植的辅助模块 ---

nonlinearity = partial(F.relu, inplace=True)


class DAM(nn.Module):
    def __init__(self, c_size, channel):
        super(DAM, self).__init__()
        self.GPA1 = nn.AdaptiveAvgPool2d((c_size, c_size))
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.sigmoid1 = nn.Sigmoid()

        self.GPA2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv5 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.stripatt = TDAtt(channel)

    def forward(self, c_x, d_x):
        x1 = self.GPA1(c_x)
        x1 = self.conv1(x1)
        x1 = nonlinearity(self.bn1(x1))
        x1 = self.sigmoid1(x1)
        x1 = torch.mul(c_x, x1)
        x2 = self.GPA2(c_x)
        x2 = self.conv2(x2)
        x2 = nonlinearity(x2)
        x2 = self.conv3(x2)
        x2 = self.sigmoid2(x2)
        x2 = torch.mul(c_x, x2)
        x2 = c_x + x1 + x2
        d_x = self.stripatt(d_x)
        x = torch.cat((x2, d_x), dim=1)
        x = self.conv5(x)
        x = nonlinearity(self.bn2(x))
        return x


class TDAtt(nn.Module):
    def __init__(self, dim):
        super(TDAtt, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class MSFF(nn.Module):
    def __init__(self, channel):
        super(MSFF, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        self.block1 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block2 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block3 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block4 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block5 = Block(channel, mlp_ratio=4, drop_path=0.2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))

        dilate1_out = self.block1(dilate1_out)
        dilate2_out = self.block2(dilate2_out)
        dilate3_out = self.block3(dilate3_out)
        dilate4_out = self.block4(dilate4_out)
        x = self.block5(x)

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.1):
        super().__init__()
        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class sfem(nn.Module):
    def __init__(self, nIn, d=1):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, 3, 1, padding=1, bn_acti=True)

        self.dconv3x1_1_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1, padding=(1, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x3_1_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1, padding=(0, 1), groups=nIn // 16, bn_acti=True)
        self.dconv5x1_1_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1, padding=(2, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x5_1_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1, padding=(0, 2), groups=nIn // 16, bn_acti=True)
        self.dconv7x1_1_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1, padding=(3, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x7_1_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1, padding=(0, 3), groups=nIn // 8, bn_acti=True)

        self.dconv3x1_2_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1, padding=(int(d / 4 + 1), 0),
                                 dilation=(int(d / 4 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x3_2_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1, padding=(0, int(d / 4 + 1)),
                                 dilation=(1, int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)
        self.dconv5x1_2_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1, padding=(2 * int(d / 4 + 1), 0),
                                 dilation=(int(d / 4 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x5_2_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1, padding=(0, 2 * int(d / 4 + 1)),
                                 dilation=(1, int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)
        self.dconv7x1_2_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1, padding=(3 * int(d / 4 + 1), 0),
                                 dilation=(int(d / 4 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x7_2_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1, padding=(0, 3 * int(d / 4 + 1)),
                                 dilation=(1, int(d / 4 + 1)), groups=nIn // 8, bn_acti=True)

        self.dconv3x1_3_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1, padding=(int(d / 2 + 1), 0),
                                 dilation=(int(d / 2 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x3_3_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1, padding=(0, int(d / 2 + 1)),
                                 dilation=(1, int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)
        self.dconv5x1_3_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1, padding=(2 * int(d / 2 + 1), 0),
                                 dilation=(int(d / 2 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x5_3_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1, padding=(0, 2 * int(d / 2 + 1)),
                                 dilation=(1, int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)
        self.dconv7x1_3_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1, padding=(3 * int(d / 2 + 1), 0),
                                 dilation=(int(d / 2 + 1), 1), groups=nIn // 16, bn_acti=True)
        self.dconv1x7_3_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1, padding=(0, 3 * int(d / 2 + 1)),
                                 dilation=(1, int(d / 2 + 1)), groups=nIn // 8, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)
        self.bnlast = nn.BatchNorm2d(nIn)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv3x1_1_1(inp);
        o1_1 = self.dconv1x3_1_1(o1_1)
        o1_2 = self.dconv5x1_1_2(o1_1);
        o1_2 = self.dconv1x5_1_2(o1_2)
        o1_3 = self.dconv7x1_1_3(o1_2);
        o1_3 = self.dconv1x7_1_3(o1_3)

        o2_1 = self.dconv3x1_2_1(inp);
        o2_1 = self.dconv1x3_2_1(o2_1)
        o2_2 = self.dconv5x1_2_2(o2_1);
        o2_2 = self.dconv1x5_2_2(o2_2)
        o2_3 = self.dconv7x1_2_3(o2_2);
        o2_3 = self.dconv1x7_2_3(o2_3)

        o3_1 = self.dconv3x1_3_1(inp);
        o3_1 = self.dconv1x3_3_1(o3_1)
        o3_2 = self.dconv5x1_3_2(o3_1);
        o3_2 = self.dconv1x5_3_2(o3_2)
        o3_3 = self.dconv7x1_3_3(o3_2);
        o3_3 = self.dconv1x7_3_3(o3_3)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        output = torch.cat([inp, ad1, ad2, ad3], 1)
        output = self.conv1x1(output)
        output = self.bn_relu_2(output)

        return output + input


class DDSCDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DDSCDecoderBlock, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(in_ch, in_ch // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_ch // 2)
        self.block1 = nn.Sequential(
            DSConv(in_ch // 2, in_ch // 2, kernel_size=3, extend_scope=3, morph=0, if_offset=True),
            nn.BatchNorm2d(in_ch // 2))
        self.block2 = nn.Sequential(
            DSConv(in_ch // 2, in_ch // 2, kernel_size=5, extend_scope=5, morph=1, if_offset=True),
            nn.BatchNorm2d(in_ch // 2))
        self.block3 = nn.Sequential(nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=7, padding=3, bias=False),
                                    nn.BatchNorm2d(in_ch // 2))
        self.conv3 = nn.Conv2d(3 * in_ch // 2, out_ch, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.deconv2(x)
        x = nonlinearity(self.norm2(x))
        x1 = nonlinearity(self.block1(x))
        x2 = nonlinearity(self.block2(x))
        x3 = nonlinearity(self.block3(x))
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv3(x)
        x = nonlinearity(self.norm3(x))
        return x


# --- 主模型 StripUnet ---

class StripUnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        """
        StripUnet 模型
        Args:
            n_channels (int): 输入通道数 (默认为3, 对应RGB图像)。
            n_classes (int): 输出类别数 (对于二分类任务，应为1)。
            pretrained (bool): 如果为True, 使用在ImageNet上预训练的ResNet-34骨干网络。
        """
        super(StripUnet, self).__init__()
        filters = [64, 128, 256, 512]

        # --- 编码器 (Encoder) ---
        base_model = resnet34(weights=None)
        if pretrained:
            backbone_weight_path = './pretrain_weight/resnet34-b627a593.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")
            try:
                state_dict = torch.load(backbone_weight_path)
                base_model.load_state_dict(state_dict, strict=False)
                print("Backbone weights loaded successfully from local file.")
            except FileNotFoundError:
                print(f"Error: Backbone weight file not found at '{backbone_weight_path}'.")
                print("Falling back to torchvision's automatic download...")
                base_model = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.firstconv = base_model.conv1
        self.firstbn = base_model.bn1
        self.firstrelu = base_model.relu
        self.firstmaxpool = base_model.maxpool
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        # --- 中间模块 ---
        self.msff = MSFF(512)
        self.dam1 = DAM(256, 64)
        self.dam2 = DAM(128, 128)
        self.dam3 = DAM(64, 256)
        self.dam4 = DAM(32, 512)
        self.convsam1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.convsam2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.convsam3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bnsam1 = nn.BatchNorm2d(128)
        self.bnsam2 = nn.BatchNorm2d(256)
        self.bnsam3 = nn.BatchNorm2d(512)
        self.jiangwei111 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.jiangwei211 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.jiangwei311 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

        # --- 解码器 (Decoder) ---
        self.sfem4 = sfem(512, d=1)
        self.sfem3 = sfem(256, d=2)
        self.sfem2 = sfem(128, d=4)
        self.sfem1 = sfem(64, d=8)
        self.dconv4 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.decoder4 = DDSCDecoderBlock(filters[3], filters[2])
        self.decoder3 = DDSCDecoderBlock(filters[2], filters[1])
        self.decoder2 = DDSCDecoderBlock(filters[1], filters[0])
        self.decoder1 = DDSCDecoderBlock(filters[0], filters[0])

        # --- 输出层 ---
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x_first = self.firstrelu(self.firstbn(x))
        x = self.firstmaxpool(x_first)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Attention
        sam1 = self.dam1(e1, x)
        sam11 = nonlinearity(self.bnsam1(self.convsam1(sam1)))
        sam2 = self.dam2(e2, sam11)
        sam22 = nonlinearity(self.bnsam2(self.convsam2(sam2)))
        sam3 = self.dam3(e3, sam22)
        sam33 = nonlinearity(self.bnsam3(self.convsam3(sam3)))
        sam4 = self.dam4(e4, sam33)

        # Center
        center = self.msff(e4 + sam4)

        # SFEM
        c4 = self.sfem4(sam4 + center)
        c44 = self.dconv4(self.jiangwei111(c4))
        c3 = self.sfem3(sam3 + c44)
        c33 = self.dconv3(self.jiangwei211(c3))
        c2 = self.sfem2(sam2 + c33)
        c22 = self.dconv2(self.jiangwei311(c2))
        c1 = self.sfem1(sam1 + c22)

        # Decoder
        d4 = center + c4
        d3 = self.decoder4(d4) + c3
        d2 = self.decoder3(d3) + c2
        d1 = self.decoder2(d2) + c1
        d0 = self.decoder1(d1)

        # Output
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # 移除最后的 Sigmoid
        return out


if __name__ == '__main__':
    # --- 维度测试 ---
    model = StripUnet(n_channels=3, n_classes=1, pretrained=True)
    model.eval()

    batch_size = 2
    input_height, input_width = 1024, 1024
    input_tensor = torch.randn(batch_size, 3, input_height, input_width)

    with torch.no_grad():
        output = model(input_tensor)

    print(f"\n--- Dimension Test ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, 1, input_height, input_width)
    print("\nDimension test passed! Output shape is as expected.")

