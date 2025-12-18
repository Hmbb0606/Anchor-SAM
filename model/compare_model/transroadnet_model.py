'''
@ARTICLE{TransRoadNet,
  author={Yang, Zhigang and Zhou, Daoxiang and Yang, Ying and Zhang, Jiapeng and Chen, Zehua},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={{TransRoadNet}: A Novel Road Extraction Method for Remote Sensing Images via Combining High-Level Semantic Feature and Context},
  year={2022},
  volume={19},
  number={},
  pages={1--5}
}
'''

import torch
from torch import nn, einsum
import numpy as np
from torchvision.models import resnet34, ResNet34_Weights
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F
from functools import partial


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
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False),
                requires_grad=False)
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True),
                requires_grad=False)

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
        nw_h, nw_w = n_h // self.window_size, n_w // self.window_size
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
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim, heads=heads, head_dim=head_dim,
                                                                     shifted=shifted, window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)
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


class SCA_Blocak(nn.Module):
    def __init__(self, inchannel=256, h=64, w=64):
        super(SCA_Blocak, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

    def forward(self, x):
        meanh1 = torch.mean(self.avg_pool_x(x).permute(0, 1, 3, 2), dim=1, keepdim=True)
        meanh2, _ = torch.max(meanh1, dim=1, keepdim=True)
        meanh = self.conv1x1(torch.cat([meanh1, meanh2], dim=1))
        meanw1 = torch.mean(self.avg_pool_y(x), dim=1, keepdim=True)
        meanw2, _ = torch.max(meanw1, dim=1, keepdim=True)
        meanw = self.conv1x1(torch.cat([meanw1, meanw2], dim=1))
        s_h = self.sigmoid(meanh.permute(0, 1, 3, 2))
        s_w = self.sigmoid(meanw)
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize=3, stride=1, pad=1, dilation=1, groups=1, has_bn=True,
                 norm_layer=nn.BatchNorm2d, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation,
                              groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d, scale=2):
        super(DecoderBlock, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                   has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                   has_bias=False)
        self.scale = scale
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


# --- Main Model: TransRoadNet ---

class TransRoadNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True,
                 hidden_dim=64, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), head_dim=32, window_size=8,
                 downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        """
        TransRoadNet Model
        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            pretrained (bool): If True, use pretrained weights for the ResNet-34 backbone.
            ... (other model-specific args)
        """
        super().__init__()

        # --- Encoder (Backbone) ---
        # Note: input channel (n_channels) is handled by the first conv layer if it's not 3.
        # This implementation assumes n_channels=3 for pretrained weights.
        base_model = resnet34(weights=None)
        if pretrained:
            backbone_weight_path = './pretrain_weight/resnet34-b627a593.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")
            try:
                state_dict = torch.load(backbone_weight_path)
                base_model.load_state_dict(state_dict, strict=False)
                print("Backbone weights loaded successfully from local file.")
            except FileNotFoundError:
                print(f"Warning: Backbone weight file not found at '{backbone_weight_path}'.")
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

        # --- Transformer and Fusion Modules ---
        self.bn = nn.BatchNorm2d(512)
        self.SCA = SCA_Blocak(inchannel=256, h=64, w=64)  # Assuming 1024 input, feature map is 64x64 at this stage
        self.stage4 = StageModule(in_channels=256, hidden_dimension=512, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)

        # Context Aggregation
        self.up2 = partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = partial(F.interpolate, scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = partial(F.interpolate, scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = partial(F.interpolate, scale_factor=16, mode='bilinear', align_corners=True)

        self.conv256 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv128 = nn.Conv2d(640, 128, 3, padding=1)
        self.conv64 = nn.Conv2d(576, 64, 3, padding=1)
        self.conv32 = nn.Conv2d(576, 64, 3, padding=1)

        # --- Decoder ---
        self.decoder5 = DecoderBlock(512, 256, scale=2)
        self.decoder4 = DecoderBlock(256, 128, scale=2)
        self.decoder3 = DecoderBlock(128, 64, scale=2)
        self.decoder2 = DecoderBlock(64, 64, scale=2)

        # --- Final Output Layers ---
        self.finaldeconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.finalconv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # --- Encoder ---
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1_pool = self.firstmaxpool(e0)
        e1 = self.encoder1(e1_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # --- Transformer Branch ---
        x3_1 = self.SCA(e3)
        x3_1 = self.stage4(x3_1)

        # Adaptive Fusion
        fused_e4 = self.relu(self.bn(self.conv1(self.gamma * e4 + (1 - self.gamma) * x3_1)))

        # --- Context Aggregation ---
        ctx_2 = self.relu(self.conv256(torch.cat((self.up2(x3_1), e3), dim=1)))
        ctx_3 = self.relu(self.conv128(torch.cat((self.up4(x3_1), e2), dim=1)))
        ctx_4 = self.relu(self.conv64(torch.cat((self.up8(x3_1), e1), dim=1)))
        ctx_5 = self.relu(self.conv32(torch.cat((self.up16(x3_1), e0), dim=1)))

        # --- Decoder with Skip Connections ---
        d4 = self.decoder5(fused_e4) + ctx_2
        d3 = self.decoder4(d4) + ctx_3
        d2 = self.decoder3(d3) + ctx_4
        d1 = self.decoder2(d2) + ctx_5

        # --- Final Layers ---
        out = self.finaldeconv1(d1)
        out = self.relu(self.finalconv1(out))
        out = self.relu(self.finalconv2(out))
        out = self.finalconv3(out)

        # Return logits without sigmoid
        return out


if __name__ == '__main__':
    # --- Dimension Test ---
    # Create a model instance (pretrained=True to test weight loading)
    # Note: Set pretrained=False if you don't have the weight file and want to avoid download.
    model = TransRoadNet(n_channels=3, n_classes=1, pretrained=True)
    model.eval()

    # Create a dummy input tensor
    batch_size = 1
    input_height, input_width = 1024, 1024
    input_tensor = torch.randn(batch_size, 3, input_height, input_width)

    # Model forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Print shapes
    print(f"\n--- Dimension Test ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == (batch_size, 1, input_height, input_width)
    print("\nDimension test passed! Output shape is as expected.")
