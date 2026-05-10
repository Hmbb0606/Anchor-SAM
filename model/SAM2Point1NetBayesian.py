# -*- coding: utf-8 -*-
"""
SAM 2.1 编码器-解码器完整分割模型

本文件整合了 SAM 2.1 编码器、自定义的 MSDA 融合模块以及一个 FPN 风格的解码器，
形成一个完整的、端到端的图像分割网络。

功能包括：
1.  完整的、自包含的模型定义。
2.  支持通过参数选择 'tiny', 'small', 'base', 'large' 四种不同大小的编码器。
3.  在模型初始化时尝试加载编码器的预训练权重。
4.  解码器使用 DySample 进行动态上采样，并与 FPN 特征逐级融合。
5.  最终模型 (SAM2Point1Net) 将编码器和解码器封装在一起。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from functools import partial
import os
import warnings

warnings.filterwarnings("ignore", "Now, we support DCNv4 in InternImage.")

# --- 导入您提供的自定义模块 ---
# 请确保 StageFour_Output.py 和 Dysample.py 与此文件位于同一目录下
try:
    from .StageFour_Output import MSDAFusionModule, DCNv3Wrapper
    from .Dysample import DySample
    # === 新增模块导入 ===
    from .DSABayesian import FPN_DSA_Module
    # 新增审稿人建议的上下问尺度聚合模块 DLinkBlock
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 StageFour_Output.py, Dysample.py 和 DSABayesian.py 文件与本脚本位于同一目录。")
    exit()


# ==============================================================================
# 辅助工具模块 (Helper Utilities)
# (此部分代码与您提供的版本保持一致)
# ==============================================================================

class DropPath(nn.Module):
    """
    Stochastic Depth a.k.a. DropPath
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将特征图分割成不重叠的窗口
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int],
                       hw: Tuple[int, int]) -> torch.Tensor:
    """
    将窗口合并回原始的特征图
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    """
    图像到 Patch 的嵌入层
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (7, 7),
            stride: Tuple[int, int] = (4, 4),
            padding: Tuple[int, int] = (3, 3),
            in_chans: int = 3,
            embed_dim: int = 96,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        return x


# ==============================================================================
# Hiera 核心模块 (Hiera Core Modules)
# (此部分代码与您提供的版本保持一致)
# ==============================================================================

def do_pool(x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    return x


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
            norm_layer: Union[nn.Module, str] = "LayerNorm",
            q_stride: Tuple[int, int] = None,
            window_size: int = 0,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    def __init__(
            self,
            embed_dim: int = 96,
            num_heads: int = 1,
            drop_path_rate: float = 0.0,
            stages: Tuple[int, ...] = (2, 3, 16, 3),
            dim_mul: float = 2.0,
            head_mul: float = 2.0,
            window_spec: Tuple[int, ...] = (8, 4, 14, 7),
            global_att_blocks: Tuple[int, ...] = None,
            window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.stages = stages
        self.window_spec = window_spec
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)

        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(stages))]

        self.blocks = nn.ModuleList()
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]]

        cur_stage_idx = 1
        for i in range(sum(stages)):
            dim_out = int(embed_dim * dim_mul) if i - 1 in self.stage_ends else embed_dim
            window_size = window_spec[cur_stage_idx - 1]
            if global_att_blocks and i in global_att_blocks:
                window_size = 0
            block = MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i],
                q_stride=(2, 2) if i in q_pool_blocks else None, window_size=window_size)
            embed_dim = dim_out
            if i - 1 in self.stage_ends:
                num_heads = int(num_heads * head_mul)
                cur_stage_idx += 1
            self.blocks.append(block)

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic", align_corners=False)
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        stage_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                stage_outputs.append(x.permute(0, 3, 1, 2))
        return stage_outputs[::-1]


# ==============================================================================
# 颈部网络 (Neck)
# (此部分代码与您提供的版本保持一致)
# ==============================================================================

class FpnNeck(nn.Module):
    def __init__(self, d_model: int, backbone_channel_list: List[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module("conv", nn.Conv2d(dim, d_model, kernel_size=1))
            self.convs.append(current)

    def forward(self, xs: List[torch.Tensor]):
        prev_features = self.convs[0](xs[0])
        outputs = [prev_features]
        for i in range(1, len(xs)):
            lateral_features = self.convs[i](xs[i])
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            prev_features = lateral_features + top_down_features
            outputs.append(prev_features)
        return outputs[::-1]


# ==============================================================================
# SAM 2.1 编码器模型 (SAM 2.1 Encoder Model)
# ==============================================================================

class SAM2Encoder(nn.Module):
    def __init__(self, model_size: str = 'base', pretrained_weights_path: str = None, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # --- (核心修改) ---
        # 扩展 configs 字典以包含所有尺寸特定的参数
        configs = {
            'tiny': {'embed_dim': 96, 'num_heads': 1, 'stages': (1, 2, 7, 2),
                     'backbone_channel_list': [768, 384, 192, 96], 'global_att_blocks': [5, 7, 9],
                     'pos_embed_size': (7, 7), 'window_spec': (8, 4, 14, 7)},
            'small': {'embed_dim': 96, 'num_heads': 1, 'stages': (1, 2, 11, 2),
                      'backbone_channel_list': [768, 384, 192, 96], 'global_att_blocks': [7, 10, 13],
                      'pos_embed_size': (7, 7), 'window_spec': (8, 4, 14, 7)},
            'base': {'embed_dim': 112, 'num_heads': 2, 'stages': (2, 3, 16, 3),
                     'backbone_channel_list': [896, 448, 224, 112], 'global_att_blocks': None,
                     'pos_embed_size': (14, 14), 'window_spec': (8, 4, 14, 7)},
            'large': {'embed_dim': 144, 'num_heads': 2, 'stages': (2, 6, 36, 4),
                      'backbone_channel_list': [1152, 576, 288, 144], 'global_att_blocks': [23, 33, 43],
                      'pos_embed_size': (7, 7), 'window_spec': (8, 4, 16, 8)}
        }

        if model_size not in configs:
            raise ValueError(f"Model size '{model_size}' not recognized. Available options: {list(configs.keys())}")
        config = configs[model_size]

        # --- (核心修改) ---
        # 实例化 Hiera 时传入动态的配置参数
        self.trunk = Hiera(embed_dim=config['embed_dim'], num_heads=config['num_heads'], stages=config['stages'],
                           global_att_blocks=config['global_att_blocks'],
                           window_pos_embed_bkg_spatial_size=config['pos_embed_size'],
                           window_spec=config['window_spec'])

        self.neck = FpnNeck(d_model=256, backbone_channel_list=config['backbone_channel_list'])
        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)

    def _load_pretrained_weights(self, path: str):
        if not os.path.exists(path):
            if self.verbose: print(f"--- 警告 ---: 预训练权重文件不存在于路径: {path}\n模型将使用随机初始化的权重。")
            return
        if self.verbose: print(f"正在从 {path} 加载编码器权重...")
        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict: state_dict = state_dict['model']
        encoder_state_dict = {}
        prefix = 'image_encoder.'
        for k, v in state_dict.items():
            if k.startswith(prefix): encoder_state_dict[k[len(prefix):]] = v
        if not encoder_state_dict:
            if self.verbose: print("--- 信息 ---: 权重文件中未找到 'image_encoder.' 前缀，将尝试直接加载。")
            encoder_state_dict = state_dict
        incompatible_keys = self.load_state_dict(encoder_state_dict, strict=False)
        if self.verbose:
            print("\n--- 编码器权重加载分析 ---")
            if incompatible_keys.missing_keys: print(
                f"模型中有 {len(incompatible_keys.missing_keys)} 个层在权重中未找到。")
            if incompatible_keys.unexpected_keys: print(
                f"权重中有 {len(incompatible_keys.unexpected_keys)} 个层在模型中未找到。")
            if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys: print(
                "所有编码器权重已成功匹配并加载！")

    def forward(self, x: torch.Tensor):
        if self.verbose:
            print("\n--- 开始编码器前向传播 ---")
            print(f"输入图像维度: {x.shape}")
        trunk_outputs = self.trunk(x)
        stage4_out, stage3_out, stage2_out, stage1_out = trunk_outputs
        if self.verbose:
            print("\nHiera 主干网络输出:")
            print(f"  Stage 1 输出维度: {stage1_out.shape}")
            print(f"  Stage 2 输出维度: {stage2_out.shape}")
            print(f"  Stage 3 输出维度: {stage3_out.shape}")
            print(f"  Stage 4 输出维度: {stage4_out.shape}")
        neck_outputs = self.neck(trunk_outputs)
        if self.verbose:
            fpn_out1, fpn_out2, fpn_out3, fpn_out4 = neck_outputs
            print("\nFPN 颈部网络输出:")
            print(f"  FPN Level 1 输出维度: {fpn_out1.shape} (对应 stage 1)")
            print(f"  FPN Level 2 输出维度: {fpn_out2.shape} (对应 stage 2)")
            print(f"  FPN Level 3 输出维度: {fpn_out3.shape} (对应 stage 3)")
            print(f"  FPN Level 4 输出维度: {fpn_out4.shape} (对应 stage 4)")
        return {"trunk_outputs": trunk_outputs, "fpn_outputs": neck_outputs}


# ==============================================================================
# === 新增模块: FPN风格的解码器 (NEW: FPN-Style Decoder) ===
# ==============================================================================

# class DecoderBlock(nn.Module):
#     """一个基本的解码器块，用于特征融合和通道数调整。"""
#
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.block(x)

class DecoderBlock(nn.Module):
    """
    一个使用 DCNv3 作为核心的、升级版的解码器块。
    它首先使用 DCNv3Wrapper 进行可变形的空间特征提取，
    然后通过一个 1x1 卷积来调整通道数。
    """
    def __init__(self, in_channels: int, out_channels: int, group_dim: int = 16):
        super().__init__()
        # DCNv3 的组数 G = C / C'，这里 group_dim 对应 C'
        # 我们需要确保组数是有效的
        num_groups = in_channels // group_dim if in_channels > 0 and in_channels % group_dim == 0 else in_channels // 16

        self.block = nn.Sequential(
            # 步骤1: 使用 DCNv3Wrapper 进行可变形卷积，不改变通道数
            DCNv3Wrapper(
                channels=in_channels,
                kernel_size=3,
                stride=1,
                pad=1,
                dilation=1,
                group=num_groups if num_groups > 0 else 1 # 确保组数至少为1
            ),
            # DCNv3 内部通常自带 Norm 和 Act，但为了结构稳定，我们可以在外部再添加
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 步骤2: 使用 1x1 卷积来调整通道数
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SAM2Decoder(nn.Module):
    """
    根据您的设计实现的解码器。
    它将编码器的输出作为输入，通过逐级上采样和与FPN特征的融合来生成最终的分割图。
    """

    def __init__(self, fpn_out_channels: int = 256, stage4_in_channels: int = 768, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # --- 模块(1): MSDAFusionModule for stage 4 ---
        # input_resolution 假设为 32x32
        self.msda_fusion = MSDAFusionModule(in_channels=stage4_in_channels, input_resolution=(32, 32))
        # self.msda_fusion = Dblock(stage4_in_channels)

        # --- Level 4 解码 (32x32 -> 64x64) ---
        # 拼接 MSDA(stage4) [768] 和 FPN4 [256] -> 1024
        self.decoder_block4 = DecoderBlock(in_channels=stage4_in_channels + fpn_out_channels, out_channels=512)
        self.upsample4 = DySample(in_channels=512, scale=2)

        # --- Level 3 解码 (64x64 -> 128x128) ---
        # 拼接 upsample4 [512] 和 FPN3 [256] -> 768
        self.decoder_block3 = DecoderBlock(in_channels=512 + fpn_out_channels, out_channels=256)
        self.upsample3 = DySample(in_channels=256, scale=2)

        # --- Level 2 解码 (128x128 -> 256x256) ---
        # 拼接 upsample3 [256] 和 FPN2 [256] -> 512
        self.decoder_block2 = DecoderBlock(in_channels=256 + fpn_out_channels, out_channels=128)
        self.upsample2 = DySample(in_channels=128, scale=2)

        # --- Level 1 解码 (256x256 -> 512x512) ---
        # 拼接 upsample2 [128] 和 FPN1 [256] -> 384
        self.decoder_block1 = DecoderBlock(in_channels=128 + fpn_out_channels, out_channels=64)
        self.upsample1 = DySample(in_channels=64, scale=2)  # 256 -> 512

        # --- 最终上采样 (512x512 -> 1024x1024) ---
        self.final_upsample = DySample(in_channels=64, scale=2)

        # --- 输出层 ---
        # 生成单通道分割图，无激活函数
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, encoder_features: dict) -> torch.Tensor:
        trunk_outputs = encoder_features["trunk_outputs"]
        fpn_outputs = encoder_features["fpn_outputs"]

        stage4_out, stage3_out, stage2_out, stage1_out = trunk_outputs
        fpn_out1, fpn_out2, fpn_out3, fpn_out4 = fpn_outputs

        if self.verbose: print("\n--- 开始解码器前向传播 ---")

        # --- Stage 4 处理 ---
        x = self.msda_fusion(stage4_out)  # [B, 768, 32, 32]
        if self.verbose: print(f"MSDA 融合后维度: {x.shape}")

        # --- Level 4 to 3 ---
        x = torch.cat([x, fpn_out4], dim=1)  # [B, 768+256=1024, 32, 32]
        x = self.decoder_block4(x)  # [B, 512, 32, 32]
        x = self.upsample4(x)  # [B, 512, 64, 64]
        if self.verbose: print(f"Level 4 解码后 (上采样至64x64) 维度: {x.shape}")

        # --- Level 3 to 2 ---
        x = torch.cat([x, fpn_out3], dim=1)  # [B, 512+256=768, 64, 64]
        x = self.decoder_block3(x)  # [B, 256, 64, 64]
        x = self.upsample3(x)  # [B, 256, 128, 128]
        if self.verbose: print(f"Level 3 解码后 (上采样至128x128) 维度: {x.shape}")

        # --- Level 2 to 1 ---
        x = torch.cat([x, fpn_out2], dim=1)  # [B, 256+256=512, 128, 128]
        x = self.decoder_block2(x)  # [B, 128, 128, 128]
        x = self.upsample2(x)  # [B, 128, 256, 256]
        if self.verbose: print(f"Level 2 解码后 (上采样至256x256) 维度: {x.shape}")

        # --- Level 1 to final ---
        x = torch.cat([x, fpn_out1], dim=1)  # [B, 128+256=384, 256, 256]
        x = self.decoder_block1(x)  # [B, 64, 256, 256]
        x = self.upsample1(x)  # [B, 64, 512, 512]
        if self.verbose: print(f"Level 1 解码后 (上采样至512x512) 维度: {x.shape}")

        # --- Final upsample and output ---
        x = self.final_upsample(x)  # [B, 64, 1024, 1024]
        logits = self.output_conv(x)  # [B, 1, 1024, 1024]
        if self.verbose: print(f"最终输出 Logits 维度: {logits.shape}")

        return logits


# ==============================================================================
# === 新增模块: 最终的完整分割模型 (NEW: Final Segmentation Model) ===
# ==============================================================================

class SAM2Point1Net(nn.Module):
    def __init__(self,
                 model_size: str = 'small',
                 pretrained_weights_path: str = None,
                 verbose: bool = True):
        super().__init__()

        # small 模型的 stage4 输出通道数为 768
        # 如果切换模型尺寸，需要手动调整这里的 `stage4_in_channels`
        # Hiera 'small': [768, 384, 192, 96]
        # Hiera 'base+': [896, 448, 224, 112]
        stage4_channels_map = {'tiny': 768, 'small': 768, 'base': 896, 'large': 1152}
        stage4_in_channels = stage4_channels_map.get(model_size)

        if stage4_in_channels is None:
            raise ValueError(f"未知的模型尺寸 '{model_size}' 无法确定 stage4 通道数。")

        self.encoder = SAM2Encoder(model_size=model_size, pretrained_weights_path=pretrained_weights_path,
                                   verbose=verbose)

        # === 新增模块实例化: FPN_DSA_Module ===
        # 在这里灵活定义DSA模块在FPN每一层的超参数
        # FPN输出通道数固定为256
        fpn_d_model = 256
        dsa_configs_list = [
            {'kernel_size': 7, 'group': 4},  # for FPN Level 1 (大特征图，用较大kernel)
            {'kernel_size': 7, 'group': 4},  # for FPN Level 2
            {'kernel_size': 5, 'group': 8},  # for FPN Level 3
            {'kernel_size': 5, 'group': 8},  # for FPN Level 4 (小特征图，用较小kernel)
        ]
        self.fpn_dsa_attention = FPN_DSA_Module(d_model=fpn_d_model, dsa_configs=dsa_configs_list)

        self.decoder = SAM2Decoder(stage4_in_channels=stage4_in_channels, verbose=verbose)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = self.encoder(x)

        # === 在编码器和解码器之间插入FPN注意力处理 ===
        # 1. 提取FPN输出
        fpn_outputs = encoder_features["fpn_outputs"]
        # 2. 应用FPN_DSA_Module
        processed_fpn_outputs = self.fpn_dsa_attention(fpn_outputs)
        # 3. 将处理后的特征放回字典，供解码器使用
        encoder_features["fpn_outputs"] = processed_fpn_outputs

        output_mask = self.decoder(encoder_features)
        return output_mask


# ==============================================================================
# 测试代码 (Test Execution)
# ==============================================================================

if __name__ == '__main__':
    # --- 1. 设置设备 ---
    # MSDAFusionModule 中的 DCNv3 需要 CUDA
    if not torch.cuda.is_available():
        print("错误: 本测试脚本需要一个支持 CUDA 的 GPU 来运行 DCNv3。")
        exit()
    device = torch.device("cuda")
    print(f"使用设备: {device}")

    # --- 2. 定义模型尺寸和权重路径 ---
    MODEL_SIZE = 'large'
    WEIGHT_PATH = "/home/wenhai-li/Code/SAM-RoadSegment/pretrain_weight/sam2.1_hiera_large.pt"

    # --- 3. 初始化完整的模型 ---
    # verbose=False 可以关闭详细的维度打印
    full_model = SAM2Point1Net(model_size=MODEL_SIZE, pretrained_weights_path=WEIGHT_PATH, verbose=True)
    full_model.to(device)
    full_model.eval()

    # --- 4. 创建一个假的输入图像 ---
    dummy_image = torch.randn(1, 3, 1024, 1024).to(device)

    # --- 5. 执行完整的端到端前向传播 ---
    print("\n" + "=" * 25 + " 开始端到端前向传播 " + "=" * 25)
    with torch.no_grad():
        final_mask = full_model(dummy_image)
    print("=" * 25 + " 端到端前向传播结束 " + "=" * 25 + "\n")

    # --- 6. 验证最终输出 ---
    print("模型最终输出验证:")
    print(f"Final output shape: {final_mask.shape}")

    # 检查输出形状是否符合预期
    expected_shape = torch.Size([1, 1, 1024, 1024])
    if final_mask.shape == expected_shape:
        print(f"输出形状符合预期 {expected_shape}，测试成功！")
    else:
        print(f"错误：输出形状 {final_mask.shape} 与预期的 {expected_shape} 不符。")
