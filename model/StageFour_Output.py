import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

# --- 导入 DCNv3 ---
# 使用您指定的官方源码编译模块的导入路径
try:
    from .ops_dcnv3.modules.dcnv3 import DCNv3
except ImportError:
    raise ImportError("无法导入 DCNv3 模块。请确保 'ops_dcnv3' 在您的Python路径中，并且已经成功编译。")


# --- DCNv3 包装器 ---
class DCNv3Wrapper(nn.Module):
    """
    一个包装器，用于处理官方DCNv3模块的 'channels-last' (N, H, W, C) 输入/输出格式，
    使其能与PyTorch标准的 'channels-first' (N, C, H, W) 工作流无缝集成。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dcnv3 = DCNv3(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的格式是: (N, C, H, W)
        # DCNv3 期望的格式是: (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # DCNv3 处理
        x = self.dcnv3(x)
        # DCNv3 输出的格式是: (N, H, W, C)
        # 将其转换回标准的 (N, C, H, W) 格式
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# --- Swin Transformer 模块及其辅助组件 ---
# Swin Transformer 块是通道注意力模块的核心

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图划分为不重叠的窗口。
    Args:
        x: 输入张量，形状为 (B, H, W, C).
        window_size (int): 窗口的高度和宽度。
    Returns:
        窗口化的张量，形状为 (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口化的张量还原为原始特征图。
    Args:
        windows: 窗口化张量，形状为 (num_windows*B, window_size, window_size, C).
        window_size (int): 窗口的高度和宽度。
        H (int): 原始特征图的高度。
        W (int): 原始特征图的宽度。
    Returns:
        还原后的张量，形状为 (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ 窗口内的多头自注意力模块 (W-MSA / SW-MSA) """

    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer 基础模块 """

    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()  # Placeholder for DropPath, can be implemented if needed
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# --- 核心模块实现 ---

class ParallelMultiScaleConvModule(nn.Module):
    """
    模块(1): 并行的全通道多尺度可变形卷积(DCNv3)模块
    该模块使用您从官方源码编译的DCNv3层（通过DCNv3Wrapper）来捕捉多尺度动态特征。
    """

    def __init__(self, in_channels: int, group_dim: int = 16):
        super().__init__()
        self.channels = in_channels
        dilations = [1, 3, 5, 7]
        # 根据论文，组数 G = C / C'
        num_groups = in_channels // group_dim

        # 创建四个并行的DCNv3分支
        self.branches = nn.ModuleList([
            DCNv3Wrapper(
                channels=self.channels,
                kernel_size=3,
                dw_kernel_size=3,
                stride=1,
                pad=dilation,  # 为保持分辨率，pad应等于dilation
                dilation=dilation,
                group=num_groups,
            ) for dilation in dilations
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
        Returns:
            一个包含四个分支输出的列表，每个元素的形状仍为 [B, C, H, W]
        """
        return [branch(x) for branch in self.branches]


class InteractiveSpatialAttentionFusionModule(nn.Module):
    """
    模块(2): 交互式空间注意力融合模块
    该模块首先融合所有尺度的特征，然后通过一个轻量级卷积网络
    生成空间注意力图，最后对原始特征进行加权求和。
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.total_channels = in_channels * 4

        # 用于生成注意力图的卷积网络
        self.attention_generator = nn.Sequential(
            # 1x1 Conv 降维
            nn.Conv2d(self.total_channels, self.total_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.total_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            # 3x3 Conv 提取空间上下文
            nn.Conv2d(self.total_channels // reduction_ratio, self.total_channels // reduction_ratio, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.total_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            # 1x1 Conv 生成最终的4通道注意力图
            nn.Conv2d(self.total_channels // reduction_ratio, 4, kernel_size=1),
            nn.Sigmoid()  # 使用Sigmoid进行归一化
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_list: 包含四个分支输出的列表.
        Returns:
            空间融合后的特征图 F_spatial，形状为 [B, C, H, W]
        """
        # 1. 拼接
        # F_concat 维度: [B, 4 * C, H, W]
        f_concat = torch.cat(feature_list, dim=1)

        # 2. 生成注意力图
        # attention_maps 维度: [B, 4, H, W]
        attention_maps = self.attention_generator(f_concat)

        # 3. 拆分
        # attn_map_i 维度: [B, 1, H, W]
        attn_map_1, attn_map_2, attn_map_3, attn_map_4 = torch.chunk(attention_maps, chunks=4, dim=1)

        # 4. & 5. 加权与求和
        # 将权重图广播到与特征图相同的通道数进行相乘
        f_weighted_1 = feature_list[0] * attn_map_1
        f_weighted_2 = feature_list[1] * attn_map_2
        f_weighted_3 = feature_list[2] * attn_map_3
        f_weighted_4 = feature_list[3] * attn_map_4

        f_spatial = f_weighted_1 + f_weighted_2 + f_weighted_3 + f_weighted_4
        return f_spatial


class SwinChannelAttentionFusionModule(nn.Module):
    """
    模块(3): 基于Swin Transformer的通道注意力融合模块
    该模块将所有尺度特征融合，利用Swin Transformer强大的空间上下文
    建模能力提炼全局信息，然后解码为各分支的通道注意力权重。
    """

    def __init__(self, in_channels: int, input_resolution: Tuple[int, int],
                 swin_embed_dim: int = 256, num_heads: int = 8,
                 window_size: int = 8, num_swin_blocks: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_branches = 4
        self.total_channels = self.in_channels * self.num_branches

        # 1. & 2. & 3. 通道降维与特征映射
        self.embedding = nn.Sequential(
            nn.Conv2d(self.total_channels, swin_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(swin_embed_dim),
            nn.GELU()
        )

        # 4. Swin Transformer 块
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=swin_embed_dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                # 交替使用常规窗口和移位窗口
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            ) for i in range(num_swin_blocks)
        ])

        # 5. 全局信息编码
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 6. 解码生成权重
        self.weight_decoder = nn.Sequential(
            nn.Linear(swin_embed_dim, self.total_channels),
            nn.Sigmoid()
        )

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = feature_list[0].shape

        # 1. 拼接
        # f_concat 维度: [B, 4 * C, H, W]
        f_concat = torch.cat(feature_list, dim=1)

        # 2. & 3. 嵌入
        # f_mapped 维度: [B, swin_embed_dim, H, W]
        f_mapped = self.embedding(f_concat)

        # 4. Swin Transformer 处理
        # 首先需要将维度从 [B, C, H, W] 转换为 [B, L, C]
        f_swin_input = f_mapped.flatten(2).transpose(1, 2)
        for block in self.swin_blocks:
            f_swin_input = block(f_swin_input)

        # 将维度转换回来 [B, swin_embed_dim, H, W]
        f_swin = f_swin_input.transpose(1, 2).view(B, -1, H, W)

        # 5. 全局池化
        # v_global 维度: [B, swin_embed_dim, 1, 1]
        v_global = self.global_pool(f_swin)
        # squeeze 后维度: [B, swin_embed_dim]
        v_global_squeezed = v_global.squeeze(-1).squeeze(-1)

        # 6. 解码
        # w_vector 维度: [B, 4 * C]
        w_vector = self.weight_decoder(v_global_squeezed)

        # 7. 权重分配与加权融合
        # w_i 维度: [B, C, 1, 1]
        w1, w2, w3, w4 = torch.chunk(w_vector, chunks=self.num_branches, dim=1)
        w1, w2, w3, w4 = [w.view(B, C, 1, 1) for w in [w1, w2, w3, w4]]

        f_weighted_1 = feature_list[0] * w1
        f_weighted_2 = feature_list[1] * w2
        f_weighted_3 = feature_list[2] * w3
        f_weighted_4 = feature_list[3] * w4

        f_channel = f_weighted_1 + f_weighted_2 + f_weighted_3 + f_weighted_4

        return f_channel


# --- 最终集成的完整模块 ---
class MSDAFusionModule(nn.Module):
    """
    集成了并行卷积、空间融合、通道融合的最终模块。
    """

    def __init__(self, in_channels: int, input_resolution: Tuple[int, int], group_dim: int = 16, **kwargs):
        super().__init__()
        self.parallel_conv = ParallelMultiScaleConvModule(in_channels=in_channels, group_dim=group_dim)
        self.spatial_fusion = InteractiveSpatialAttentionFusionModule(in_channels=in_channels)
        self.channel_fusion = SwinChannelAttentionFusionModule(in_channels=in_channels,
                                                               input_resolution=input_resolution, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 并行多尺度特征提取
        multi_scale_features = self.parallel_conv(x)

        # 2. 并行进行空间和通道注意力融合
        f_spatial = self.spatial_fusion(multi_scale_features)
        f_channel = self.channel_fusion(multi_scale_features)

        # 3. 最终合并
        output = f_spatial + f_channel

        return output

class MSDAFusionModule2(nn.Module):
    """
    集成了并行卷积、空间融合、通道融合的最终模块。
    """

    def __init__(self, in_channels: int, input_resolution: Tuple[int, int], group_dim: int = 16, **kwargs):
        super().__init__()
        self.parallel_conv = ParallelMultiScaleConvModule(in_channels=in_channels, group_dim=group_dim)
        self.spatial_fusion = InteractiveSpatialAttentionFusionModule(in_channels=in_channels)
        self.channel_fusion = SwinChannelAttentionFusionModule(in_channels=in_channels,
                                                               input_resolution=input_resolution, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 并行多尺度特征提取
        multi_scale_features = self.parallel_conv(x)

        # 2. 并行进行空间和通道注意力融合
        f_spatial = self.spatial_fusion(multi_scale_features)
        f_channel = self.channel_fusion(multi_scale_features)

        # 3. 最终合并
        # output = f_spatial + f_channel

        return f_spatial, f_channel


# --- 测试代码 ---
if __name__ == '__main__':
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("错误: 本测试脚本需要一个支持CUDA的GPU来运行DCNv3。")
        exit()

    device = torch.device("cuda")
    print(f"在设备上运行测试: {device}")


    def print_gpu_memory(stage=""):
        """打印当前和峰值GPU显存使用情况的工具函数"""
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"[{stage:^40}] GPU显存: {allocated:.2f} MB 已分配 / {peak:.2f} MB 峰值")


    # 定义输入张量的参数
    BATCH_SIZE = 2
    CHANNELS = 768
    HEIGHT = 32
    WIDTH = 32
    GROUP_DIM = 16  # C' in paper

    print_gpu_memory("初始状态")

    # 创建一个在GPU上的随机输入张量
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device=device)

    print("\n----------- 模块实例化并移至GPU -----------")
    # 实例化所有模块并移动到GPU
    parallel_conv_module = ParallelMultiScaleConvModule(in_channels=CHANNELS, group_dim=GROUP_DIM).to(device)
    spatial_fusion_module = InteractiveSpatialAttentionFusionModule(in_channels=CHANNELS).to(device)
    swin_channel_fusion_module = SwinChannelAttentionFusionModule(
        in_channels=CHANNELS,
        input_resolution=(HEIGHT, WIDTH),
        swin_embed_dim=256,
        num_heads=8,
        window_size=8,
        num_swin_blocks=2
    ).to(device)
    final_module = MSDAFusionModule(
        in_channels=CHANNELS,
        input_resolution=(HEIGHT, WIDTH),
        group_dim=GROUP_DIM,
        swin_embed_dim=256,
        num_heads=8,
        window_size=8,
        num_swin_blocks=2,
    ).to(device)

    print_gpu_memory("模型加载到GPU后")

    print("\n----------- 模块(1) 测试: ParallelMultiScaleConvModule (with compiled DCNv3) -----------")
    print(f"输入张量形状: {dummy_input.shape}")
    multi_scale_features = parallel_conv_module(dummy_input)
    print(f"输出为列表，包含 {len(multi_scale_features)} 个特征图")
    for i, feature in enumerate(multi_scale_features):
        print(f"  - 分支 {i + 1} 输出形状: {feature.shape}")

    print("\n----------- 模块(2) 测试: InteractiveSpatialAttentionFusionModule -----------")
    f_spatial = spatial_fusion_module(multi_scale_features)
    print(f"空间融合模块输出形状: {f_spatial.shape}")

    print("\n----------- 模块(3) 测试: SwinChannelAttentionFusionModule -----------")
    f_channel = swin_channel_fusion_module(multi_scale_features)
    print(f"通道融合模块输出形状: {f_channel.shape}")

    print("\n----------- 最终集成模块测试 (带显存分析) -----------")
    # 同步并重置峰值显存计数器，以精确测量前向传播的开销
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    print_gpu_memory("最终前向传播之前")

    # 前向传播
    final_output = final_module(dummy_input)

    # 同步以确保所有GPU操作完成
    torch.cuda.synchronize()
    print_gpu_memory("最终前向传播之后")
    print(f"最终集成模块输出形状: {final_output.shape}")

    # 检查所有输出形状是否符合预期
    assert multi_scale_features[0].shape == (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    assert f_spatial.shape == (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    assert f_channel.shape == (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    assert final_output.shape == (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    print("\n所有模块的输入输出形状均符合预期，测试通过！")

