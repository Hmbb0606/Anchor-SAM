import torch
import torch.nn as nn
# 外部依赖，在当前文件中缺失
from .ops_dscn.dscn import DSCNX, DSCNY

# === 新增模块: BFB Adapter 及其辅助类 (模仿 GBC.py) ===
# 这是我们实现贝叶斯融合瓶颈的核心代码

class BottConv(nn.Module):
    """
    GBC.py 中的瓶颈卷积块，作为 BFB 适配器的基础组件。
    结构: 1x1 降维 -> kxk 深度卷积 -> 1x1 升维
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

def get_norm_layer(norm_type, channels, num_groups=16):
    """
    GBC.py 中的归一化层获取函数。
    """
    if norm_type == 'GN':
        # 确保 num_groups 是 channels 的有效除数
        if channels > 0 and channels % num_groups != 0:
            # 寻找一个有效的 num_groups, 比如 8, 4, 2, 1
            valid_groups = [g for g in [16, 8, 4, 2, 1] if channels % g == 0]
            if valid_groups:
                num_groups = valid_groups[0]
            else: # 如果通道数很奇特，则无法分组
                return nn.Identity()
        if channels == 0:
            return nn.Identity()
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        # 默认或备选的归一化层
        return nn.BatchNorm2d(channels)

class BFB_Adapter(nn.Module):
    """
    贝叶斯融合瓶颈 (Bayesian Fusion Bottleneck, BFB) 适配器。
    该模块受 GBC 设计启发，用于智能融合“先验”(F_base)和“证据”(F_horizontal)特征。
    """
    def __init__(self, channels, norm_type='GN'):
        super(BFB_Adapter, self).__init__()
        # 瓶颈结构的中间通道数，通常为输入的 1/8
        bottleneck_channels = channels // 8

        # 先验处理分支: 使用 1x1 卷积快速提炼 F_base 的通道信息
        self.prior_path = nn.Sequential(
            BottConv(channels, channels, bottleneck_channels, kernel_size=1),
            get_norm_layer(norm_type, channels),
            nn.ReLU()
        )

        # 证据处理分支: 使用两个 3x3 卷积深度提取 F_horizontal 的空间上下文
        self.evidence_path = nn.Sequential(
            BottConv(channels, channels, bottleneck_channels, kernel_size=3, padding=1),
            get_norm_layer(norm_type, channels),
            nn.ReLU(),
            BottConv(channels, channels, bottleneck_channels, kernel_size=3, padding=1),
            get_norm_layer(norm_type, channels),
            nn.ReLU()
        )

        # 融合后处理分支: 对门控后的特征进行最终提炼
        self.post_fusion_path = nn.Sequential(
            BottConv(channels, channels, bottleneck_channels, kernel_size=1),
            get_norm_layer(norm_type, channels),
            nn.ReLU()
        )

    def forward(self, F_base, F_horizontal):
        # 并行处理“先验”和“证据”
        prior_feature = self.prior_path(F_base)
        evidence_feature = self.evidence_path(F_horizontal)

        # 特征门控: 用先验特征调制证据特征，实现贝叶斯更新
        gated_feature = prior_feature * evidence_feature

        # 融合后处理
        refined_feature = self.post_fusion_path(gated_feature)

        # 残差连接: 模块学习对“先验”信息流的增量修正
        posterior_guidance = refined_feature + F_base

        return posterior_guidance

class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)

        # 调用了外部文件中定义的 DSCNX 和 DSCNY
        self.dscn_x = DSCNX(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation,
                            group=group)  # , offset_scale=0.4)
        self.dscn_y = DSCNY(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation,
                            group=group)  # , offset_scale=0.4)
        self.conv = nn.Conv2d(d_model, d_model, 1)

        # *** 核心改动 1: 实例化 BFB_Adapter ***
        # 我们在这里创建了新的适配器模块，用于生成智能的指导信号
        self.bfb_adapter = BFB_Adapter(d_model, norm_type='GN')

    def forward(self, x):
        u = x.clone()
        x = self.conv0(x)

        # 准备被采样的特征 attn 和 指导特征 x_for_offset
        # 在原始DSA中，attn和x_for_offset均来自conv0的输出
        attn_permuted = x.permute(0, 2, 3, 1).contiguous()
        x_for_offset = x  # x_for_offset 此时就是 F_base (先验)

        # 水平采样, 得到 F_horizontal (permuted)
        attn_permuted = self.dscn_x(attn_permuted, x_for_offset)

        # --- BCDA 核心创新点 ---
        # 1. 将水平采样的结果转回 channels-first，作为“证据”
        F_horizontal = attn_permuted.permute(0, 3, 1, 2)
        # 2. 调用适配器，融合“先验”(x_for_offset)和“证据”(F_horizontal)
        posterior_guidance = self.bfb_adapter(x_for_offset, F_horizontal)
        # --- BCDA 核心创新点结束 ---

        # 条件垂直采样, 使用新的后验指导特征
        attn_permuted = self.dscn_y(attn_permuted, posterior_guidance)

        # --- 维持原始DSA的融合策略 ---
        # 1. 维度恢复
        attn = attn_permuted.permute(0, 3, 1, 2)
        # 2. 最终的 1x1 卷积只处理经过两次采样后的结果
        attn = self.conv(attn)

        # 应用空间注意力门控
        return u * attn


class DSA(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        # self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x) # 1*1卷积 输入与输出维度一致，不改变通道数
        x = self.activation(x) # 激活函数 GELU
        # x = self.dwconv(x)
        x = self.spatial_gating_unit(x) # 这是DSA模块最核心的部分，所有的可变形采样和注意力图生成都在这里完成。
        x = self.proj_2(x) # 经过核心单元处理后，再通过一个 1x1 卷积 (proj_2) 进行另一次特征变换，同样不改变通道数
        x = x + shorcut
        return x


# ==============================================================================
# === 新增模块: FPN特征的DSA注意力模块 (NEW: FPN DSA Attention Module) ===
# ==============================================================================
class FPN_DSA_Module(nn.Module):
    """
    对FPN输出的多层特征图分别应用DSA注意力模块。
    """

    def __init__(self, d_model, dsa_configs: list):
        """
        初始化FPN DSA模块。
        :param d_model: FPN特征图的通道数 (例如 256)。
        :param dsa_configs: 一个配置列表，每个元素都是一个字典，
                            用于配置对应FPN层级的DSA模块。
                            示例: [
                                {'kernel_size': 7, 'group': 4}, # for FPN Level 1
                                {'kernel_size': 7, 'group': 4}, # for FPN Level 2
                                ...
                            ]
        """
        super().__init__()
        self.dsa_layers = nn.ModuleList()
        for config in dsa_configs:
            kernel_size = config['kernel_size']
            group = config['group']
            # 根据kernel_size自动计算padding
            padding = (kernel_size - 1) // 2

            # 为每个FPN层级创建一个独立的DSA模块
            self.dsa_layers.append(
                DSA(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dw_kernel_size=kernel_size,  # dw_kernel_size通常与kernel_size相同
                    pad=padding,
                    stride=1,  # FPN特征处理通常步长为1
                    dilation=1,  # FPN特征处理通常膨胀系数为1
                    group=group
                )
            )

    def forward(self, fpn_outputs: list):
        """
        对输入的FPN特征图列表进行前向传播。
        :param fpn_outputs: 一个包含多个FPN层级特征图的列表。
        :return: 处理后、维度不变的特征图列表。
        """
        if len(fpn_outputs) != len(self.dsa_layers):
            raise ValueError(f"输入的FPN层数 ({len(fpn_outputs)}) 与配置的DSA模块数 ({len(self.dsa_layers)}) 不匹配。")

        # 对每一层FPN输出应用对应的DSA模块
        processed_outputs = []
        for i, feature_map in enumerate(fpn_outputs):
            processed_outputs.append(self.dsa_layers[i](feature_map))

        return processed_outputs


if __name__ == '__main__':

    # 1. 导入必要的库
    import torch

    # --- DSA 模块测试 (保持不变) ---
    # 2. 定义DSA模块所需的超参数
    d_model = 64
    kernel_size = 7
    dw_kernel_size = 7
    padding = 3
    stride = 1
    dilation = 1
    group = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        model = DSA(d_model=d_model, kernel_size=kernel_size, dw_kernel_size=dw_kernel_size, pad=padding, stride=stride,
                    dilation=dilation, group=group).to(device)
        model.eval()
        print("DSA model instantiated successfully.")
    except NameError as e:
        print(f"Error: Could not instantiate DSA. Make sure all dependencies like DSCNX are available.")
        print(f"Original error: {e}")
        exit()

    # 创建一个模拟输入张量
    dummy_input_dsa = torch.randn(4, d_model, 56, 56).to(device)
    print("\n--- DSA Module I/O Test ---")
    print(f"Input tensor shape:  {dummy_input_dsa.shape}")
    with torch.no_grad():
        output_dsa = model(dummy_input_dsa)
    print(f"Output tensor shape: {output_dsa.shape}")
    if dummy_input_dsa.shape == output_dsa.shape:
        print("\n[SUCCESS] Test passed for DSA: Input and output shapes match.")
    else:
        print("\n[FAILURE] Test failed for DSA: Input and output shapes do not match.")

    # --- 新增: FPN_DSA_Module 模块测试 ---
    print("\n" + "=" * 50)
    fpn_channels = 256
    # 灵活配置每一层的DSA参数
    fpn_dsa_configs = [
        {'kernel_size': 7, 'group': 4},  # FPN Level 1 (256x256)
        {'kernel_size': 7, 'group': 4},  # FPN Level 2 (128x128)
        {'kernel_size': 5, 'group': 8},  # FPN Level 3 (64x64)
        {'kernel_size': 5, 'group': 8},  # FPN Level 4 (32x32)
    ]
    fpn_model = FPN_DSA_Module(d_model=fpn_channels, dsa_configs=fpn_dsa_configs).to(device)
    fpn_model.eval()
    print("FPN_DSA_Module instantiated successfully.")

    # 创建一个模拟的FPN输出列表
    dummy_fpn_input = [
        torch.randn(1, fpn_channels, 256, 256).to(device),
        torch.randn(1, fpn_channels, 128, 128).to(device),
        torch.randn(1, fpn_channels, 64, 64).to(device),
        torch.randn(1, fpn_channels, 32, 32).to(device),
    ]
    print("\n--- FPN_DSA_Module I/O Test ---")
    print("Input tensor shapes:")
    for i, t in enumerate(dummy_fpn_input):
        print(f"  Level {i + 1}: {t.shape}")

    with torch.no_grad():
        fpn_output = fpn_model(dummy_fpn_input)

    print("\nOutput tensor shapes:")
    all_match = True
    for i, t_out in enumerate(fpn_output):
        print(f"  Level {i + 1}: {t_out.shape}")
        if t_out.shape != dummy_fpn_input[i].shape:
            all_match = False

    if all_match:
        print("\n[SUCCESS] Test passed for FPN_DSA_Module: All input and output shapes match.")
    else:
        print("\n[FAILURE] Test failed for FPN_DSA_Module: Shape mismatch detected.")
