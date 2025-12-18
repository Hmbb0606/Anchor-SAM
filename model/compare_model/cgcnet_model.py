'''
@ARTICLE{CGCNet,
  author={Liu, Peng and Gao, Xin and Shi, Chaojun and Lu, Yiguo and Bai, Lu and Fan, Yingying and Xing, Yalong and Qian, Yurong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={CGCNet: Road Extraction From Remote Sensing Image With Compact Global Context-Aware},
  year={2025},
  volume={63},
  number={},
  pages={1-12}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from functools import partial
import os


nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

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


class CompactGlobalContextawareBlock(nn.Module):
    """
    紧凑的全局上下文感知模块 (CGCB)。
    这是CGCNet的核心创新点，用于捕获长距离依赖。
    """

    def __init__(self, in_channels, size=(64, 64)):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channel = self.in_channels // 2
        self.conv_g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.in_channels, kernel_size=1,
                                   stride=1,
                                   padding=0, bias=False)

        self.pooling_size = 2
        self.token_len = self.pooling_size * self.pooling_size

        self.to_qk = nn.Linear(self.in_channels, 2 * self.inter_channel, bias=False)

        self.conv_a = nn.Conv2d(in_channels, self.token_len, kernel_size=1,
                                padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=False),
            nn.Sigmoid()
        )

        self.with_pos = True
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, 4, in_channels))

        self.with_pos_2 = True
        if self.with_pos_2:
            # 动态适应特征图尺寸，这是适配不同输入分辨率的关键
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.inter_channel,
                                                            size[0], size[1]))

    def compact_representation(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        # channel attention
        channel_attention = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(channel_attention).view(b, c, 1)
        x = x * channel_attention

        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def forward(self, x):
        b, c, h, w = x.size()
        x_clone = x
        x_compact = self.compact_representation(x)

        if self.with_pos:
            x_compact = x_compact + self.pos_embedding

        _, n, _ = x_compact.size()
        qk = self.to_qk(x_compact).chunk(2, dim=-1)
        q, k = qk[0].reshape(b, -1, n), qk[1]

        if self.with_pos_2:
            x_g = (self.conv_g(x_clone) + self.pos_embedding_2).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        else:
            x_g = self.conv_g(x_clone).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()

        mul_theta_phi = torch.matmul(q, k)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(x_g, mul_theta_phi)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)

        out = mask + x_clone
        return out


class CGCNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True, input_size=1024):
        """
        CGCNet 模型
        Args:
            n_channels (int): 输入通道数 (默认为3, 对应RGB图像)。
            n_classes (int): 输出类别数 (对于二分类任务，应为1)。
            pretrained (bool): 如果为True, 使用在ImageNet上预训练的ResNet-34骨干网络。
            input_size (int): 输入图像的尺寸，用于动态计算CGCB模块的位置编码尺寸。
        """
        super(CGCNet, self).__init__()


        base_model = resnet34(weights=None)
        if pretrained:
            backbone_weight_path = './pretrain_weight/resnet34-b627a593.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")
            if os.path.exists(backbone_weight_path):
                try:
                    state_dict = torch.load(backbone_weight_path)
                    base_model.load_state_dict(state_dict, strict=False)
                    print("Backbone weights loaded successfully from local file.")
                except Exception as e:
                    print(f"Error loading local backbone weights: {e}. Falling back to torchvision download.")
                    base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            else:
                print(f"Warning: Backbone weight file not found at '{backbone_weight_path}'.")
                print("Falling back to torchvision's automatic download...")
                base_model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # --- 编码器 (Encoder) ---
        # CGCNet 使用 ResNet-34 的前三个stage作为编码器
        self.first_conv = base_model.conv1
        self.first_bn = base_model.bn1
        self.first_relu = base_model.relu
        self.first_maxpool = base_model.maxpool
        self.encoder1 = base_model.layer1  # -> 64 channels, 256x256
        self.encoder2 = base_model.layer2  # -> 128 channels, 128x128
        self.encoder3 = base_model.layer3  # -> 256 channels, 64x64

        # --- 解码器 (Decoder) ---
        # ResNet layer3 输出 256 channels, layer2 输出 128, layer1 输出 64
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # --- 最终分类层 ---
        self.final_deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.final_relu1 = nonlinearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nonlinearity
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

        # --- CGCB 模块相关层 ---
        # 对encoder3的输出进行维度降低，应用CGCB，然后升维
        self.reduction_conv = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.increase_conv = nn.Conv2d(32, 256, kernel_size=3, padding=1)

        # 动态计算CGCB内部位置编码的尺寸
        # ResNet-34对输入的降采样倍数: conv1(2) * maxpool(2) * layer2(2) * layer3(2) = 16
        feature_map_size = input_size // 16
        cgcb_feature_size = (feature_map_size, feature_map_size)
        print(f"Initializing CGCB with feature size: {cgcb_feature_size} for input size {input_size}x{input_size}")
        self.cgcb = CompactGlobalContextawareBlock(in_channels=32, size=cgcb_feature_size)

    def compact_global_contextaware_block_reduction_increase(self, x):
        x_reduced = self.reduction_conv(x)
        x_cgcb = self.cgcb(x_reduced)
        x_increased = self.increase_conv(x_cgcb)
        return x_increased

    def forward(self, x):
        # --- 编码器 ---
        skip_connections = []

        x0 = self.first_conv(x)
        x0 = self.first_bn(x0)
        x0 = self.first_relu(x0)
        x0 = self.first_maxpool(x0)

        e1 = self.encoder1(x0)
        skip_connections.append(e1)
        e2 = self.encoder2(e1)
        skip_connections.append(e2)
        e3 = self.encoder3(e2)

        # --- 中心模块 ---
        e3_enhanced = e3 + self.compact_global_contextaware_block_reduction_increase(e3)

        # --- 解码器 ---
        d3 = self.decoder3(e3_enhanced) + skip_connections[1]  # skip: e2
        d2 = self.decoder2(d3) + skip_connections[0]  # skip: e1
        d1 = self.decoder1(d2)

        # --- 输出 ---
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out


if __name__ == '__main__':

    # 创建一个模型实例
    test_input_size = 1024
    model = CGCNet(n_channels=3, n_classes=1, pretrained=True, input_size=test_input_size)
    model.eval()

    # 创建一个模拟输入张量
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, test_input_size, test_input_size)

    # 模型前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的维度
    print(f"\n--- CGCNet Dimension Test ---")
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 验证输出维度是否正确
    expected_shape = (batch_size, 1, test_input_size, test_input_size)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, but got {output.shape}"
    print("\nDimension test passed! Output shape is as expected.")
