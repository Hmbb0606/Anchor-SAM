'''
@inproceedings{D-LinkNet,
  title = {{D-LinkNet}: LinkNet With Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction},
  author = {Lichen Zhou and Chuang Zhang and Ming Wu},
  year = {2018},
  pages = {182--186},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}
}
'''

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    D-LinkNet的解码器模块。
    它包含一个卷积层，然后是一个转置卷积层来进行上采样。
    """

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

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


class Dblock(nn.Module):
    """
    D-LinkNet中心的空洞卷积模块。
    """

    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DLinkNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        """
        D-LinkNet 模型
        Args:
            n_channels (int): 输入通道数 (默认为3, 对应RGB图像)。
            n_classes (int): 输出类别数 (对于二分类任务，应为1)。
            pretrained (bool): 如果为True, 使用在ImageNet上预训练的ResNet-34骨干网络。
        """
        super(DLinkNet, self).__init__()

        # --- 编码器 (Encoder) ---
        base_model = resnet34(weights=None)

        if pretrained:
            # 预训练权重加载逻辑，与DeepLabv3+模型完全一致
            # 优先从本地相对路径加载，失败则回退到torchvision自动下载
            backbone_weight_path = './pretrain_weight/resnet34-b627a593.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")
            try:
                state_dict = torch.load(backbone_weight_path)
                base_model.load_state_dict(state_dict, strict=False)
                print("Backbone weights loaded successfully from local file.")
            except FileNotFoundError:
                print(f"Error: Backbone weight file not found at '{backbone_weight_path}'.")
                print("Please download it first by running this command in your project root:")
                print("wget https://download.pytorch.org/models/resnet34-b627a593.pth -P ./pretrain_weight/")
                print("\nFalling back to torchvision's automatic download...")
                base_model = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.in_block = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        # --- 中心模块 (Center) ---
        self.dblock = Dblock(512)

        # --- 解码器 (Decoder) ---
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # --- 最终分类层 ---
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        # --- 编码器 ---
        e0 = self.in_block(x)  # -> 64 channels
        e1 = self.encoder1(e0)  # -> 64 channels
        e2 = self.encoder2(e1)  # -> 128 channels
        e3 = self.encoder3(e2)  # -> 256 channels
        e4 = self.encoder4(e3)  # -> 512 channels

        # --- 中心模块 ---
        d_center = self.dblock(e4)

        # --- 解码器 ---
        # 解码器输入 = 中心模块输出 + 编码器跳跃连接
        d4 = self.decoder4(d_center) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # --- 输出 ---
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


if __name__ == '__main__':
    # --- 维度测试 ---
    # 创建一个模型实例 (pretrained=True 以测试权重加载逻辑)
    model = DLinkNet(n_channels=3, n_classes=1, pretrained=True)
    model.eval()

    # 创建一个模拟输入张量 (1024x1024)
    batch_size = 2
    input_height, input_width = 1024, 1024
    input_tensor = torch.randn(batch_size, 3, input_height, input_width)

    # 模型前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的维度
    print(f"\n--- Dimension Test ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 验证输出维度是否正确
    assert output.shape == (batch_size, 1, input_height, input_width)
    print("\nDimension test passed! Output shape is as expected.")
