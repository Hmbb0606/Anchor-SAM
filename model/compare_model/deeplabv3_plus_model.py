'''
@inproceedings{DeepLabv3+,
  title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={801--818},
  year={2018}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        if backbone == 'resnet101':
            inplanes = 2048
        else:
            inplanes = 512  # Fallback for other backbones, adjust if necessary

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            low_level_inplanes = 256
        else:
            low_level_inplanes = 64  # Fallback for other backbones

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, backbone='resnet101', output_stride=16, pretrained=True):
        """
        DeepLabV3+ Model for Semantic Segmentation.
        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB).
            n_classes (int): Number of output classes. For binary segmentation, this should be 1.
            backbone (str): Name of the backbone network. Currently only 'resnet101' is supported.
            output_stride (int): The ratio of input image spatial resolution to the final output resolution of the encoder.
            pretrained (bool): If True, use a backbone pre-trained on ImageNet.
        """
        super(DeepLabV3Plus, self).__init__()
        if backbone != 'resnet101':
            raise NotImplementedError("Only 'resnet101' backbone is implemented for now.")

        # --- Encoder (Backbone) ---
        # 更新为使用新的 weights API, weights=None 相当于之前的 pretrained=False
        self.backbone = resnet101(weights=None)

        if pretrained:
            # 如果 pretrained 为 True, 则从指定的相对路径加载权重
            # 这个路径是相对于项目根目录 (即 train.py 所在的位置)
            backbone_weight_path = './pretrain_weight/resnet101-5d3b4d8f.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")

            try:
                # 从文件加载状态字典
                state_dict = torch.load(backbone_weight_path)
                # 将权重加载到骨干网络中
                # 使用 strict=False 是因为我们不需要ImageNet预训练模型最后的'fc'分类层
                self.backbone.load_state_dict(state_dict, strict=False)
                print("Backbone weights loaded successfully from local file.")
            except FileNotFoundError:
                print(f"Error: Backbone weight file not found at '{backbone_weight_path}'.")
                print("Please download it first by running this command in your project root:")
                print("wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -P ./pretrain_weight/")
                print("\nFalling back to torchvision's automatic download using the new 'weights' API...")
                # 如果本地文件未找到, 则回退到 torchvision 的默认行为 (自动下载)
                # 更新为使用新的 weights API, ResNet101_Weights.DEFAULT 会获取最佳的可用权重
                self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

        # 修改骨干网络以支持空洞卷积
        if output_stride == 16:
            self.backbone.layer3.apply(self._apply_stride_and_dilation(stride=1, dilation=2))
            self.backbone.layer4.apply(self._apply_stride_and_dilation(stride=1, dilation=4))
        elif output_stride == 8:
            self.backbone.layer2.apply(self._apply_stride_and_dilation(stride=1, dilation=2))
            self.backbone.layer3.apply(self._apply_stride_and_dilation(stride=1, dilation=4))
            self.backbone.layer4.apply(self._apply_stride_and_dilation(stride=1, dilation=8))
        else:
            raise NotImplementedError

        # --- ASPP ---
        self.aspp = ASPP(backbone, output_stride)

        # --- Decoder ---
        self.decoder = Decoder(n_classes, backbone)

        # 初始化 ASPP 和 Decoder 的权重
        self._init_weight()

    def _apply_stride_and_dilation(self, stride, dilation):
        def _func(m):
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (stride, stride)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)

        return _func

    def forward(self, x):
        # 存储原始图像尺寸
        input_size = x.size()[2:]

        # --- Encoder Forward Pass ---
        # 注意: resnet 的层名为 conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        low_level_feat = self.backbone.layer1(x)  # 从 layer1 获取低阶特征
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # --- ASPP Forward Pass ---
        x = self.aspp(x)

        # --- Decoder Forward Pass ---
        x = self.decoder(x, low_level_feat)

        # --- Final Upsampling ---
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    # --- 维度测试 ---
    # 创建一个模型实例
    # n_classes=1 表示二分类任务
    # pretrained=False 避免在测试时下载权重
    model = DeepLabV3Plus(n_channels=3, n_classes=1, backbone='resnet101', pretrained=True)
    model.eval()

    # 创建一个模拟输入张量
    # 尺寸为 (batch_size, channels, height, width)
    # 更新为 1024x1024 来匹配您的默认训练尺寸
    batch_size = 2
    input_height, input_width = 1024, 1024
    input_tensor = torch.randn(batch_size, 3, input_height, input_width)

    # 模型前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的维度
    print(f"输入张量维度 (Input shape): {input_tensor.shape}")
    print(f"输出张量维度 (Output shape): {output.shape}")

    # 验证输出维度是否正确
    # 对于二分类，输出通道数应为1，且高宽应与输入一致
    assert output.shape == (batch_size, 1, input_height, input_width)
    print("\n维度测试通过！输出维度符合预期 (1024x1024)。")


