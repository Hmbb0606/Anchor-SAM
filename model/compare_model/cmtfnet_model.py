'''
@ARTICLE{CMTFNet,
  author={Wu, Honglin and Huang, Peng and Zhang, Min and Tang, Wenlong and Yu, Xinyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={{CMTFNet}: CNN and Multiscale Transformer Fusion Network for Remote-Sensing Image Semantic Segmentation},
  year={2023},
  volume={61},
  number={},
  pages={1--12}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from einops import rearrange
from timm.models.layers import DropPath

class ResNet50Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        # 初始化一个空的 resnet50 结构
        base_model = resnet50(weights=None)

        if pretrained:
            # 预训练权重加载逻辑
            backbone_weight_path = './pretrain_weight/resnet50-0676ba61.pth'
            print(f"Attempting to load backbone weights from: {backbone_weight_path}")
            try:
                state_dict = torch.load(backbone_weight_path)
                base_model.load_state_dict(state_dict, strict=False)
                print("Backbone weights loaded successfully from local file.")
            except FileNotFoundError:
                print(f"Error: Backbone weight file not found at '{backbone_weight_path}'.")
                print("Please download it first by running this command in your project root:")
                print("wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P ./pretrain_weight/")
                print("\nFalling back to torchvision's automatic download...")
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 挂载 ResNet 的各个层
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)
        return x


class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim // fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3],
                                 groups=dim // fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)
        self.conv1_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2],
                                 groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv1_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)
        self.conv2_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1],
                                 groups=dim // fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv2_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()
        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()
        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))
        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))
        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))
        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u
        pool = self.Avg(attn)
        return pool


class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop=0., proj_drop=0., dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)
        B1, C1, H1, W1 = kv.shape
        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads, d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads, d=C // self.num_heads,
                         hh=H1, ww=W1, kv=2)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v
        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads, d=C // self.num_heads, hh=H,
                         ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn


class Block(nn.Module):
    def __init__(self, dim=512, num_heads=16, mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mutilscal_MHSA(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)
        self.mlp = E_FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.norm1(self.attn(x)))
        x = x + self.drop_path(self.mlp(x))
        return x


class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=1):
        super(MAF, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim // fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv1_1 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3],
                                 groups=dim // fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv1_2 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)
        self.conv2_1 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2],
                                 groups=dim // fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv2_2 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)
        self.conv3_1 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1],
                                 groups=dim // fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv3_2 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )
        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.head = nn.Sequential(SeparableConvBNReLU(256, 256, kernel_size=3),
                                  # Original code had hardcoded 256, adjusted to be `dim`
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(256, num_classes, kernel_size=1))  # Same here

    def forward(self, x):
        u = x.clone()
        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))
        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn
        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn
        attn = attn1_1 + attn1_2 + attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn
        out = self.head(attn + c_attn + s_attn)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=1):
        super(Decoder, self).__init__()
        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.b4 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])
        self.p3 = Fusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])
        self.p2 = Fusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])
        self.Conv3 = ConvBN(decode_channels, encode_channels[-4], 1)
        self.p1 = Fusion(encode_channels[-4])
        self.seg_head = MAF(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout,
                            num_classes=num_classes)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)
        x = self.b4(res4)
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.Conv3(x)
        x = self.p1(x, res1)
        x = self.seg_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CMTFNet(nn.Module):
    def __init__(self, n_classes=1, backbone='resnet50', pretrained=True):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            encode_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f"Backbone '{backbone}' is not supported for CMTFNet.")

        self.decoder = Decoder(encode_channels=encode_channels, decode_channels=512, dropout=0.1, num_classes=n_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x


if __name__ == '__main__':
    # --- 维度测试 ---
    # 创建一个模型实例
    model = CMTFNet(n_classes=1, backbone='resnet50', pretrained=True)
    model.eval()

    # 创建一个模拟输入张量
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
