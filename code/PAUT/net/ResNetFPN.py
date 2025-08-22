import torch
import torch.nn as nn
import torchvision.models as models


# 定义FPN模块
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # 为每个输入特征图定义一个1x1卷积层，用于调整通道数
        self.inner_blocks = nn.ModuleList()
        # 为每个输出特征图定义一个3x3卷积层，用于平滑特征
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, x):
        # 从后往前处理特征图
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            # 上采样
            last_inner = nn.functional.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = last_inner + inner_lateral
            results.insert(0, self.layer_blocks[i](last_inner))

        return results


# 定义特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, out_channels):
        super(FeatureFusion, self).__init__()
        self.out_channels = out_channels

    def forward(self, features):
        # 对所有特征图进行上采样到相同大小
        target_size = features[0].shape[-2:]
        upsampled_features = []
        for feat in features:
            upsampled_feat = nn.functional.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            upsampled_features.append(upsampled_feat)
        # 在通道维度上进行拼接
        fused_features = torch.cat(upsampled_features, dim=1)
        return fused_features


# 定义分类器模块
class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义包含FPN和分类器的ResNet模型
class ResNetWithFPNClassifier(nn.Module):
    def __init__(self, resnet_model, out_channels=256, num_classes=4):
        super(ResNetWithFPNClassifier, self).__init__()
        self.resnet = resnet_model
        # 获取ResNet不同阶段的特征图通道数
        self.in_channels_list = [
            self.resnet.layer1[-1].conv2.out_channels,
            self.resnet.layer2[-1].conv2.out_channels,
            self.resnet.layer3[-1].conv2.out_channels,
            self.resnet.layer4[-1].conv2.out_channels
        ]
        self.fpn = FPN(self.in_channels_list, out_channels)
        self.feature_fusion = FeatureFusion(out_channels)
        self.classifier = Classifier(out_channels * len(self.in_channels_list), num_classes)

    def forward(self, x):
        # 通过ResNet的各个阶段
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        c1 = self.resnet.layer1(x)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        c4 = self.resnet.layer4(c3)
        # 将ResNet不同阶段的特征图输入到FPN中
        features = [c1, c2, c3, c4]
        p_features = self.fpn(features)
        # 特征融合
        fused_features = self.feature_fusion(p_features)
        # 分类
        output = self.classifier(fused_features)
        return output


def resnet18_fpn():
    resnet = models.resnet18()
    resnet.load_state_dict(torch.load('./weights/resnet18_v1.pth'), strict=False)
    model = ResNetWithFPNClassifier(resnet, num_classes=4)
    return model


# 示例使用
if __name__ == "__main__":
    # 加载预训练的ResNet18模型
    resnet = models.resnet18()
    model = ResNetWithFPNClassifier(resnet, num_classes=4)
    # 随机生成一个输入图像
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    for p in output:
        print(p.shape)

    model = models.resnet18()
    # 随机生成一个输入图像
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    for p in output:
        print(p.shape)
