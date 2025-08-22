import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import ResNet
from ResNet import resnet18, resnet50
from mobilenetv3 import MobileNetV3_Small
from MobileNetV2 import MobileNetV2


class Fuse_PAUT(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            int_feat_dim=1
    ) -> None:
        super().__init__()

        self.img_feat_dim = 512  # ResNet50输出维度

        self.model_s = resnet18_att()
        self.model_s.load_state_dict(torch.load('./weights/resnet18_v1.pth'), strict=False)

        # 整数特征处理
        self.int_feat_fc = nn.Sequential(
            nn.Linear(int_feat_dim, 128),  # 扩展维度
            nn.BatchNorm1d(128),  # 新增批归一化
            nn.ReLU(),
            nn.Linear(128, self.img_feat_dim)  # 对齐图像特征维度
        )

        # 特征融合与分类
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim * 2, 256),  # 拼接后1024->256
            nn.ReLU(),
            nn.Dropout(0.3),  # 适当降低dropout
            nn.Linear(256, num_classes)
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x_s: Tensor, int_feat: Tensor) -> Tensor:

        img_feat = self.model_s(x_s)

        # 处理整数特征
        int_feat = int_feat.float().view(-1, 1)  # 显式转换为(batch_size, 1)
        int_feat = self.int_feat_fc(int_feat)  # (batch_size, 512)

        # 特征拼接融合
        fused_feat = torch.cat([img_feat , int_feat], dim=1)

        # 最终分类
        return self.fusion_fc(fused_feat)


if __name__ == '__main__':
    model = Fuse_PAUT()

    print(model.features)
    total_params = sum(p.numel() for p in model.parameters())
    print("================================================================")
    print(f"Total Parameters: {total_params}")
    print("================================================================")
