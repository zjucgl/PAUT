import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from collections import OrderedDict
import timm
import math

import torchvision.models


def load_dict(model):
    """加载预训练权重"""
    weight_path = "weights/efficientnet_b0_ra.pth"
    ckpt = torch.load(weight_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[new_k] = v

    load_res = model.load_state_dict(new_state_dict, strict=False)
    print("missing keys:", load_res.missing_keys)
    print("unexpected keys:", load_res.unexpected_keys)
    return model


class PositionalGeometryEncoding(nn.Module):
    """
    几何位置编码（受Transformer启发）
    核心思路：不让模型学习几何特征，而是直接编码为位置信息（不可学习）
    """

    def __init__(self, geo_channels, d_model=64):
        super().__init__()
        self.d_model = d_model

        # 将6个几何通道映射到d_model维（固定权重，不学习）
        self.geo_projection = nn.Conv2d(geo_channels, d_model, 1, bias=False)

        # 冻结此层，不参与训练
        for param in self.geo_projection.parameters():
            param.requires_grad = False

        # 用正弦/余弦初始化（类似Transformer位置编码）
        self._init_positional_weights()

    def _init_positional_weights(self):
        """用固定的正弦余弦函数初始化"""
        with torch.no_grad():
            weight = self.geo_projection.weight  # [d_model, geo_channels, 1, 1]
            # 防止 index overflow
            C = weight.shape[1]
            for i in range(self.d_model):
                for j in range(C):
                    if i % 2 == 0:
                        weight[i, j, 0, 0] = math.sin(i / (10000 ** (j / max(1, C - 1))))
                    else:
                        weight[i, j, 0, 0] = math.cos(i / (10000 ** (j / max(1, C - 1))))

    def forward(self, geo_features):
        """
        geo_features: [B, 6, H, W]
        return: [B, d_model, H, W] 位置编码
        """
        pos_encoding = self.geo_projection(geo_features)
        return pos_encoding


class GeometryGuidedEfficientNet(nn.Module):
    """
    改进：Late fusion（mid-level spatial + channel gating）
    其它结构（geo编码不可学习、EfficientNet backbone、classifier）保持不变
    """

    def __init__(self,
                 num_classes=3,
                 img_channels=3,
                 selected_geo_indices: Optional[List[int]] = None,
                 pretrained=True,
                 # 可以调整 drop 参数以匹配你之前的实验
                 drop_rate=0.3,
                 drop_path_rate=0.2):
        super().__init__()

        self.selected_geo_indices = selected_geo_indices or list(range(6))
        active_geo_channels = len(self.selected_geo_indices)

        # ===== 几何位置编码（不可学习） =====
        self.geo_encoder = PositionalGeometryEncoding(
            geo_channels=active_geo_channels,
            d_model=16  # 编码为16维（保持原样）
        )

        # ===== light modules for late fusion (mid-level) =====
        # spatial_conv: 将 geo_pos -> 单通道 spatial mask (然后下采样到 feat_map 大小)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True)  # 输出单通道 raw mask
        )

        # channel MLP: 将 geo_pos 全局池化 -> 得到 channel gate
        # 使用较小的隐藏维度，避免显存/参数激增
        self.channel_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1280),
            nn.Sigmoid()  # gate in (0,1)
        )

        # learned scalar controlling fusion strength (init near 0 => weak influence)
        self.alpha_param = nn.Parameter(torch.tensor(0.0))

        # ===== EfficientNet-B0 主干（预训练） =====
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0,
            global_pool='avg',  # 使用全局平均池化
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # 手动加载预训练权重
        if pretrained:
            print("load EfficientNet pre weight...")
            self.backbone = load_dict(self.backbone)

        # EfficientNet-B0输出维度
        self.feature_dim = 1280

        # ===== 分类头=====
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_new_layers()

    def _initialize_new_layers(self):
        """初始化新增层（保持与之前相同的风格）"""
        for m in [self.spatial_conv, self.channel_mlp, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        nn.init.constant_(layer.weight, 1)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, img, geo_features):
        """
        img: [B, 3, H, W]
        geo_features: [B, 6, H, W]
        """
        # 选择几何通道
        geo = geo_features[:, self.selected_geo_indices, :, :]  # [B, G, H, W]

        # ===== 几何位置编码（不可学习） =====
        geo_pos = self.geo_encoder(geo)  # [B, 16, H, W]

        # ===== Backbone: 获取中间特征图（feature map）和最终池化特征 =====
        # timm EfficientNet 提供 forward_features 返回 feature map (B, C=1280, Hf=7, Wf=7)
        feat_map = self.backbone.forward_features(img)  # [B, C, Hf, Wf], C typically 1280
        # record spatial size
        _, C, Hf, Wf = feat_map.shape  # (1280, 7, 7)

        # ===== Spatial mask from geo_pos (downsample to feat_map size) =====
        # 1) downsample geo_pos to feat_map spatial resolution
        geo_ds = F.adaptive_avg_pool2d(geo_pos, (Hf, Wf))  # [B, 16, Hf, Wf]

        # 2) spatial_conv -> single channel raw map
        raw_mask = self.spatial_conv(geo_ds)  # [B,1,Hf,Wf]

        # 3) normalize mask to (0,1) via sigmoid. This is spatial attention.
        spatial_mask = torch.sigmoid(raw_mask)  # [B,1,Hf,Wf]

        # ===== Channel gate from geo_pos (global pooled) =====
        geo_vec = F.adaptive_avg_pool2d(geo_pos, (1, 1)).flatten(1)  # [B,16]
        channel_gate = self.channel_mlp(geo_vec)  # [B, C] in (0,1)

        # ===== Fusion: apply spatial mask and channel gate to feat_map =====
        # compute alpha (fusion strength)
        alpha = torch.sigmoid(self.alpha_param)  # scalar in (0,1)

        # spatial modulation: map spatial_mask from (0,1) -> multiplier approx (0.5,1.5)
        feat_map = feat_map * (1.0 + alpha * (spatial_mask - 0.5))

        # channel modulation: gate in (0,1) -> map to (0.5,1.5) as well
        gate = channel_gate.view(channel_gate.size(0), C, 1, 1)  # [B, C, 1, 1]
        feat_map = feat_map * (1.0 + alpha * (gate - 0.5))

        # ===== Global pooling & classification =====
        # use backbone's global pool logic: apply adaptive avg pool then classifier
        pooled = feat_map.mean(dim=[2, 3])  # [B, C] (equivalent to global avg pool)
        out = self.classifier(pooled)  # [B, num_classes]

        return out


def train_with_geometry(selected_geo_indices=None):
    if selected_geo_indices is None:
        selected_geo_indices = [0, 1, 2, 3, 4, 5]

    model = GeometryGuidedEfficientNet(
        num_classes=3,
        img_channels=3,
        selected_geo_indices=selected_geo_indices,
        pretrained=True,
        drop_rate=0.2,
        drop_path_rate=0.1
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    new_params = total_params - backbone_params

    print(f"\n{'=' * 60}")
    print(f"new EfficientNet fuse")
    print(f"{'=' * 60}")
    print(f"all canshu:       {total_params / 1e6:.2f}M")
    print(f"  Backbone:   {backbone_params / 1e6:.2f}M")
    print(f"  new model:    {new_params / 1e3:.2f}K")
    print(f"{'=' * 60}\n")

    return model, criterion, optimizer, scheduler


if __name__ == '__main__':
    # 快速 smoke test
    model, criterion, opt, sch = train_with_geometry()
    x = torch.randn(2, 3, 224, 224)
    geo = torch.randn(2, 6, 224, 224)
    logits = model(x, geo)
    print("logits", logits.shape)
