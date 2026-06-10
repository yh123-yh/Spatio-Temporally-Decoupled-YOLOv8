from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class C2fLite(nn.Module):
    """Small C2f-style block used to keep the reproduction dependency-light."""

    def __init__(self, c: int, n: int = 2):
        super().__init__()
        hidden = max(c // 2, 16)
        self.reduce = ConvBNAct(c, hidden, 1)
        self.blocks = nn.ModuleList(ConvBNAct(hidden, hidden, 3) for _ in range(n))
        self.expand = ConvBNAct(hidden * (n + 1), c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        parts = [y]
        for block in self.blocks:
            y = block(y)
            parts.append(y)
        return self.expand(torch.cat(parts, dim=1))


class P2FidelityBranch(nn.Module):
    """Stride-4 branch for fine pod boundaries and small-object energy."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = ConvBNAct(channels, channels, 3)
        self.dw5 = nn.Sequential(
            ConvBNAct(channels, channels, 5, groups=channels),
            ConvBNAct(channels, channels, 1),
        )
        self.offset_like = ConvBNAct(channels, channels, 3)
        self.compress = ConvBNAct(channels * 3, channels, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.compress(torch.cat([self.conv3(x), self.dw5(x), self.offset_like(x)], dim=1))
        return y * self.channel_gate(y)


class Backbone(nn.Module):
    def __init__(self, width: float = 0.5, use_p2: bool = True):
        super().__init__()
        c1, c2, c3, c4, c5 = [max(int(v * width), 16) for v in (64, 128, 256, 512, 768)]
        self.channels = (c2, c3, c4, c5)
        self.stem = ConvBNAct(3, c1, 3, 2)
        p2_layers: list[nn.Module] = [ConvBNAct(c1, c2, 3, 2), C2fLite(c2, 1)]
        if use_p2:
            p2_layers.append(P2FidelityBranch(c2))
        self.p2 = nn.Sequential(*p2_layers)
        self.p3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), C2fLite(c3, 2))
        self.p4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), C2fLite(c4, 2))
        self.p5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), C2fLite(c5, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5


class SpatialStructureBranch(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], out_channels: int = 128):
        super().__init__()
        _, c3, c4, c5 = channels
        self.s1 = nn.Sequential(ConvBNAct(c3, out_channels, 3), C2fLite(out_channels, 1), ConvBNAct(out_channels, out_channels, 1))
        self.s2 = nn.Sequential(ConvBNAct(c4, out_channels, 3), C2fLite(out_channels, 1))
        self.s3 = nn.Sequential(ConvBNAct(c5, out_channels, 1), ConvBNAct(out_channels, out_channels, 3), ConvBNAct(out_channels, out_channels, 1))

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s1(p3), self.s2(p4), self.s3(p5)


class TemporalAssociationBranch(nn.Module):
    """Encodes 8/16-frame center drift, scale jitter, and visibility changes."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            ConvBNAct(3, 16, 3, 2),
            ConvBNAct(16, 32, 3, 2),
            ConvBNAct(32, 48, 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.diff_encoder = nn.Sequential(
            nn.Conv1d(48, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, out_dim, 3, padding=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim), nn.Sigmoid())

    def forward(self, sequence: torch.Tensor | None, fallback_image: torch.Tensor) -> torch.Tensor:
        if sequence is None:
            sequence = fallback_image[:, None]
        if sequence.ndim != 5:
            raise ValueError("sequence must be [B, T, C, H, W]")
        b, t, c, h, w = sequence.shape
        frames = sequence.reshape(b * t, c, h, w)
        emb = self.frame_encoder(frames).flatten(1).reshape(b, t, -1)
        if t > 1:
            diff = emb[:, 1:] - emb[:, :-1]
            diff = F.pad(diff, (0, 0, 1, 0))
        else:
            diff = torch.zeros_like(emb)
        feat = self.diff_encoder(diff.transpose(1, 2)).mean(dim=-1)
        return feat * self.gate(feat)


class GatedFusion(nn.Module):
    def __init__(self, feat_channels: int, spatial_channels: int = 128, temporal_dim: int = 64):
        super().__init__()
        self.spatial_proj = ConvBNAct(spatial_channels, feat_channels, 1)
        self.temporal_gate = nn.Sequential(nn.Linear(temporal_dim, feat_channels), nn.Sigmoid())
        self.mix = ConvBNAct(feat_channels * 2, feat_channels, 1)

    def forward(self, base: torch.Tensor, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_proj(spatial)
        gate = self.temporal_gate(temporal).unsqueeze(-1).unsqueeze(-1)
        return self.mix(torch.cat([base, spatial * gate], dim=1))


class DetectionHead(nn.Module):
    def __init__(self, channels: tuple[int, int, int], num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList(
            nn.Sequential(ConvBNAct(c, c, 3), nn.Conv2d(c, 4 + num_classes, 1)) for c in channels
        )

    def forward(self, feats: list[torch.Tensor], img_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        boxes_all: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        img_h, img_w = img_size
        for feat, layer in zip(feats, self.layers):
            raw = layer(feat)
            b, _, h, w = raw.shape
            raw = raw.permute(0, 2, 3, 1).reshape(b, h * w, -1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=feat.device),
                torch.arange(w, device=feat.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, h * w, 2).float()
            xy = (torch.sigmoid(raw[..., 0:2]) + grid) / raw.new_tensor([w, h])
            wh = torch.sigmoid(raw[..., 2:4]).pow(2) * 0.35
            boxes = torch.cat([xy, wh], dim=-1).clamp(0, 1)
            boxes_all.append(boxes)
            logits_all.append(raw[..., 4 : 4 + self.num_classes])
        return torch.cat(boxes_all, dim=1), torch.cat(logits_all, dim=1).squeeze(-1)


@dataclass
class ModelConfig:
    width: float = 0.5
    num_classes: int = 1
    use_p2: bool = True
    use_spatial: bool = True
    use_temporal: bool = True
    use_fusion: bool = True
    use_count_head: bool = True


class STPodYOLO(nn.Module):
    """YOLOv8-inspired spatio-temporally decoupled detector/counting model."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = Backbone(width=self.config.width, use_p2=self.config.use_p2)
        c2, c3, c4, c5 = self.backbone.channels
        self.spatial = SpatialStructureBranch(self.backbone.channels)
        self.temporal = TemporalAssociationBranch(64)
        self.fuse3 = GatedFusion(c3)
        self.fuse4 = GatedFusion(c4)
        self.fuse5 = GatedFusion(c5)
        self.head = DetectionHead((c3, c4, c5), self.config.num_classes)
        self.count_head = nn.Sequential(
            nn.Linear(c5 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, sequence: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, _, h, w = image.shape
        p2, p3, p4, p5 = self.backbone(image)
        temporal = self.temporal(sequence, image) if self.config.use_temporal else torch.zeros(image.shape[0], 64, device=image.device)
        if self.config.use_spatial and self.config.use_fusion:
            s3, s4, s5 = self.spatial(p3, p4, p5)
            p3 = self.fuse3(p3, s3, temporal)
            p4 = self.fuse4(p4, s4, temporal)
            p5 = self.fuse5(p5, s5, temporal)
        boxes, logits = self.head([p3, p4, p5], (h, w))
        pooled = F.adaptive_avg_pool2d(p5, 1).flatten(1)
        count = self.count_head(torch.cat([pooled, temporal], dim=1)).squeeze(-1)
        return {"boxes": boxes, "logits": logits, "count": F.softplus(count)}


def build_model_from_dict(data: dict) -> STPodYOLO:
    cfg = ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig.__annotations__})
    return STPodYOLO(cfg)
