#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PatchEncoder(nn.Module):
    def __init__(self, backbone_name: str = 'vit_base_patch16_224.augreg_in21k_ft_in1k', out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = F.normalize(feat, dim=-1)
        return feat


class HieraPatchDesc(nn.Module):
    def __init__(self, tags_by_mag: Dict[str, List[str]], hidden_dim: int = 512,
                 backbone_name: str = 'vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained_backbone: bool = True):
        super().__init__()
        self.tags_by_mag = tags_by_mag
        self.mags = sorted(tags_by_mag.keys(), key=lambda x: int(x.replace('x', '')))
        self.mag_to_id = {m: i for i, m in enumerate(self.mags)}

        self.encoder = PatchEncoder(backbone_name=backbone_name, out_dim=hidden_dim, pretrained=pretrained_backbone)
        self.mag_embed = nn.Embedding(len(self.mags), hidden_dim)
        self.parent_to_children = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.child_to_parent = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.heads = nn.ModuleDict()
        for m, tags in tags_by_mag.items():
            self.heads[m] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(tags))
            )

    def add_mag_embedding(self, feat: torch.Tensor, mag: str) -> torch.Tensor:
        idx = torch.tensor([self.mag_to_id[mag]], device=feat.device)
        emb = self.mag_embed(idx)
        return feat + emb

    def forward_group(self, parent_img: torch.Tensor, child_imgs: torch.Tensor, parent_mag: str, child_mag: str):
        # parent_img: [B,3,H,W], child_imgs: [B,K,3,H,W]
        B, K = child_imgs.shape[:2]
        parent_feat = self.encoder(parent_img)  # [B,D]
        child_feats = self.encoder(child_imgs.view(B * K, *child_imgs.shape[2:])).view(B, K, -1)  # [B,K,D]

        parent_feat = self.add_mag_embedding(parent_feat, parent_mag)  # [B,D]
        child_feats = child_feats + self.mag_embed(torch.full((B, K), self.mag_to_id[child_mag], device=child_feats.device))

        q_parent = parent_feat.unsqueeze(1)      # [B,1,D]
        kv_children = child_feats                # [B,K,D]
        parent_ctx, _ = self.parent_to_children(q_parent, kv_children, kv_children)
        parent_ctx = self.norm(q_parent + parent_ctx).squeeze(1)  # [B,D]

        q_children = child_feats                 # [B,K,D]
        kv_parent = parent_feat.unsqueeze(1)     # [B,1,D]
        child_ctx, _ = self.child_to_parent(q_children, kv_parent, kv_parent)
        child_ctx = self.norm(q_children + child_ctx)             # [B,K,D]

        parent_logits = self.heads[parent_mag](parent_ctx)
        child_logits = self.heads[child_mag](child_ctx)

        align_parent = F.normalize(parent_ctx, dim=-1)
        align_children = F.normalize(child_ctx.mean(dim=1), dim=-1)
        return parent_logits, child_logits, align_parent, align_children


def bce_tag_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def cosine_alignment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()


def child_diversity_loss(child_logits: torch.Tensor) -> torch.Tensor:
    # child_logits: [B,K,C]
    probs = torch.sigmoid(child_logits)
    B, K, C = probs.shape
    if K <= 1:
        return torch.tensor(0.0, device=probs.device)
    sims = []
    for i in range(K):
        for j in range(i + 1, K):
            sims.append(F.cosine_similarity(probs[:, i], probs[:, j], dim=-1).mean())
    if not sims:
        return torch.tensor(0.0, device=probs.device)
    # 惩罚所有子patch都一样，越相似损失越大
    return torch.stack(sims).mean()
