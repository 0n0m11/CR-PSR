#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


class FusionGroupedDataset(Dataset):
    def __init__(self, groups_jsonl, pseudolabel_jsonl, group_config_json, image_size=224, feature_bank_pt=None, path_to_idx_json=None):
        self.groups = load_jsonl(groups_jsonl)
        self.group_config = json.load(open(group_config_json, 'r', encoding='utf-8'))
        self.label_map = {}
        for row in load_jsonl(pseudolabel_jsonl):
            choice_map = row.get('stable_choice_by_group', row.get('choice_by_group', {}))
            self.label_map[row['patch_path']] = choice_map

        self.use_features = bool(feature_bank_pt) and bool(path_to_idx_json)
        self.feature_bank = None
        self.path_to_idx = None
        self.feature_dim = None
        if self.use_features:
            self.feature_bank = torch.load(feature_bank_pt, map_location='cpu')
            if isinstance(self.feature_bank, dict):
                for k in ['features', 'feature_bank', 'embeddings', 'aligned_features', 'aligned_feature_bank']:
                    if k in self.feature_bank and torch.is_tensor(self.feature_bank[k]):
                        self.feature_bank = self.feature_bank[k]
                        break
            if not torch.is_tensor(self.feature_bank):
                self.feature_bank = torch.tensor(self.feature_bank, dtype=torch.float32)
            self.path_to_idx = json.load(open(path_to_idx_json, 'r', encoding='utf-8'))
            self.feature_dim = int(self.feature_bank.shape[1])
        else:
            from torchvision import transforms
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        filtered = []
        for g in self.groups:
            parent_mag = g['from_mag']
            child_mag = g['to_mag']
            if parent_mag not in self.group_config or child_mag not in self.group_config:
                continue
            parent = g.get('parent_path', g.get('parent_image', ''))
            children = g.get('child_paths', g.get('child_images', []))
            if not parent or len(children) == 0:
                continue
            if parent not in self.label_map or any(ch not in self.label_map for ch in children):
                continue
            if self.use_features and (parent not in self.path_to_idx or any(ch not in self.path_to_idx for ch in children)):
                continue
            filtered.append(g)
        self.groups = filtered

    def __len__(self):
        return len(self.groups)

    def load_img(self, path):
        from PIL import Image
        return self.tf(Image.open(path).convert('RGB'))

    def load_feat(self, path):
        return self.feature_bank[self.path_to_idx[path]].float()

    def choice_to_idx(self, mag, group_name, choice):
        choices = self.group_config[mag][group_name]
        if choice not in choices:
            choice = 'none' if 'none' in choices else choices[-1]
        return choices.index(choice)

    def __getitem__(self, idx):
        g = self.groups[idx]
        parent_mag = g['from_mag']
        child_mag = g['to_mag']
        parent_path = g.get('parent_path', g.get('parent_image', ''))
        child_paths = g.get('child_paths', g.get('child_images', []))

        if self.use_features:
            parent_x = self.load_feat(parent_path)
            child_x = torch.stack([self.load_feat(p) for p in child_paths], dim=0)
        else:
            parent_x = self.load_img(parent_path)
            child_x = torch.stack([self.load_img(p) for p in child_paths], dim=0)

        parent_choice_map = self.label_map[parent_path]
        child_choice_maps = [self.label_map[p] for p in child_paths]
        parent_targets = {gname: torch.tensor(self.choice_to_idx(parent_mag, gname, parent_choice_map[gname]), dtype=torch.long)
                          for gname in self.group_config[parent_mag]}
        child_targets = {gname: torch.tensor([self.choice_to_idx(child_mag, gname, cm[gname]) for cm in child_choice_maps], dtype=torch.long)
                         for gname in self.group_config[child_mag]}
        return {'group_id': g.get('group_id', f'group_{idx}'), 'parent_mag': parent_mag, 'child_mag': child_mag,
                'parent_x': parent_x, 'child_x': child_x, 'parent_targets': parent_targets, 'child_targets': child_targets}


def collate_fn(batch):
    out = {'group_id': [x['group_id'] for x in batch], 'parent_mag': batch[0]['parent_mag'], 'child_mag': batch[0]['child_mag'],
           'parent_x': torch.stack([x['parent_x'] for x in batch], dim=0), 'child_x': torch.stack([x['child_x'] for x in batch], dim=0),
           'parent_targets': {}, 'child_targets': {}}
    for g in batch[0]['parent_targets']:
        out['parent_targets'][g] = torch.stack([x['parent_targets'][g] for x in batch], dim=0)
    for g in batch[0]['child_targets']:
        out['child_targets'][g] = torch.stack([x['child_targets'][g] for x in batch], dim=0)
    return out


class FusionGroupedModel(nn.Module):
    def __init__(self, group_config, hidden_dim=512, backbone_name='vit_base_patch16_224', pretrained_backbone=False,
                 feature_dim=None, child_ckpt_path='', freeze_child_proj=False):
        super().__init__()
        self.group_config = group_config
        self.mags = sorted(group_config.keys(), key=lambda x: int(x.replace('x', '')))
        self.parent_mag = self.mags[0]
        self.child_mag = self.mags[1]
        self.use_features = feature_dim is not None

        if self.use_features:
            self.encoder = None
            self.parent_proj = nn.Linear(feature_dim, hidden_dim)
            self.child_proj = nn.Linear(feature_dim, hidden_dim)
        else:
            import timm
            self.encoder = timm.create_model(backbone_name, pretrained=pretrained_backbone, num_classes=0)
            feat_dim = self.encoder.num_features
            self.parent_proj = nn.Linear(feat_dim, hidden_dim)
            self.child_proj = nn.Linear(feat_dim, hidden_dim)

        self.parent_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.child_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.parent_heads = nn.ModuleDict({g: nn.Linear(hidden_dim, len(c)) for g, c in group_config[self.parent_mag].items()})
        self.child_heads = nn.ModuleDict({g: nn.Linear(hidden_dim, len(c)) for g, c in group_config[self.child_mag].items()})

    def encode(self, parent_x, child_x):
        B, K = child_x.shape[:2]
        if self.use_features:
            parent_local = F.normalize(self.parent_proj(parent_x), dim=-1)
            child_local = F.normalize(self.child_proj(child_x.view(B * K, -1)), dim=-1).view(B, K, -1)
        else:
            parent_feat = self.encoder(parent_x)
            child_feat = self.encoder(child_x.view(B * K, *child_x.shape[2:])).view(B, K, -1)
            parent_local = F.normalize(self.parent_proj(parent_feat), dim=-1)
            child_local = F.normalize(self.child_proj(child_feat.view(B * K, -1)), dim=-1).view(B, K, -1)
        return parent_local, child_local

    def forward(self, parent_x, child_x):
        B, K = child_x.shape[:2]
        parent_local, child_local = self.encode(parent_x, child_x)
        child_mean = child_local.mean(dim=1, keepdim=True)
        parent_hidden = self.parent_fusion(torch.cat([parent_local, child_mean.squeeze(1)], dim=-1))
        child_hidden = self.child_fusion(torch.cat([child_local, parent_local.unsqueeze(1).expand(-1, K, -1), child_local - child_mean.expand(-1, K, -1)], dim=-1))
        parent_logits = {g: head(parent_hidden) for g, head in self.parent_heads.items()}
        child_logits = {g: head(child_hidden) for g, head in self.child_heads.items()}
        return parent_logits, child_logits


def update_confusion(counter_dict, mag, group_name, y_true, y_pred):
    for t, p in zip(y_true, y_pred):
        counter_dict[(mag, group_name)][(int(t), int(p))] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_jsonl', type=str, required=True)
    parser.add_argument('--pseudolabel_jsonl', type=str, required=True)
    parser.add_argument('--group_config_json', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feature_bank_pt', type=str, default='')
    parser.add_argument('--path_to_idx_json', type=str, default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location='cpu')
    group_config = json.load(open(args.group_config_json, 'r', encoding='utf-8'))
    model_args = ckpt['args']
    feature_dim = ckpt.get('feature_dim', None)

    dataset = FusionGroupedDataset(args.groups_jsonl, args.pseudolabel_jsonl, args.group_config_json,
                                  image_size=args.image_size,
                                  feature_bank_pt=args.feature_bank_pt if args.feature_bank_pt else None,
                                  path_to_idx_json=args.path_to_idx_json if args.path_to_idx_json else None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = FusionGroupedModel(group_config, hidden_dim=model_args['hidden_dim'], backbone_name=model_args['backbone_name'],
                               pretrained_backbone=False, feature_dim=dataset.feature_dim if dataset.use_features else None)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    parent_mag, child_mag = sorted(group_config.keys(), key=lambda x: int(x.replace('x', '')))
    per_group_correct = Counter(); per_group_total = Counter(); confusion = defaultdict(Counter)
    child_diverse = Counter(); child_total = Counter()

    with torch.no_grad():
        for batch in loader:
            parent_x = batch['parent_x'].to(device)
            child_x = batch['child_x'].to(device)
            parent_logits, child_logits = model(parent_x, child_x)

            for g, logits in parent_logits.items():
                pred = torch.argmax(logits, dim=-1).cpu()
                true = batch['parent_targets'][g]
                per_group_correct[(parent_mag, g, 'parent')] += int((pred == true).sum().item())
                per_group_total[(parent_mag, g, 'parent')] += int(true.numel())
                update_confusion(confusion, parent_mag, f'{g}_parent', true.tolist(), pred.tolist())

            for g, logits in child_logits.items():
                pred = torch.argmax(logits, dim=-1).cpu()
                true = batch['child_targets'][g]
                per_group_correct[(child_mag, g, 'child')] += int((pred == true).sum().item())
                per_group_total[(child_mag, g, 'child')] += int(true.numel())
                update_confusion(confusion, child_mag, f'{g}_child', true.view(-1).tolist(), pred.view(-1).tolist())
                for b in range(pred.shape[0]):
                    child_total[(child_mag, g)] += 1
                    if torch.unique(pred[b]).numel() > 1:
                        child_diverse[(child_mag, g)] += 1

    report = {'num_groups': len(dataset), 'per_group_accuracy': {}, 'child_diversity_rate': {}, 'confusion': {}}
    for key, total in per_group_total.items():
        mag, g, role = key
        report['per_group_accuracy'][f'{mag}/{role}/{g}'] = {'correct': per_group_correct[key], 'total': total, 'accuracy': per_group_correct[key] / max(1, total)}
    for key, total in child_total.items():
        mag, g = key
        report['child_diversity_rate'][f'{mag}/{g}'] = {'diverse_groups': child_diverse[key], 'total_groups': total, 'rate': child_diverse[key] / max(1, total)}
    for key, cnt in confusion.items():
        mag, g = key
        report['confusion'][f'{mag}/{g}'] = {f'{t}->{p}': c for (t, p), c in cnt.items()}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('[DONE] fusion evaluation saved to:', args.out_json)
    print('\n=== per_group_accuracy ===')
    for k, v in report['per_group_accuracy'].items():
        print(k, v)
    print('\n=== child_diversity_rate ===')
    for k, v in report['child_diversity_rate'].items():
        print(k, v)


if __name__ == '__main__':
    main()
