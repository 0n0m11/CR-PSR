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
from tqdm import tqdm
from common import split_indices, subset_dataset



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


class FusionGroupedDataset(Dataset):
    def __init__(
        self,
        groups_jsonl: str,
        pseudolabel_jsonl: str,
        group_config_json: str,
        image_size: int = 224,
        feature_bank_pt: Optional[str] = None,
        path_to_idx_json: Optional[str] = None,
    ):
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

    def load_img(self, path: str):
        from PIL import Image
        return self.tf(Image.open(path).convert('RGB'))

    def load_feat(self, path: str):
        return self.feature_bank[self.path_to_idx[path]].float()

    def choice_to_idx(self, mag: str, group_name: str, choice: str) -> int:
        choices = self.group_config[mag][group_name]
        if choice not in choices:
            choice = 'none' if 'none' in choices else choices[-1]
        return choices.index(choice)

    def __getitem__(self, idx: int):
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

        parent_targets = {
            gname: torch.tensor(self.choice_to_idx(parent_mag, gname, parent_choice_map[gname]), dtype=torch.long)
            for gname in self.group_config[parent_mag]
        }
        child_targets = {
            gname: torch.tensor([self.choice_to_idx(child_mag, gname, cm[gname]) for cm in child_choice_maps], dtype=torch.long)
            for gname in self.group_config[child_mag]
        }

        return {
            'group_id': g.get('group_id', f'group_{idx}'),
            'parent_mag': parent_mag,
            'child_mag': child_mag,
            'parent_x': parent_x,
            'child_x': child_x,
            'parent_targets': parent_targets,
            'child_targets': child_targets,
        }


def collate_fn(batch):
    out = {
        'group_id': [x['group_id'] for x in batch],
        'parent_mag': batch[0]['parent_mag'],
        'child_mag': batch[0]['child_mag'],
        'parent_x': torch.stack([x['parent_x'] for x in batch], dim=0),
        'child_x': torch.stack([x['child_x'] for x in batch], dim=0),
        'parent_targets': {},
        'child_targets': {},
    }
    for g in batch[0]['parent_targets']:
        out['parent_targets'][g] = torch.stack([x['parent_targets'][g] for x in batch], dim=0)
    for g in batch[0]['child_targets']:
        out['child_targets'][g] = torch.stack([x['child_targets'][g] for x in batch], dim=0)
    return out


class FusionGroupedModel(nn.Module):
    def __init__(self, group_config, hidden_dim=512, backbone_name='vit_base_patch16_224', pretrained_backbone=True,
                 feature_dim=None, child_ckpt_path: str = '', freeze_child_proj: bool = False):
        super().__init__()
        self.group_config = group_config
        self.mags = sorted(group_config.keys(), key=lambda x: int(x.replace('x', '')))
        self.parent_mag = self.mags[0]
        self.child_mag = self.mags[1]
        self.use_features = feature_dim is not None

        if self.use_features:
            self.parent_proj = nn.Linear(feature_dim, hidden_dim)
            self.child_proj = nn.Linear(feature_dim, hidden_dim)
            self.encoder = None
        else:
            import timm
            self.encoder = timm.create_model(backbone_name, pretrained=pretrained_backbone, num_classes=0)
            feat_dim = self.encoder.num_features
            self.parent_proj = nn.Linear(feat_dim, hidden_dim)
            self.child_proj = nn.Linear(feat_dim, hidden_dim)

        # 从 child-only ckpt 初始化 child_proj
        if child_ckpt_path:
            ckpt = torch.load(child_ckpt_path, map_location='cpu')
            child_state = ckpt['model']
            if 'feat_proj.weight' in child_state:
                self.child_proj.weight.data.copy_(child_state['feat_proj.weight'])
                self.child_proj.bias.data.copy_(child_state['feat_proj.bias'])
                print('[INFO] child_proj initialized from child-only feat_proj')
            elif 'img_proj.weight' in child_state:
                self.child_proj.weight.data.copy_(child_state['img_proj.weight'])
                self.child_proj.bias.data.copy_(child_state['img_proj.bias'])
                print('[INFO] child_proj initialized from child-only img_proj')

        if freeze_child_proj:
            for p in self.child_proj.parameters():
                p.requires_grad = False

        self.parent_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.child_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.parent_heads = nn.ModuleDict()
        for gname, choices in group_config[self.parent_mag].items():
            self.parent_heads[gname] = nn.Linear(hidden_dim, len(choices))
        self.child_heads = nn.ModuleDict()
        for gname, choices in group_config[self.child_mag].items():
            self.child_heads[gname] = nn.Linear(hidden_dim, len(choices))

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
        child_hidden = self.child_fusion(torch.cat([
            child_local,
            parent_local.unsqueeze(1).expand(-1, K, -1),
            child_local - child_mean.expand(-1, K, -1)
        ], dim=-1))

        parent_logits = {g: head(parent_hidden) for g, head in self.parent_heads.items()}
        child_logits = {g: head(child_hidden) for g, head in self.child_heads.items()}
        return parent_logits, child_logits


def compute_group_class_weights(dataset: Dataset, group_config):
    counts = defaultdict(Counter)
    for i in range(len(dataset)):
        x = dataset[i]
        parent_mag = x['parent_mag']
        child_mag = x['child_mag']
        for g, t in x['parent_targets'].items():
            counts[(parent_mag, g)][int(t.item())] += 1
        for g, tvec in x['child_targets'].items():
            for t in tvec.tolist():
                counts[(child_mag, g)][int(t)] += 1

    weights = defaultdict(dict)
    for (mag, group_name), cnt in counts.items():
        ncls = len(group_config[mag][group_name])
        arr = np.array([cnt.get(i, 0) for i in range(ncls)], dtype=np.float32)
        arr = np.maximum(arr, 1.0)
        w = arr.sum() / (ncls * arr)
        w = w / w.mean()
        weights[mag][group_name] = torch.tensor(w, dtype=torch.float32)
    return weights


def group_ce_loss(logits_dict, targets_dict, mag, class_weights, device, is_child=False):
    total = 0.0
    count = 0
    for g, logits in logits_dict.items():
        w = class_weights[mag][g].to(device)
        if is_child:
            B, K, C = logits.shape
            loss = F.cross_entropy(logits.view(B*K, C), targets_dict[g].to(device).view(B*K), weight=w)
        else:
            loss = F.cross_entropy(logits, targets_dict[g].to(device), weight=w)
        total += loss
        count += 1
    return total / max(1, count)


def evaluate(model, loader, device, class_weights, parent_mag, child_mag):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            parent_x = batch['parent_x'].to(device)
            child_x = batch['child_x'].to(device)
            parent_logits, child_logits = model(parent_x, child_x)
            loss = group_ce_loss(parent_logits, batch['parent_targets'], parent_mag, class_weights, device, False)
            loss += group_ce_loss(child_logits, batch['child_targets'], child_mag, class_weights, device, True)
            total += loss.item()
            count += 1
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_jsonl', type=str, required=True)
    parser.add_argument('--pseudolabel_jsonl', type=str, required=True)
    parser.add_argument('--group_config_json', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--child_ckpt', type=str, required=True)
    parser.add_argument('--freeze_child_proj_epochs', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feature_bank_pt', type=str, default='')
    parser.add_argument('--path_to_idx_json', type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    group_config = json.load(open(args.group_config_json, 'r', encoding='utf-8'))
    parent_mag, child_mag = sorted(group_config.keys(), key=lambda x: int(x.replace('x', '')))

    dataset = FusionGroupedDataset(
        args.groups_jsonl,
        args.pseudolabel_jsonl,
        args.group_config_json,
        image_size=args.image_size,
        feature_bank_pt=args.feature_bank_pt if args.feature_bank_pt else None,
        path_to_idx_json=args.path_to_idx_json if args.path_to_idx_json else None,
    )

    train_idx, val_idx = split_indices(len(dataset), seed=args.seed)
    train_set = subset_dataset(dataset, train_idx)
    val_set = subset_dataset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    feature_dim = dataset.feature_dim if dataset.use_features else None
    model = FusionGroupedModel(
        group_config,
        hidden_dim=args.hidden_dim,
        backbone_name=args.backbone_name,
        pretrained_backbone=True,
        feature_dim=feature_dim,
        child_ckpt_path=args.child_ckpt,
        freeze_child_proj=True if args.freeze_child_proj_epochs > 0 else False,
    ).to(device)

    class_weights = compute_group_class_weights(train_set, group_config)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        # 到了指定轮数后解冻 child_proj
        if epoch == args.freeze_child_proj_epochs + 1:
            for p in model.child_proj.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            print('[INFO] child_proj unfrozen')

        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            parent_x = batch['parent_x'].to(device)
            child_x = batch['child_x'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                parent_logits, child_logits = model(parent_x, child_x)
                loss = group_ce_loss(parent_logits, batch['parent_targets'], parent_mag, class_weights, device, False)
                loss += group_ce_loss(child_logits, batch['child_targets'], child_mag, class_weights, device, True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        val_loss = evaluate(model, val_loader, device, class_weights, parent_mag, child_mag)
        print(f'[VAL] epoch={epoch} loss={val_loss:.6f}')

        ckpt = {
            'model': model.state_dict(),
            'group_config': group_config,
            'args': vars(args),
            'use_features': dataset.use_features,
            'feature_dim': feature_dim,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f'[SAVE] best.pt updated, val_loss={best_val:.6f}')

    print('[DONE] fusion training finished')


if __name__ == '__main__':
    main()
