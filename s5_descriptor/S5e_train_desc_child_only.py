#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import split_indices, subset_dataset



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


class ChildPatchDataset(Dataset):
    def __init__(
        self,
        pseudolabel_jsonl: str,
        group_config_json: str,
        target_mag: str = '10x',
        image_size: int = 224,
        feature_bank_pt: Optional[str] = None,
        path_to_idx_json: Optional[str] = None,
    ):
        rows = load_jsonl(pseudolabel_jsonl)
        self.group_config = json.load(open(group_config_json, 'r', encoding='utf-8'))
        self.target_mag = target_mag
        self.groups = self.group_config[target_mag]

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
        for row in rows:
            if row.get('mag') != target_mag:
                continue
            if int(row.get('parse_ok', 1)) != 1:
                continue
            choice_map = row.get('stable_choice_by_group', row.get('choice_by_group', {}))
            ok = all(g in choice_map for g in self.groups)
            if not ok:
                continue
            if self.use_features and row['patch_path'] not in self.path_to_idx:
                continue
            row['_choice_map'] = choice_map
            filtered.append(row)
        self.rows = filtered

    def __len__(self):
        return len(self.rows)

    def load_img(self, path: str):
        from PIL import Image
        return self.tf(Image.open(path).convert('RGB'))

    def load_feat(self, path: str):
        idx = self.path_to_idx[path]
        return self.feature_bank[idx].float()

    def choice_to_idx(self, group_name: str, choice: str) -> int:
        choices = self.groups[group_name]
        if choice not in choices:
            choice = 'none' if 'none' in choices else choices[-1]
        return choices.index(choice)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = self.load_feat(row['patch_path']) if self.use_features else self.load_img(row['patch_path'])
        targets = {
            g: torch.tensor(self.choice_to_idx(g, row['_choice_map'][g]), dtype=torch.long)
            for g in self.groups
        }
        return {
            'patch_path': row['patch_path'],
            'mag': self.target_mag,
            'x': x,
            'targets': targets,
        }


def collate_fn(batch):
    out = {
        'patch_path': [x['patch_path'] for x in batch],
        'mag': batch[0]['mag'],
        'x': torch.stack([x['x'] for x in batch], dim=0),
        'targets': {},
    }
    for g in batch[0]['targets']:
        out['targets'][g] = torch.stack([x['targets'][g] for x in batch], dim=0)
    return out


class ChildOnlyGroupedClassifier(nn.Module):
    def __init__(self, group_config: Dict[str, List[str]], hidden_dim: int = 512,
                 backbone_name: str = 'vit_base_patch16_224', pretrained_backbone: bool = True,
                 feature_dim: Optional[int] = None):
        super().__init__()
        self.group_config = group_config
        self.use_features = feature_dim is not None

        if self.use_features:
            self.encoder = None
            self.feat_proj = nn.Linear(feature_dim, hidden_dim)
        else:
            self.encoder = self._build_backbone(backbone_name, pretrained_backbone)
            feat_dim = self.encoder.num_features
            self.img_proj = nn.Linear(feat_dim, hidden_dim)

        self.heads = nn.ModuleDict()
        for group_name, choices in group_config.items():
            self.heads[group_name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(choices)),
            )

    def _build_backbone(self, backbone_name: str, pretrained_backbone: bool):
        import timm
        candidates = [backbone_name, 'vit_base_patch16_224', 'resnet50']
        for cand in candidates:
            try:
                if 'vit' in cand.lower():
                    model = timm.create_model(cand, pretrained=pretrained_backbone, num_classes=0)
                else:
                    model = timm.create_model(cand, pretrained=pretrained_backbone, num_classes=0, global_pool='avg')
                print(f'[INFO] Using backbone: {cand} | pretrained={pretrained_backbone}')
                return model
            except Exception:
                continue
        for cand in candidates:
            try:
                if 'vit' in cand.lower():
                    model = timm.create_model(cand, pretrained=False, num_classes=0)
                else:
                    model = timm.create_model(cand, pretrained=False, num_classes=0, global_pool='avg')
                print(f'[INFO] Using backbone fallback: {cand} | pretrained=False')
                return model
            except Exception:
                continue
        raise RuntimeError(f'All backbone candidates failed: {candidates}')

    def forward(self, x):
        if self.use_features:
            feat = F.normalize(self.feat_proj(x), dim=-1)
        else:
            feat = F.normalize(self.img_proj(self.encoder(x)), dim=-1)
        logits = {g: head(feat) for g, head in self.heads.items()}
        return logits


def compute_class_weights(dataset: Dataset, group_config: Dict[str, List[str]]):
    counts = defaultdict(Counter)
    for i in range(len(dataset)):
        x = dataset[i]
        for g, t in x['targets'].items():
            counts[g][int(t.item())] += 1

    weights = {}
    for g, cnt in counts.items():
        ncls = len(group_config[g])
        arr = np.array([cnt.get(i, 0) for i in range(ncls)], dtype=np.float32)
        arr = np.maximum(arr, 1.0)
        w = arr.sum() / (ncls * arr)
        w = w / w.mean()
        weights[g] = torch.tensor(w, dtype=torch.float32)
    return weights


def grouped_ce_loss(logits_dict, targets_dict, class_weights, device):
    total = 0.0
    count = 0
    for g, logits in logits_dict.items():
        w = class_weights[g].to(device)
        loss = F.cross_entropy(logits, targets_dict[g].to(device), weight=w)
        total += loss
        count += 1
    return total / max(1, count)


def evaluate(model, loader, device, class_weights):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = Counter()
    total = Counter()
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            logits = model(x)
            loss = grouped_ce_loss(logits, batch['targets'], class_weights, device)
            total_loss += loss.item()
            total_batches += 1
            for g, out in logits.items():
                pred = torch.argmax(out, dim=-1).cpu()
                y = batch['targets'][g]
                correct[g] += int((pred == y).sum().item())
                total[g] += int(y.numel())
    acc = {g: correct[g] / max(1, total[g]) for g in total}
    return total_loss / max(1, total_batches), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudolabel_jsonl', type=str, required=True)
    parser.add_argument('--group_config_json', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--target_mag', type=str, default='10x')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
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
    all_group_config = json.load(open(args.group_config_json, 'r', encoding='utf-8'))
    group_config = all_group_config[args.target_mag]

    dataset = ChildPatchDataset(
        args.pseudolabel_jsonl,
        args.group_config_json,
        target_mag=args.target_mag,
        image_size=args.image_size,
        feature_bank_pt=args.feature_bank_pt if args.feature_bank_pt else None,
        path_to_idx_json=args.path_to_idx_json if args.path_to_idx_json else None,
    )

    train_idx, val_idx = split_indices(len(dataset), seed=args.seed)
    train_set = subset_dataset(dataset, train_idx)
    val_set = subset_dataset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    feature_dim = dataset.feature_dim if dataset.use_features else None
    model = ChildOnlyGroupedClassifier(
        group_config,
        hidden_dim=args.hidden_dim,
        backbone_name=args.backbone_name,
        pretrained_backbone=True,
        feature_dim=feature_dim,
    ).to(device)

    class_weights = compute_class_weights(train_set, group_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            x = batch['x'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = grouped_ce_loss(logits, batch['targets'], class_weights, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        val_loss, val_acc = evaluate(model, val_loader, device, class_weights)
        print(f'[VAL] epoch={epoch} loss={val_loss:.6f} acc={val_acc}')

        ckpt = {
            'model': model.state_dict(),
            'target_mag': args.target_mag,
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

    print('[DONE] child-only training finished')


if __name__ == '__main__':
    main()
