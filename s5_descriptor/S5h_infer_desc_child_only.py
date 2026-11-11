#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from common import save_jsonl



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def token_to_text(token: str) -> str:
    return token.replace('_', ' ')


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
                return model
            except Exception:
                continue
        for cand in candidates:
            try:
                if 'vit' in cand.lower():
                    model = timm.create_model(cand, pretrained=False, num_classes=0)
                else:
                    model = timm.create_model(cand, pretrained=False, num_classes=0, global_pool='avg')
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


def build_caption(choice_map: Dict[str, str]) -> str:
    parts = []
    for group_name, choice in choice_map.items():
        if choice == 'none':
            continue
        parts.append(f'{group_name}: {token_to_text(choice)}')
    if not parts:
        return 'no clear descriptor'
    return '; '.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudolabel_jsonl', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_jsonl', type=str, required=True)
    parser.add_argument('--feature_bank_pt', type=str, default='')
    parser.add_argument('--path_to_idx_json', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location='cpu')

    target_mag = ckpt['target_mag']
    group_config = ckpt['group_config']
    model_args = ckpt['args']
    feature_dim = ckpt.get('feature_dim', None)
    use_features = ckpt.get('use_features', False)

    rows = load_jsonl(args.pseudolabel_jsonl)
    rows = [r for r in rows if r.get('mag') == target_mag and int(r.get('parse_ok', 1)) == 1]

    if use_features:
        if not args.feature_bank_pt or not args.path_to_idx_json:
            raise ValueError('This checkpoint expects --feature_bank_pt and --path_to_idx_json')
        feature_bank = torch.load(args.feature_bank_pt, map_location='cpu')
        if isinstance(feature_bank, dict):
            for k in ['features', 'feature_bank', 'embeddings', 'aligned_features', 'aligned_feature_bank']:
                if k in feature_bank and torch.is_tensor(feature_bank[k]):
                    feature_bank = feature_bank[k]
                    break
        if not torch.is_tensor(feature_bank):
            feature_bank = torch.tensor(feature_bank, dtype=torch.float32)
        path_to_idx = json.load(open(args.path_to_idx_json, 'r', encoding='utf-8'))
        rows = [r for r in rows if r['patch_path'] in path_to_idx]
    else:
        from PIL import Image
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((model_args.get('image_size', 224), model_args.get('image_size', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    model = ChildOnlyGroupedClassifier(
        group_config=group_config,
        hidden_dim=model_args.get('hidden_dim', 512),
        backbone_name=model_args.get('backbone_name', 'vit_base_patch16_224'),
        pretrained_backbone=False,
        feature_dim=feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    out_rows = []

    def batch_iter(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    with torch.no_grad():
        for batch_rows in tqdm(list(batch_iter(rows, args.batch_size)), desc='Infer child-only'):
            if use_features:
                x = torch.stack([feature_bank[path_to_idx[r['patch_path']]].float() for r in batch_rows], dim=0).to(device)
            else:
                imgs = []
                for r in batch_rows:
                    img = Image.open(r['patch_path']).convert('RGB')
                    imgs.append(tf(img))
                x = torch.stack(imgs, dim=0).to(device)

            logits = model(x)

            pred_by_group_batch = {}
            score_by_group_batch = {}

            for group_name, out in logits.items():
                probs = F.softmax(out, dim=-1).cpu()
                pred_idx = torch.argmax(probs, dim=-1).tolist()
                pred_choices = [group_config[group_name][i] for i in pred_idx]
                pred_by_group_batch[group_name] = pred_choices

                score_maps = []
                for i in range(probs.shape[0]):
                    score_maps.append({
                        choice: float(probs[i, j].item())
                        for j, choice in enumerate(group_config[group_name])
                    })
                score_by_group_batch[group_name] = score_maps

            for i, r in enumerate(batch_rows):
                choice_map = {g: pred_by_group_batch[g][i] for g in group_config}
                score_map = {g: score_by_group_batch[g][i] for g in group_config}

                out = dict(r)
                out['pred_choice_by_group'] = choice_map
                out['pred_scores_by_group'] = score_map
                out['pred_caption'] = build_caption(choice_map)
                out_rows.append(out)

    save_jsonl(out_rows, args.out_jsonl)
    print(f'[DONE] saved -> {args.out_jsonl}')


if __name__ == '__main__':
    main()