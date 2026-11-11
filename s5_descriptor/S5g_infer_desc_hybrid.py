#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


CHOICE_DESC = {
    'gland_rich': 'gland-rich architecture',
    'solid_area': 'solid area',
    'fibrous_stroma': 'fibrous stroma',
    'loose_stroma': 'loose stroma',
    'irregular_distribution': 'irregular distribution',
    'clustered_distribution': 'clustered distribution',
    'diffuse_inflammation': 'diffuse inflammation',
    'necrotic_focus': 'necrotic focus',
    'narrow_lumen': 'narrow lumen',
    'dilated_lumen': 'dilated lumen',
    'crowded_glands': 'crowded glands',
    'gland_irregularity': 'irregular gland contour',
    'fibrotic_stroma': 'fibrotic stroma',
    'inflammatory_cluster': 'inflammatory cluster',
    'necrotic_debris': 'necrotic debris',
    'none': 'no dominant feature',
}


def build_group_id(item: Dict[str, Any]) -> str:
    if item.get('group_id', ''):
        return item['group_id']
    meta = item.get('metadata', {}) or {}
    slide = item.get('slide', '') or item.get('slide_id', '')
    case_id = item.get('case_id', '')
    bag_index = meta.get('bag_index', item.get('bag_index', ''))
    if case_id and slide and bag_index != '':
        return f"{case_id}__{slide}__bag_{bag_index}"
    return os.path.splitext(os.path.basename(item.get('parent_path', 'group')))[0]


def verbalize_group_choice_dict(mag: str, pred_by_group: Dict[str, str]) -> str:
    ordered_groups_5x = ['architecture_main', 'stroma_type', 'distribution_pattern', 'background_finding']
    ordered_groups_10x = ['lumen_pattern', 'gland_pattern', 'stromal_background']
    group_order = ordered_groups_5x if mag == '5x' else ordered_groups_10x
    parts = []
    for g in group_order:
        choice = pred_by_group.get(g, 'none')
        if choice != 'none':
            parts.append(CHOICE_DESC.get(choice, choice.replace('_', ' ')))
    if not parts:
        return 'No dominant feature.'
    return '; '.join(parts).capitalize() + '.'


def predict_group_choice(logits: torch.Tensor, choices: List[str]):
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    score_dict = {c: float(p) for c, p in zip(choices, probs)}
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    return choices[pred_idx], score_dict


class ChildOnlyGroupedClassifier(nn.Module):
    def __init__(self, group_config, hidden_dim=512, feature_dim=768):
        super().__init__()
        self.group_config = group_config
        self.feat_proj = nn.Linear(feature_dim, hidden_dim)
        self.heads = nn.ModuleDict()
        for g, choices in group_config.items():
            self.heads[g] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(choices)),
            )

    def forward(self, x):
        feat = F.normalize(self.feat_proj(x), dim=-1)
        return {g: head(feat) for g, head in self.heads.items()}


class FusionGroupedModel(nn.Module):
    def __init__(self, group_config, hidden_dim=512, feature_dim=768):
        super().__init__()
        self.group_config = group_config
        mags = sorted(group_config.keys(), key=lambda x: int(x.replace('x', '')))
        self.parent_mag = mags[0]
        self.child_mag = mags[1]
        self.parent_proj = nn.Linear(feature_dim, hidden_dim)
        self.child_proj = nn.Linear(feature_dim, hidden_dim)
        self.parent_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.child_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.parent_heads = nn.ModuleDict({g: nn.Linear(hidden_dim, len(c)) for g, c in group_config[self.parent_mag].items()})
        self.child_heads = nn.ModuleDict({g: nn.Linear(hidden_dim, len(c)) for g, c in group_config[self.child_mag].items()})

    def forward(self, parent_x, child_x):
        B, K = child_x.shape[:2]
        parent_local = F.normalize(self.parent_proj(parent_x), dim=-1)
        child_local = F.normalize(self.child_proj(child_x.view(B * K, -1)), dim=-1).view(B, K, -1)
        child_mean = child_local.mean(dim=1, keepdim=True)
        parent_hidden = self.parent_fusion(torch.cat([parent_local, child_mean.squeeze(1)], dim=-1))
        child_hidden = self.child_fusion(torch.cat([child_local, parent_local.unsqueeze(1).expand(-1, K, -1), child_local - child_mean.expand(-1, K, -1)], dim=-1))
        parent_logits = {g: head(parent_hidden) for g, head in self.parent_heads.items()}
        child_logits = {g: head(child_hidden) for g, head in self.child_heads.items()}
        return parent_logits, child_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_jsonl', type=str, required=True)
    parser.add_argument('--fusion_ckpt', type=str, required=True)
    parser.add_argument('--child_ckpt', type=str, required=True)
    parser.add_argument('--feature_bank_pt', type=str, required=True)
    parser.add_argument('--path_to_idx_json', type=str, required=True)
    parser.add_argument('--out_jsonl', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fusion_ckpt = torch.load(args.fusion_ckpt, map_location='cpu')
    child_ckpt = torch.load(args.child_ckpt, map_location='cpu')

    # 加载 feature bank
    feature_bank = torch.load(args.feature_bank_pt, map_location='cpu')
    if isinstance(feature_bank, dict):
        for k in ['features', 'feature_bank', 'embeddings', 'aligned_features', 'aligned_feature_bank']:
            if k in feature_bank and torch.is_tensor(feature_bank[k]):
                feature_bank = feature_bank[k]
                break
    if not torch.is_tensor(feature_bank):
        feature_bank = torch.tensor(feature_bank, dtype=torch.float32)
    path_to_idx = json.load(open(args.path_to_idx_json, 'r', encoding='utf-8'))
    feature_dim = int(feature_bank.shape[1])

    fusion_group_config = fusion_ckpt['group_config']
    child_group_config = child_ckpt['group_config']

    fusion_model = FusionGroupedModel(fusion_group_config, hidden_dim=fusion_ckpt['args']['hidden_dim'], feature_dim=feature_dim)
    fusion_model.load_state_dict(fusion_ckpt['model'])
    fusion_model.to(device).eval()

    child_model = ChildOnlyGroupedClassifier(child_group_config, hidden_dim=child_ckpt['args']['hidden_dim'], feature_dim=feature_dim)
    child_model.load_state_dict(child_ckpt['model'])
    child_model.to(device).eval()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, 'w', encoding='utf-8') as fw:
        for row in load_jsonl(args.groups_jsonl):
            parent_mag = row['from_mag']
            child_mag = row['to_mag']
            if parent_mag not in fusion_group_config or child_mag not in fusion_group_config:
                continue
            parent_path = row.get('parent_path', row.get('parent_image', ''))
            child_paths = row.get('child_paths', row.get('child_images', []))
            parent_x = feature_bank[path_to_idx[parent_path]].float().unsqueeze(0).to(device)
            child_x = torch.stack([feature_bank[path_to_idx[p]].float() for p in child_paths], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                parent_logits, _ = fusion_model(parent_x, child_x)

            parent_pred_by_group = {}
            parent_scores_by_group = {}
            for g, logits in parent_logits.items():
                pred_choice, score_dict = predict_group_choice(logits[0], fusion_group_config[parent_mag][g])
                parent_pred_by_group[g] = pred_choice
                parent_scores_by_group[g] = score_dict
            parent_caption = verbalize_group_choice_dict(parent_mag, parent_pred_by_group)

            child_items = []
            with torch.no_grad():
                for i, p in enumerate(child_paths):
                    single_x = feature_bank[path_to_idx[p]].float().unsqueeze(0).to(device)
                    logits = child_model(single_x)
                    child_pred_by_group = {}
                    child_scores_by_group = {}
                    for g, out in logits.items():
                        pred_choice, score_dict = predict_group_choice(out[0], child_group_config[g])
                        child_pred_by_group[g] = pred_choice
                        child_scores_by_group[g] = score_dict
                    child_items.append({
                        'child_index': i,
                        'child_image': p,
                        'child_mag': child_mag,
                        'child_pred_by_group': child_pred_by_group,
                        'child_scores_by_group': child_scores_by_group,
                        'child_caption': verbalize_group_choice_dict(child_mag, child_pred_by_group),
                    })

            uniq_local = []
            seen = set()
            for x in child_items:
                cap = x['child_caption']
                if cap not in seen:
                    uniq_local.append(cap.rstrip('.'))
                    seen.add(cap)
            group_caption = f"Overall: {parent_caption.rstrip('.')}"
            if uniq_local:
                group_caption += '; local: ' + '; '.join(uniq_local) + '.'

            out = dict(row)
            out['group_id'] = build_group_id(row)
            out['parent_image'] = parent_path
            out['parent_mag'] = parent_mag
            out['parent_pred_by_group'] = parent_pred_by_group
            out['parent_scores_by_group'] = parent_scores_by_group
            out['parent_caption'] = parent_caption
            out['child_images'] = child_paths
            out['child_captions'] = child_items
            out['group_caption'] = group_caption
            fw.write(json.dumps(out, ensure_ascii=False) + '\n')

    print(f'[DONE] hybrid descriptors saved to: {args.out_jsonl}')


if __name__ == '__main__':
    main()
