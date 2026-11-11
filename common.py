# -*- coding: utf-8 -*-
"""CR-PSR 公共工具函数（S8 训练脚本共用）。
仅收纳各脚本中逻辑完全一致的函数；存在实质差异的函数（模型/数据集/评估/40×处理等）
仍保留在各脚本内，避免改动任何逻辑。
"""
import os
import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# =========================================================
# I/O
# =========================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 固定 descriptor vocab + one_hot
# =========================================================
DESC_VOCAB = {
    "5x_architecture_main": ["gland_rich", "solid_area", "none"],
    "5x_stroma_type": ["fibrous_stroma", "loose_stroma", "none"],
    "5x_distribution_pattern": ["irregular_distribution", "clustered_distribution", "none"],
    "5x_background_finding": ["diffuse_inflammation", "necrotic_focus", "none"],

    "10x_lumen_pattern": ["narrow_lumen", "dilated_lumen", "none"],
    "10x_gland_pattern": ["crowded_glands", "gland_irregularity", "none"],
    "10x_stromal_background": ["fibrotic_stroma", "inflammatory_cluster", "necrotic_debris", "none"],
}


def one_hot(value: str, choices: List[str]) -> List[float]:
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


# =========================================================
# 选项归一化 + 题目映射（S8c/S8d 组）
# =========================================================
def normalize_choices(choices):
    if isinstance(choices, list):
        return [str(x) for x in choices]
    if isinstance(choices, str):
        try:
            obj = json.loads(choices)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        if '|' in choices:
            return [x.strip() for x in choices.split('|')]
        if ',' in choices:
            return [x.strip() for x in choices.split(',')]
        return [choices]
    return []


def build_question_maps(rows):
    q_to_choices = {}
    for r in rows:
        q = str(r.get('question', ''))
        ans = str(r.get('answer', ''))
        choices = normalize_choices(r.get('choices', []))
        q_to_choices.setdefault(q, [])
        for c in choices:
            if c not in q_to_choices[q]:
                q_to_choices[q].append(c)
        if ans and ans not in q_to_choices[q]:
            q_to_choices[q].append(ans)
    q_to_idx = {q: i for i, q in enumerate(sorted(q_to_choices.keys()))}
    q_idx_to_choices = {q_to_idx[q]: q_to_choices[q] for q in q_to_choices}
    return q_to_choices, q_to_idx, q_idx_to_choices


def load_path_to_idx(path):
    with open(path, 'r', encoding='utf-8') as f:
        mp = json.load(f)
    return {str(k): int(v) for k, v in mp.items()}


# =========================================================
# 切分（按 base_sample_id / 患者）
# =========================================================
def split_by_base_sample_id(rows: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42):
    base_ids = sorted(list(set([x["base_sample_id"] for x in rows])))
    rng = random.Random(seed)
    rng.shuffle(base_ids)

    n_val = max(1, int(len(base_ids) * val_ratio))
    val_ids = set(base_ids[:n_val])
    train_ids = set(base_ids[n_val:])

    train_rows = [x for x in rows if x["base_sample_id"] in train_ids]
    val_rows = [x for x in rows if x["base_sample_id"] in val_ids]
    return train_rows, val_rows


def split_by_patient(rows: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42):
    def _patient_of(row):
        p = (row.get("image_5x") or "").split("/")
        pid = p[-3] if len(p) >= 3 else row["base_sample_id"]
        return re.sub(r'_[a-z]+$', '', pid)

    patient_ids = sorted(list(set([_patient_of(x) for x in rows])))
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_patients = set(patient_ids[:n_val])

    train_rows = [x for x in rows if _patient_of(x) not in val_patients]
    val_rows = [x for x in rows if _patient_of(x) in val_patients]
    return train_rows, val_rows


# =========================================================
# 类别权重 + 损失
# =========================================================
def compute_questionwise_class_weights(rows, question_to_choices):
    counts = defaultdict(Counter)
    for r in rows:
        q = r["question"]
        a = r["answer"]
        counts[q][a] += 1

    weights = {}
    for q, choices in question_to_choices.items():
        arr = np.array([counts[q].get(c, 0) for c in choices], dtype=np.float32)
        arr = np.maximum(arr, 1.0)
        w = arr.sum() / (len(choices) * arr)
        w = w / w.mean()
        weights[q] = torch.tensor(w, dtype=torch.float32)
    return weights


def focal_loss(logits, targets, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    if weight is not None:
        focal = focal * weight[targets]
    return focal.mean()


# =========================================================
# S5 共用：索引切分 + 数据集子集
# =========================================================
def split_indices(n: int, seed: int = 42, val_ratio: float = 0.1):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def subset_dataset(dataset: Dataset, indices: List[int]):
    class _Subset(Dataset):
        def __init__(self, ds, idxs):
            self.ds = ds
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.ds[self.idxs[i]]

    return _Subset(dataset, indices)
