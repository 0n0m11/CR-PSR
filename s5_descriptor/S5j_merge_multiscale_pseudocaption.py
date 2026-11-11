#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional
from common import save_jsonl



# =========================================================
# 基础工具
# =========================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_non_none(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def norm_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return str(p).replace("\\", "/").strip()


def basename_noext(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.splitext(os.path.basename(p))[0]


# =========================================================
# 路径抽取
# =========================================================
def get_5x_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "path_5x", "img_5x", "image_5x", "parent_5x_path", "p5_path", "path5x"
    ]))


def get_10x_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "path_10x", "img_10x", "image_10x", "child_10x_path", "c10_path", "path10x"
    ]))


def get_20x_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "path_20x", "img_20x", "image_20x", "child_20x_path", "c20_path", "path20x", "patch_path"
    ]))


def get_40x_paths(row: Dict[str, Any]) -> List[str]:
    vals = first_non_none(row, [
        "paths_40x", "imgs_40x", "images_40x", "child_40x_paths", "c40_paths"
    ], default=[])
    vals = ensure_list(vals)
    return [norm_path(x) for x in vals if x is not None]


def get_parent_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "parent_path", "parent_img", "parent_image", "parent"
    ]))


def get_child_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "child_path", "child_img", "child_image", "child"
    ]))


def get_child_paths(row: Dict[str, Any]) -> List[str]:
    vals = first_non_none(row, [
        "child_paths", "children", "child_imgs", "child_images"
    ], default=[])
    vals = ensure_list(vals)
    return [norm_path(x) for x in vals if x is not None]


def get_any_image_path(row: Dict[str, Any]) -> Optional[str]:
    return norm_path(first_non_none(row, [
        "image_path", "img_path", "path", "image", "img", "patch_path",
        "path_20x", "path_40x", "path_10x", "path_5x",
        "child_path", "parent_path"
    ]))


# =========================================================
# 语义字段抽取
# =========================================================
def extract_descriptor_block(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k, v in row.items():
        lk = k.lower()
        if any(token in lk for token in [
            "desc", "label", "pseudo", "group", "stable", "tag", "semantic",
            "architecture", "pattern", "density", "background", "gland",
            "crowding", "lumen", "irregularity", "detail", "choice_by_group", "raw_by_group"
        ]):
            keep[k] = v
    return keep


def extract_soft20_block(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k, v in row.items():
        lk = k.lower()
        if any(token in lk for token in [
            "soft", "score", "prob", "margin", "uncert", "density",
            "crowding", "overlap", "none"
        ]):
            keep[k] = v
    return keep


# =========================================================
# 伪 caption 生成
# =========================================================
def prettify_token(x: str) -> str:
    return str(x).replace("_", " ").replace("\\_", " ").strip()


def build_pseudocaption_from_groups(row: Dict[str, Any]) -> str:
    """
    根据 choice_by_group / raw_by_group 自动生成一条简洁伪caption
    """
    mag = row.get("mag", "")

    choice_by_group = row.get("choice_by_group", {})
    raw_by_group = row.get("raw_by_group", {})

    if not isinstance(choice_by_group, dict):
        choice_by_group = {}
    if not isinstance(raw_by_group, dict):
        raw_by_group = {}

    group_items = []
    source = choice_by_group if len(choice_by_group) > 0 else raw_by_group

    for gname, gval in source.items():
        if gval is None:
            continue
        gname_text = prettify_token(gname)
        gval_text = prettify_token(gval)
        group_items.append((gname_text, gval_text))

    if len(group_items) == 0:
        if mag:
            return f"{mag} patch with no stable semantic description."
        return "Patch with no stable semantic description."

    # 生成自然一点的短句
    parts = []
    for gname_text, gval_text in group_items:
        parts.append(f"{gname_text}: {gval_text}")

    body = "; ".join(parts)

    if mag:
        return f"{mag} patch shows {body}."
    else:
        return f"Patch shows {body}."


# =========================================================
# 索引构建
# =========================================================
def build_index_5x10x(rows_5x10x: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx = defaultdict(list)
    for row in rows_5x10x:
        p5 = get_5x_path(row)
        p10 = get_10x_path(row)

        if p5 is None and p10 is None:
            parent = get_parent_path(row)
            child = get_child_path(row)
            children = get_child_paths(row)
            p5 = parent
            if child is not None:
                p10 = child
            elif children:
                for c in children:
                    idx[c].append({
                        "path_5x": p5,
                        "path_10x": c,
                        "raw_5x10x": row
                    })
                continue

        if p10 is not None:
            idx[p10].append({
                "path_5x": p5,
                "path_10x": p10,
                "raw_5x10x": row
            })
    return idx


def build_index_10x20x(rows_10x20x: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chains = []
    for row in rows_10x20x:
        p10 = get_10x_path(row)
        p20 = get_20x_path(row)

        if p10 is None and p20 is None:
            parent = get_parent_path(row)
            child = get_child_path(row)
            children = get_child_paths(row)
            p10 = parent
            if child is not None:
                p20 = child
                chains.append({
                    "path_10x": p10,
                    "path_20x": p20,
                    "raw_10x20x": row
                })
                continue
            elif children:
                for c in children:
                    chains.append({
                        "path_10x": p10,
                        "path_20x": c,
                        "raw_10x20x": row
                    })
                continue

        if p10 is not None and p20 is not None:
            chains.append({
                "path_10x": p10,
                "path_20x": p20,
                "raw_10x20x": row
            })
    return chains


def build_index_20x40x_stable(rows_20x40x_stable: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx = defaultdict(list)
    for row in rows_20x40x_stable:
        p20 = get_20x_path(row)
        p40s = get_40x_paths(row)

        if p20 is None:
            parent = get_parent_path(row)
            child = get_child_path(row)
            children = get_child_paths(row)
            p20 = parent
            if child is not None:
                p40s = [child]
            elif children:
                p40s = children

        desc = extract_descriptor_block(row)
        pseudo_caption = build_pseudocaption_from_groups(row)

        if p20 is not None:
            idx[p20].append({
                "path_20x": p20,
                "paths_40x": p40s,
                "desc_block": desc,
                "pseudo_caption": pseudo_caption,
                "raw_20x40x": row
            })
    return idx


def build_index_soft20(rows_soft20: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for row in rows_soft20:
        p20 = get_20x_path(row)
        if p20 is None:
            p20 = get_any_image_path(row)
        if p20 is not None:
            idx[p20] = {
                "path_20x": p20,
                "soft_block": extract_soft20_block(row),
                "raw_soft20": row
            }
    return idx


# =========================================================
# 合并
# =========================================================
def merge_all(
    rows_5x10x: List[Dict[str, Any]],
    rows_10x20x: List[Dict[str, Any]],
    rows_20x40x_stable: List[Dict[str, Any]],
    rows_soft20: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    idx_5x10x = build_index_5x10x(rows_5x10x)
    chains_10x20x = build_index_10x20x(rows_10x20x)
    idx_20x40x = build_index_20x40x_stable(rows_20x40x_stable)
    idx_soft20 = build_index_soft20(rows_soft20)

    merged = []

    for chain in chains_10x20x:
        p10 = chain["path_10x"]
        p20 = chain["path_20x"]

        candidates_5x = idx_5x10x.get(p10, [])
        if len(candidates_5x) == 0:
            candidates_5x = [{
                "path_5x": None,
                "path_10x": p10,
                "raw_5x10x": {}
            }]

        stable_20x40x = idx_20x40x.get(p20, [])
        soft20 = idx_soft20.get(p20, {"soft_block": {}})

        for item5 in candidates_5x:
            p5 = item5["path_5x"]

            desc_5x = extract_descriptor_block(item5["raw_5x10x"])
            desc_10x = extract_descriptor_block(chain["raw_10x20x"])

            # 这里 5x / 10x 如果没有显式 caption，就只留空
            caption_5x = ""
            caption_10x = ""

            paths_40x = []
            desc_40x_list = []
            caption_40x_list = []

            for s in stable_20x40x:
                for p40 in s["paths_40x"]:
                    if p40 is not None:
                        paths_40x.append(p40)
                        desc_40x_list.append({
                            "path_40x": p40,
                            "desc_40x": s["desc_block"]
                        })
                        caption_40x_list.append({
                            "path_40x": p40,
                            "pseudo_caption_40x": s["pseudo_caption"]
                        })

            sample_id = "|".join([
                basename_noext(p5) or "none5x",
                basename_noext(p10) or "none10x",
                basename_noext(p20) or "none20x"
            ])

            merged.append({
                "sample_id": sample_id,

                "path_5x": p5,
                "path_10x": p10,
                "path_20x": p20,
                "paths_40x": paths_40x,

                "desc_5x": desc_5x,
                "desc_10x": desc_10x,
                "desc_20x": stable_20x40x[0]["desc_block"] if len(stable_20x40x) > 0 else {},
                "soft_20x": soft20["soft_block"],
                "desc_40x_list": desc_40x_list,

                "caption_5x": caption_5x,
                "caption_10x": caption_10x,
                "pseudo_caption_20x": stable_20x40x[0]["pseudo_caption"] if len(stable_20x40x) > 0 else "",
                "pseudo_caption_40x_list": caption_40x_list,

                "raw_ref": {
                    "from_5x10x": item5["raw_5x10x"],
                    "from_10x20x": chain["raw_10x20x"],
                    "from_20x40x_count": len(stable_20x40x)
                }
            })

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_5x10x", type=str, required=True)
    parser.add_argument("--groups_10x20x", type=str, required=True)
    parser.add_argument("--stable_20x40x", type=str, required=True)
    parser.add_argument("--soft20_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    args = parser.parse_args()

    rows_5x10x = load_jsonl(args.groups_5x10x)
    rows_10x20x = load_jsonl(args.groups_10x20x)
    rows_20x40x_stable = load_jsonl(args.stable_20x40x)
    rows_soft20 = load_jsonl(args.soft20_jsonl)

    merged = merge_all(
        rows_5x10x=rows_5x10x,
        rows_10x20x=rows_10x20x,
        rows_20x40x_stable=rows_20x40x_stable,
        rows_soft20=rows_soft20
    )

    save_jsonl(merged, args.out_jsonl)

    print(f"[DONE] merged samples: {len(merged)}")
    print(f"[DONE] saved to: {args.out_jsonl}")


if __name__ == "__main__":
    main()