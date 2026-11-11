#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict, Counter
from common import save_jsonl



# =========================================================
# 按你当前伪标签结果定制的稳定版 group 配置
# =========================================================
STABLE_GROUPS = {
    "5x": {
        "architecture_main": ["gland_rich", "solid_area", "none"],
        "stroma_type": ["fibrous_stroma", "loose_stroma", "none"],
        "distribution_pattern": ["irregular_distribution", "clustered_distribution", "none"],
        "background_finding": ["diffuse_inflammation", "necrotic_focus", "none"],
    },
    "10x": {
        "lumen_pattern": ["narrow_lumen", "dilated_lumen", "none"],
        "gland_pattern": ["crowded_glands", "gland_irregularity", "none"],
        "stromal_background": ["fibrotic_stroma", "inflammatory_cluster", "necrotic_debris", "none"],
    },
}

# 需要直接删掉的 group
DROP_GROUPS = {
    "5x": ["boundary_pattern"],
    "10x": ["cellularity_pattern"],
}

# 需要并入 none 的稀有 choice
MAP_TO_NONE = {
    "5x": {
        "architecture_main": ["mixed_architecture"],
    },
    "10x": {},
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--out_group_config_json", type=str, required=True)
    parser.add_argument("--out_stats_json", type=str, required=True)
    args = parser.parse_args()

    rows = list(load_jsonl(args.in_jsonl))

    # 1) 先过滤 parse_ok=1
    rows = [r for r in rows if int(r.get("parse_ok", 0)) == 1]

    stable_rows = []
    stats = {
        "input_rows_after_parse_filter": len(rows),
        "per_mag": {},
    }

    group_choice_counter = defaultdict(Counter)

    for r in rows:
        mag = r["mag"]

        # 这里只处理当前稳定版关心的倍率
        if mag not in STABLE_GROUPS:
            continue

        raw_choice_by_group = dict(r.get("choice_by_group", {}))
        stable_choice_by_group = {}
        stable_pseudo_tags = {}

        # 初始化所有稳定版 tag = 0
        for group_name, choices in STABLE_GROUPS[mag].items():
            for c in choices:
                if c != "none":
                    stable_pseudo_tags[c] = 0

        # 逐 group 处理
        for group_name, keep_choices in STABLE_GROUPS[mag].items():
            old_choice = raw_choice_by_group.get(group_name, "none")

            # 稀有类并到 none
            rare_map = MAP_TO_NONE.get(mag, {}).get(group_name, [])
            if old_choice in rare_map:
                new_choice = "none"
            else:
                new_choice = old_choice

            # 若 choice 不在稳定版 choices 里，也直接归 none
            if new_choice not in keep_choices:
                new_choice = "none"

            stable_choice_by_group[group_name] = new_choice
            group_choice_counter[(mag, group_name)][new_choice] += 1

            if new_choice != "none":
                stable_pseudo_tags[new_choice] = 1

        # 删除不再使用的 group 信息，但保留原始信息方便追溯
        new_row = dict(r)
        new_row["stable_choice_by_group"] = stable_choice_by_group
        new_row["stable_pseudo_tags"] = stable_pseudo_tags
        stable_rows.append(new_row)

    # 2) 统计
    per_mag_stats = {}
    for (mag, group_name), cnt in group_choice_counter.items():
        if mag not in per_mag_stats:
            per_mag_stats[mag] = {}
        total = sum(cnt.values())
        per_mag_stats[mag][group_name] = {
            k: {
                "count": v,
                "ratio": v / max(1, total)
            }
            for k, v in cnt.items()
        }

    stats["per_mag"] = per_mag_stats
    stats["stable_groups"] = STABLE_GROUPS
    stats["drop_groups"] = DROP_GROUPS
    stats["map_to_none"] = MAP_TO_NONE
    stats["output_rows"] = len(stable_rows)

    # 3) 保存
    os.makedirs(os.path.dirname(args.out_group_config_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_stats_json), exist_ok=True)

    save_jsonl(stable_rows, args.out_jsonl)

    with open(args.out_group_config_json, "w", encoding="utf-8") as f:
        json.dump(STABLE_GROUPS, f, ensure_ascii=False, indent=2)

    with open(args.out_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[DONE] stable grouped pseudolabel file saved to:", args.out_jsonl)
    print("[DONE] stable grouped config saved to:", args.out_group_config_json)
    print("[DONE] stable stats saved to:", args.out_stats_json)


if __name__ == "__main__":
    main()