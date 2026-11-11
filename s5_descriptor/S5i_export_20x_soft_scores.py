#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from common import save_jsonl



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
    args = parser.parse_args()

    out_rows = []

    for row in load_jsonl(args.in_jsonl):
        score_map = row.get("pred_scores_by_group", {}).get("cell_density_pattern", {})
        p_crowding = float(score_map.get("cell_crowding", 0.0))
        p_overlap = float(score_map.get("nuclear_overlap", 0.0))
        p_none = float(score_map.get("none", 0.0))

        out = {
            "patch_path": row["patch_path"],
            "mag": row.get("mag", "20x"),
            "source_group_file": row.get("source_group_file", ""),
            "cell_density_soft_score": {
                "p_cell_crowding": p_crowding,
                "p_nuclear_overlap": p_overlap,
                "p_none": p_none
            },
            "cell_density_margin": p_crowding - p_overlap,
            "cell_density_entropy_proxy": 1.0 - abs(p_crowding - p_overlap),
            "cell_density_hard_pred": "cell_crowding" if p_crowding >= p_overlap else "nuclear_overlap"
        }
        out_rows.append(out)

    save_jsonl(out_rows, args.out_jsonl)
    print(f"[DONE] saved -> {args.out_jsonl}")


if __name__ == "__main__":
    main()