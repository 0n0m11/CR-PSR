#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

IN_FILES = [
    "/mnt/CR_PSR/align_artifacts_clean/groups_10x_20x.jsonl",
    "/mnt/CR_PSR/align_artifacts_clean/groups_20x_40x.jsonl",
]

OUT_FILE = "/mnt/CR_PSR/hierapatch_desc/all_patch_table_20x40x.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    seen = set()
    total = 0

    with open(OUT_FILE, "w", encoding="utf-8") as fw:
        for in_file in IN_FILES:
            for row in load_jsonl(in_file):
                parent_mag = row["from_mag"]
                child_mag = row["to_mag"]
                parent_path = row.get("parent_path", row.get("parent_image", ""))
                child_paths = row.get("child_paths", row.get("child_images", []))

                if parent_path:
                    key = (parent_mag, parent_path)
                    if key not in seen:
                        seen.add(key)
                        out = {
                            "patch_path": parent_path,
                            "mag": parent_mag,
                            "source_group_file": in_file
                        }
                        fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                        total += 1

                for p in child_paths:
                    key = (child_mag, p)
                    if key not in seen:
                        seen.add(key)
                        out = {
                            "patch_path": p,
                            "mag": child_mag,
                            "source_group_file": in_file
                        }
                        fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                        total += 1

    print(f"[DONE] saved -> {OUT_FILE}")
    print(f"[INFO] total unique patches = {total}")

if __name__ == "__main__":
    main()