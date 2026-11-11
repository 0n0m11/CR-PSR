#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

SCHEMA_SHRUNK = {
    "20x": {
        "cell_density_pattern": [
            "cell_crowding",
            "nuclear_overlap",
            "none"
        ],
        "background_detail": [
            "stromal_fibrosis",
            "inflammatory_cells",
            "mucin_like_space",
            "none"
        ]
    },
    "40x": {
        "nuclear_abnormality": [
            "nuclear_atypia_present",
            "none"
        ],
        "mitotic_activity": [
            "mitotic_or_apoptotic_activity",
            "none"
        ]
    }
}

def main():
    out_dir = "/mnt/CR_PSR/hierapatch_desc"
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "tag_schema_20x40x_shrunk.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(SCHEMA_SHRUNK, f, ensure_ascii=False, indent=2)

    print("[DONE] saved:", out_json)

if __name__ == "__main__":
    main()