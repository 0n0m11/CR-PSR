#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List


def load_any_json(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == '[':
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_group_id(item: Dict[str, Any]) -> str:
    if 'group_id' in item and item['group_id']:
        return item['group_id']
    meta = item.get('metadata', {}) or {}
    slide = item.get('slide', '') or item.get('slide_id', '')
    case_id = item.get('case_id', '')
    bag_index = meta.get('bag_index', item.get('bag_index', ''))
    if case_id and slide and bag_index != '':
        return f"{case_id}__{slide}__bag_{bag_index}"
    if slide and bag_index != '':
        return f"{slide}__bag_{bag_index}"
    parent = item.get('parent_path', item.get('parent_image', ''))
    return os.path.splitext(os.path.basename(parent))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_jsonls', type=str, nargs='+', required=True,
                        help='例如 groups_5x_10x.jsonl groups_10x_20x.jsonl groups_20x_40x.jsonl')
    parser.add_argument('--out_groups_jsonl', type=str, required=True)
    parser.add_argument('--out_patch_table_jsonl', type=str, required=True)
    args = parser.parse_args()

    all_groups = []
    patch_rows = []

    for src in args.group_jsonls:
        items = load_any_json(src)
        for item in items:
            gid = build_group_id(item)
            from_mag = item.get('from_mag', '')
            to_mag = item.get('to_mag', '')
            parent = item.get('parent_path', item.get('parent_image', ''))
            children = item.get('child_paths', item.get('child_images', []))
            if parent:
                patch_rows.append({
                    'patch_path': parent,
                    'group_id': gid,
                    'role': 'parent',
                    'mag': from_mag,
                    'slide': item.get('slide', ''),
                    'case_id': item.get('case_id', ''),
                    'slide_id': item.get('slide_id', ''),
                    'source_group_file': src
                })
            for idx, ch in enumerate(children):
                patch_rows.append({
                    'patch_path': ch,
                    'group_id': gid,
                    'role': 'child',
                    'child_index': idx,
                    'mag': to_mag,
                    'slide': item.get('slide', ''),
                    'case_id': item.get('case_id', ''),
                    'slide_id': item.get('slide_id', ''),
                    'source_group_file': src
                })
            out_item = dict(item)
            out_item['group_id'] = gid
            all_groups.append(out_item)

    os.makedirs(os.path.dirname(args.out_groups_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_patch_table_jsonl), exist_ok=True)

    with open(args.out_groups_jsonl, 'w', encoding='utf-8') as fg:
        for row in all_groups:
            fg.write(json.dumps(row, ensure_ascii=False) + '\n')

    with open(args.out_patch_table_jsonl, 'w', encoding='utf-8') as fp:
        for row in patch_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'[DONE] groups: {len(all_groups)} -> {args.out_groups_jsonl}')
    print(f'[DONE] patch rows: {len(patch_rows)} -> {args.out_patch_table_jsonl}')


if __name__ == '__main__':
    main()
