#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse
import hashlib
import random
from typing import Dict, Any, List, Optional

# =========================================================
# 优先补 LLaVA 源码路径
# 注意：先放 /mnt/MR-PLIP/LLaVA，避免误用 /mnt/MR-PLIP/llava
# =========================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_ROOTS = [
    "/mnt/MR-PLIP/LLaVA",
    "/mnt/MR-PLIP",
    os.path.abspath(os.path.join(THIS_DIR, "..")),
]

for root in CANDIDATE_ROOTS:
    if os.path.isdir(os.path.join(root, "llava")) and root not in sys.path:
        sys.path.insert(0, root)

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# =========================================================
# choice 的自然语言描述
# 新旧版本都兼容
# =========================================================
CHOICE_DESC = {
    # 5x
    "gland_rich": "gland-rich architecture",
    "solid_area": "solid growth area",
    "mixed_architecture": "mixed glandular and solid architecture",
    "fibrous_stroma": "fibrous stroma",
    "loose_stroma": "loose stroma",
    "irregular_distribution": "irregular tissue distribution",
    "clustered_distribution": "clustered tissue distribution",
    "sharp_boundary": "sharp tissue boundary",
    "diffuse_inflammation": "diffuse inflammation",
    "necrotic_focus": "necrotic focus",

    # 10x
    "narrow_lumen": "narrow lumen",
    "dilated_lumen": "dilated lumen",
    "crowded_glands": "crowded glands",
    "gland_irregularity": "irregular gland contour",
    "fibrotic_stroma": "fibrotic stroma",
    "inflammatory_cluster": "localized inflammatory cluster",
    "necrotic_debris": "necrotic debris",
    "increased_cellularity": "increased cellularity",

    # 20x old
    "cell_crowding": "cell crowding",
    "nuclear_overlap": "nuclear overlap",
    "hyperchromatic_nuclei": "hyperchromatic nuclei",
    "conspicuous_nucleoli": "conspicuous nucleoli",
    "stromal_fibrosis": "stromal fibrosis",
    "inflammatory_cells": "inflammatory cells",
    "mucin_like_space": "mucin-like space",

    # 40x old
    "pleomorphism": "nuclear pleomorphism",
    "nuclear_membrane_irregularity": "irregular nuclear membrane",
    "coarse_chromatin": "coarse chromatin",
    "prominent_nucleoli": "prominent nucleoli",
    "mitotic_figure": "mitotic figure",
    "apoptotic_body": "apoptotic body",
    "inflammatory_background": "inflammatory background",

    # 40x shrunk
    "nuclear_abnormality": "nuclear abnormality",
    "nuclear_atypia_present": "obvious nuclear atypia present",
    "mitotic_activity": "mitotic or apoptotic activity",
    "mitotic_or_apoptotic_activity": "mitotic or apoptotic activity present",
    "necrotic_background": "necrotic background",

    "none": "none of the above / no clear feature"
}


# =========================================================
# 工具函数
# =========================================================
def load_image(path: str):
    return Image.open(path).convert("RGB")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_shuffle(items: List[str], key: str) -> List[str]:
    items = list(items)
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("```", "")
    text = text.replace("`", "")
    text = text.replace('"', "")
    text = text.replace("'", "")

    # 修复 \_
    text = text.replace("\\_", "_")
    text = text.replace("\\", "")

    text = text.strip(" .,:;!?")
    return text


def token_to_desc(token: str) -> str:
    if token in CHOICE_DESC:
        return CHOICE_DESC[token]
    return token.replace("_", " ")


def parse_choice(text: str, choices: List[str]) -> Optional[str]:
    t = normalize_text(text)
    if not t:
        return None

    # 1) 完全相等优先
    for c in choices:
        if t == c.lower():
            return c

    # 2) whole-word 匹配
    for c in choices:
        pattern = r"\b" + re.escape(c.lower()) + r"\b"
        if re.search(pattern, t):
            return c

    # 3) 描述匹配
    for c in choices:
        desc = token_to_desc(c).lower()
        desc = desc.replace("\\_", "_")
        if desc in t:
            return c

    # 4) 对 none 宽松兜底
    if "none" in choices and (
        "none" in t
        or "no clear feature" in t
        or "no obvious feature" in t
        or "none of the above" in t
    ):
        return "none"

    return None


# =========================================================
# 从 schema 动态构造 group
# schema 格式：
# {
#   "20x": {
#       "cell_density_pattern": ["cell_crowding", "nuclear_overlap", "none"],
#       ...
#   },
#   ...
# }
# =========================================================
QUESTION_TEMPLATES = {
    "architecture_main": "Which global architecture is most evident in this patch?",
    "stroma_type": "Which stromal pattern is more evident?",
    "distribution_pattern": "Which tissue distribution pattern is more evident?",
    "boundary_pattern": "Is a sharp boundary evident?",
    "background_finding": "Which additional background finding is more evident?",

    "lumen_pattern": "Which lumen pattern is more evident?",
    "gland_pattern": "Which gland pattern is more evident?",
    "stromal_background": "Which stromal or background finding is more evident?",
    "cellularity_pattern": "Is increased cellularity evident?",

    "cell_density_pattern": "Which cell density pattern is more evident?",
    "nuclear_feature": "Which nuclear feature is more evident?",
    "background_detail": "Which background feature is more evident?",

    "nuclear_atypia": "Which nuclear atypia pattern is more evident?",
    "chromatin_nucleoli": "Which chromatin or nucleolar feature is more evident?",
    "mitosis_apoptosis_background": "Which additional finding is more evident?",

    # shrunk
    "nuclear_abnormality": "Is obvious nuclear abnormality evident?",
    "mitotic_activity": "Is mitotic or apoptotic activity evident?",
    "micro_background": "Which microscopic background finding is more evident?",
}


def default_question_from_group(group_name: str, choices: List[str]) -> str:
    # 二分类：X / none
    non_none = [c for c in choices if c != "none"]
    if len(non_none) == 1 and "none" in choices:
        return f"Is {token_to_desc(non_none[0])} evident?"
    return f"Which {group_name.replace('_', ' ')} is more evident?"


def build_groups_from_schema(schema: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups_by_mag = {}
    for mag, group_map in schema.items():
        groups = []
        for group_name, choices in group_map.items():
            if not isinstance(choices, list) or len(choices) == 0:
                continue

            question = QUESTION_TEMPLATES.get(
                group_name,
                default_question_from_group(group_name, choices)
            )

            groups.append({
                "group_name": group_name,
                "question": question,
                "choices": choices
            })
        groups_by_mag[mag] = groups
    return groups_by_mag


# =========================================================
# Quilt / LLaVA 推理器
# =========================================================
class QuiltForcedChoiceTagger:
    def __init__(self, model_path: str, conv_mode: str = "llava_v1", device: str = "cuda"):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device=device,
        )
        self.device = device
        self.conv_mode = conv_mode
        self.model.eval()

    def infer_text(self, image_path: str, prompt_text: str, max_new_tokens: int = 16) -> str:
        image = load_image(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32

        if isinstance(image_tensor, list):
            image_tensor = [x.to(self.device, dtype=dtype) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=dtype)

        if getattr(self.model.config, "mm_use_im_start_end", False):
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        if output_ids.shape[1] > input_ids.shape[1]:
            gen_ids = output_ids[:, input_ids.shape[1]:]
        else:
            gen_ids = output_ids

        out = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return out


# =========================================================
# Prompt
# =========================================================
def build_forced_choice_prompt(mag: str, group_name: str, question: str, choices: List[str]) -> str:
    choice_lines = []
    for c in choices:
        choice_lines.append(f"- {c}: {token_to_desc(c)}")

    choice_block = "\n".join(choice_lines)
    choice_tokens = ", ".join(choices)

    return f"""
You are labeling a pathology patch for a structured descriptor model.

Magnification: {mag}
Group: {group_name}

Question:
{question}

Available choices:
{choice_block}

Rules:
1. Select exactly one choice.
2. Return only the choice token.
3. Do not explain.
4. Do not output diagnosis words.
5. If none is clearly visible, return none.

Allowed output tokens:
{choice_tokens}
""".strip()


# =========================================================
# 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_table_jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--schema_json", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    with open(args.schema_json, "r", encoding="utf-8") as f:
        schema = json.load(f)

    groups_by_mag = build_groups_from_schema(schema)

    print("[INFO] loaded schema mags:", list(groups_by_mag.keys()))
    for mag, groups in groups_by_mag.items():
        print(f"[INFO] {mag}: {[g['group_name'] for g in groups]}")

    tagger = QuiltForcedChoiceTagger(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fw:
        for row in tqdm(load_jsonl(args.patch_table_jsonl), desc="Pseudo labeling"):
            mag = row["mag"]
            groups = groups_by_mag.get(mag, [])
            if not groups:
                continue

            all_tags = []
            for g in groups:
                for c in g["choices"]:
                    if c != "none" and c not in all_tags:
                        all_tags.append(c)

            pseudo_tags = {t: 0 for t in all_tags}
            raw_by_group = {}
            choice_by_group = {}

            parsed_group_count = 0
            total_group_count = len(groups)

            for g in groups:
                group_name = g["group_name"]
                question = g["question"]
                choices = g["choices"]

                shuffled_choices = stable_shuffle(
                    choices,
                    row["patch_path"] + "::" + mag + "::" + group_name
                )

                prompt = build_forced_choice_prompt(
                    mag=mag,
                    group_name=group_name,
                    question=question,
                    choices=shuffled_choices,
                )

                raw = tagger.infer_text(
                    image_path=row["patch_path"],
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                parsed_choice = parse_choice(raw, shuffled_choices)

                raw_by_group[group_name] = raw
                choice_by_group[group_name] = parsed_choice if parsed_choice is not None else "__parse_fail__"

                if parsed_choice is not None:
                    parsed_group_count += 1
                    if parsed_choice != "none":
                        pseudo_tags[parsed_choice] = 1

            out = dict(row)
            out["raw_by_group"] = raw_by_group
            out["choice_by_group"] = choice_by_group
            out["pseudo_tags"] = pseudo_tags
            out["parse_ok"] = 1 if parsed_group_count == total_group_count else 0
            out["parsed_group_count"] = parsed_group_count
            out["total_group_count"] = total_group_count

            fw.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[DONE] saved -> {args.out_jsonl}")


if __name__ == "__main__":
    main()