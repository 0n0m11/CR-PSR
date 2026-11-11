#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List

TAG_ORDER = {
    "5x": [
        "mixed_architecture", "gland_rich", "solid_area", "fibrous_stroma", "loose_stroma",
        "irregular_distribution", "clustered_distribution", "sharp_boundary",
        "diffuse_inflammation", "necrotic_focus"
    ],
    "10x": [
        "crowded_glands", "narrow_lumen", "dilated_lumen", "gland_irregularity",
        "fibrotic_stroma", "increased_cellularity", "inflammatory_cluster", "necrotic_debris"
    ],
    "20x": [
        "cell_crowding", "nuclear_overlap", "hyperchromatic_nuclei", "conspicuous_nucleoli",
        "stromal_fibrosis", "inflammatory_cells", "mucin_like_space"
    ],
    "40x": [
        "pleomorphism", "coarse_chromatin", "prominent_nucleoli", "mitotic_figure",
        "nuclear_membrane_irregularity", "apoptotic_body", "inflammatory_background"
    ]
}

PHRASE = {
    "5x": {
        "mixed_architecture": "mixed glandular and solid architecture",
        "gland_rich": "gland-rich architecture",
        "solid_area": "focal solid areas",
        "fibrous_stroma": "fibrous stroma",
        "loose_stroma": "loose stromal background",
        "irregular_distribution": "irregular tissue distribution",
        "clustered_distribution": "clustered glandular distribution",
        "sharp_boundary": "relatively sharp tissue boundary",
        "diffuse_inflammation": "diffuse inflammatory background",
        "necrotic_focus": "focal necrotic area"
    },
    "10x": {
        "crowded_glands": "crowded glands",
        "narrow_lumen": "narrow lumina",
        "dilated_lumen": "dilated lumina",
        "gland_irregularity": "irregular gland contours",
        "fibrotic_stroma": "fibrotic stroma",
        "increased_cellularity": "increased cellular density",
        "inflammatory_cluster": "localized inflammatory cluster",
        "necrotic_debris": "focal necrotic debris"
    },
    "20x": {
        "cell_crowding": "cell crowding",
        "nuclear_overlap": "nuclear overlap",
        "hyperchromatic_nuclei": "hyperchromatic nuclei",
        "conspicuous_nucleoli": "conspicuous nucleoli",
        "stromal_fibrosis": "stromal fibrosis",
        "inflammatory_cells": "scattered inflammatory cells",
        "mucin_like_space": "mucin-like spaces"
    },
    "40x": {
        "pleomorphism": "nuclear pleomorphism",
        "coarse_chromatin": "coarse chromatin",
        "prominent_nucleoli": "prominent nucleoli",
        "mitotic_figure": "mitotic figure",
        "nuclear_membrane_irregularity": "irregular nuclear membrane",
        "apoptotic_body": "apoptotic body",
        "inflammatory_background": "inflammatory background"
    }
}

DEFAULT_FALLBACK = {
    "5x": "Simple global tissue architecture.",
    "10x": "Limited local morphologic detail.",
    "20x": "Limited intermediate cellular detail.",
    "40x": "Limited high-power nuclear detail."
}


def verbalize_from_scores(mag: str, score_dict: Dict[str, float], top_k: int = 2, threshold: float = 0.5) -> str:
    tags = TAG_ORDER.get(mag, [])
    ranked = sorted(tags, key=lambda t: score_dict.get(t, 0.0), reverse=True)
    selected: List[str] = []
    for t in ranked:
        if score_dict.get(t, 0.0) >= threshold:
            selected.append(t)
        if len(selected) >= top_k:
            break

    if not selected:
        return DEFAULT_FALLBACK.get(mag, "Limited morphologic detail.")

    phrases = [PHRASE[mag][t] for t in selected if t in PHRASE.get(mag, {})]
    if not phrases:
        return DEFAULT_FALLBACK.get(mag, "Limited morphologic detail.")

    if mag == "5x":
        if len(phrases) == 1:
            return phrases[0].capitalize() + "."
        return (phrases[0].capitalize() + " with " + ", ".join(phrases[1:]) + ".")

    if mag == "10x":
        if len(phrases) == 1:
            return phrases[0].capitalize() + "."
        return (phrases[0].capitalize() + " with " + ", ".join(phrases[1:]) + ".")

    if mag in ["20x", "40x"]:
        if len(phrases) == 1:
            return phrases[0].capitalize() + "."
        return (phrases[0].capitalize() + " and " + ", ".join(phrases[1:]) + ".")

    return phrases[0].capitalize() + "."
