# CR-WSI-VQA

**Cross-resolution Pseudo-semantic Representation for Multi-magnification Laryngeal H&E Pathology VQA**

> This repository currently hosts the **S5 pseudo-semantic descriptor pipeline**, one of the core contributions of the CR-PSR framework. The remaining modules (S1–S4 alignment, S6 report generation, S7 VQA data construction, S8 VQA training, and the evaluation toolchain) will be released in subsequent batches.

---

## Abstract

Multi-magnification whole-slide image (WSI) analysis is central to laryngeal squamous cell carcinoma (LSCC) pathology, where diagnostically meaningful structures span four magnifications (5×, 10×, 20×, 40×). However, most pathology visual question answering (VQA) models operate on a single magnification and discard the cross-scale correspondence that pathologists routinely exploit. We introduce **CR-PSR**, a cross-resolution pseudo-semantic representation framework that couples (i) cross-resolution feature alignment across adjacent magnifications, (ii) a compact, interpretable pseudo-semantic descriptor distilled from a frozen Quilt-LLaVA vision-language model, and (iii) a multi-magnification closed-set VQA head with question-aware masking.

CR-PSR is evaluated under a strict-nested, patient-level protocol: 69 patient-identifier groups from the HANCOCK laryngeal H&E cohort, 21,354 fixed Quilt pseudo-label questions across global, local, and cross-scale types, five patient-level folds with inner-model selection and outer refitting, and a fixed nine-class answer set. The full model achieves **66.21% accuracy and 43.40% macro-F1** under the fixed nine-class convention, with cross-resolution retrieval reaching a patient-level R@1 of **0.9724** after alignment. Hyperparameter sensitivity is characterized across learning rate and global hidden dimension, and descriptor quality is audited by two hospital pathology experts on 35 patch samples (Cohen's κ = 0.544).

The pseudo-semantic descriptor (S5) is the core of this framework: it distills the frozen Quilt-LLaVA's image-grounded knowledge into a structured, interpretable intermediate representation that is cheap to inspect, reusable across downstream tasks, and auditable by domain experts. This repository provides the complete S5 pipeline, from tag-schema definition and pseudo-label bootstrap to descriptor training, inference, and multi-scale merging.

---

## Framework Overview

![CR-WSI-VQA framework](fig1.png)

The full pipeline (S1–S8) is organized as follows:

| Stage | Scope | Role |
|-------|-------|------|
| S1 | WSI patching | Multi-resolution tiling (5×/10×/20×/40×) |
| S2 | Group building | Adjacent-resolution patch pairing |
| S3 | Feature extraction | ViT-B/16 embeddings |
| S4 | Alignment | Cross-resolution feature alignment (+ raw ViT retrieval baseline) |
| **S5** | **Descriptor** | **Quilt-LLaVA pseudo-labels → structured descriptor → 20× soft scores** |
| S6 | Report | Report generation + consistency evaluation + LLM baselines |
| S7 | VQA data | Closed-set VQA construction (fixed nine-class answer set) |
| S8 | VQA training | Masked VQA training / ablations / baselines / reduced inputs |

> This batch releases **S5** (the pseudo-semantic descriptor pipeline). The other stages follow in later batches.

---

## S5 Pseudo-semantic Descriptor Pipeline

S5 distills the frozen Quilt-LLaVA vision-language model into a compact, structured, and interpretable descriptor, which serves as the auditable pseudo-semantic intermediate representation of the framework.

| Script | Role |
|--------|------|
| `S5a_make_tag_schema_20x40x_shrunk.py` | Define the 20×/40× descriptor tag schema |
| `S5a0_build_patch_desc_dataset.py` | Build the patch descriptor dataset |
| `S5b_build_patch_table_20x40x.py` | Build the 20×/40× patch table |
| `S5c_bootstrap_pseudolabels_quilt.py` | Quilt-LLaVA pseudo-label bootstrap (frozen model) |
| `S5d_stabilize_pseudolabels.py` | Pseudo-label stabilization |
| `S5e_train_desc_child_only.py` | Descriptor training (child-only) |
| `S5f_train_desc_fusion.py` | Descriptor training (fusion) |
| `S5g_infer_desc_hybrid.py` | Descriptor inference (hybrid) |
| `S5h_infer_desc_child_only.py` | Descriptor inference (child-only) |
| `S5i_export_20x_soft_scores.py` | Export 20× soft scores |
| `S5j_merge_multiscale_pseudocaption.py` | Merge multi-scale descriptors |
| `modeling_hierapatch_desc.py` | Descriptor model definitions |
| `verbalizer.py` | Descriptor-to-text verbalization |

---

## Directory Structure

```
CR-WSI-VQA/
├── README.md
├── fig1.png             # framework overview
├── common.py            # shared utilities (load/save/split/subset)
└── s5_descriptor/       # S5 pseudo-semantic descriptor pipeline (13 files)
```

---

## Environment

- Conda environment: `llava_env` (Python 3.10, torch 2.1.2+cu121)
- Dependencies: see `requirements.txt` (included in a later batch)

Runtime environment variables:

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/llava_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/root/miniconda3/envs/mr_train/lib:/root/miniconda3/envs/llava_env/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/CR-WSI-VQA
```

> `PYTHONPATH` must point to the repository root so that `from common import ...` resolves to the top-level `common.py`.

---

## Data & Weights (not included)

| Item | Note |
|------|------|
| Quilt-LLaVA | `checkpoints/Quilt-Llava-v1.5-7b/` (required by S5c; download from HuggingFace) |
| Patches | `dataset/mrplip_processed/` (5×/10×/20×/40× tiles) |

---

## Quick Start

```bash
cd /path/to/CR-WSI-VQA
export PYTHONPATH=/path/to/CR-WSI-VQA
export LD_LIBRARY_PATH=...  # see above

# Pseudo-label bootstrap
python s5_descriptor/S5c_bootstrap_pseudolabels_quilt.py --help

# Descriptor training
python s5_descriptor/S5e_train_desc_child_only.py --help
```

---

## Upcoming Batches

- S1–S4 cross-resolution alignment
- S6 report generation
- S7 VQA data construction
- S8 VQA training (main result: 66.21% acc / 43.40% macro-F1)
- Evaluation toolchain and sensitivity analysis
