# CR-PSR
# Multi-scale Pathology VQA Framework

This repository provides a multi-scale pathology visual question answering (VQA) framework based on visual representation learning, cross-scale feature modeling, weakly supervised semantic enhancement, and robust evaluation. The project is designed for whole-slide-image-derived pathology patches under multiple magnification levels, including 5×, 10×, 20×, and 40×.

The framework aims to improve the ability of deep learning models to jointly understand global tissue architecture, local structural morphology, and fine-grained visual patterns in pathology images.

---

## 1. Overview

Digital pathology images contain rich information at different magnification levels. Low magnification provides global tissue structure, while high magnification reveals local cellular and morphological details. Directly using single-scale visual features often leads to unstable performance when the model needs to reason across different resolutions.

This project builds a multi-scale pathology VQA pipeline with the following components:

* Multi-scale patch-level visual feature extraction
* Cross-resolution visual representation modeling
* Weakly supervised pseudo-semantic descriptor construction
* Question-aware visual-semantic feature fusion
* Closed-set VQA prediction
* Baseline comparison, ablation study, and robustness evaluation
* Cross-dataset inference evaluation

---

## 2. Main Features

### Multi-scale pathology representation

The framework supports pathology patches from multiple magnification levels:

```text
5×   global tissue architecture
10×  local structural morphology
20×  fine-grained semantic evidence
40×  detailed high-resolution visual context
```

### Visual feature extraction

A frozen visual encoder, such as ViT-B/16, is used to extract patch-level visual features. The extracted features are then organized according to their multi-scale parent-child relationship.

### Cross-resolution modeling

The framework includes a feature-level alignment and fusion strategy to reduce representation mismatch across magnification levels and improve cross-scale visual consistency.

### Pseudo-semantic descriptor construction

Weakly supervised semantic information can be generated from pathology-oriented vision-language models and converted into structured descriptors through:

* pseudo-caption generation
* schema-based parsing
* terminology normalization
* stable group-level aggregation

### Question-aware VQA fusion

The VQA module combines:

* aligned multi-scale visual features
* question representation
* structured semantic descriptor
* uncertainty-aware soft semantic evidence

The final answer is predicted through a classification head.

### Robustness and evaluation

The repository supports:

* Raw visual feature baseline
* Alignment-only visual baseline
* Reduced-input baseline
* Descriptor-shuffled control
* Visual-shuffled control
* Bootstrap confidence interval
* Multi-seed stability test
* External no-retraining evaluation

---

## 3. Repository Structure

A recommended project structure is shown below:

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── train_vqa.yaml
│   ├── alignment.yaml
│   └── eval.yaml
├── data/
│   ├── example_qa.jsonl
│   ├── example_split.json
│   └── schema_candidate_labels.json
├── scripts/
│   ├── extract_vit_features.py
│   ├── train_alignment.py
│   ├── build_pseudo_semantic_descriptors.py
│   ├── train_multiscale_vqa.py
│   ├── evaluate_vqa.py
│   ├── run_ablation_baselines.py
│   └── bootstrap_ci.py
├── src/
│   ├── datasets/
│   ├── models/
│   ├── alignment/
│   ├── descriptors/
│   ├── vqa/
│   └── utils/
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
└── docs/
    ├── method_overview.md
    ├── data_format.md
    └── evaluation_protocol.md
```

The actual directory names can be adjusted according to your local implementation.

---

## 4. Installation

### 4.1 Create environment

```bash
conda create -n pathology_vqa python=3.10 -y
conda activate pathology_vqa
```

### 4.2 Install dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm pillow pyyaml
```

Optional dependencies:

```bash
pip install timm transformers opencv-python
```

---

## 5. Data Preparation

The expected input data consists of multi-scale patches and a VQA annotation file.

### 5.1 Multi-scale patch structure

A typical patch directory may look like:

```text
data/patches/
├── slide_001/
│   ├── 5x.png
│   ├── 10x_0.png
│   ├── 20x_0.png
│   ├── 40x_0.png
│   ├── 40x_1.png
│   └── ...
├── slide_002/
│   ├── 5x.png
│   ├── 10x_0.png
│   ├── 20x_0.png
│   └── ...
```

### 5.2 VQA JSONL format

Each row in the VQA file should contain one sample:

```json
{
  "question_id": "qtrain_00000001",
  "base_sample_id": "sample_00000001",
  "question_type": "global",
  "answer_type": "single_choice",
  "question": "What is the dominant global architecture?",
  "choices": ["gland_rich", "solid_area", "none"],
  "answer": "gland_rich",
  "image_5x": "path/to/5x.png",
  "image_10x": "path/to/10x.png",
  "image_20x": "path/to/20x.png",
  "images_40x": ["path/to/40x_0.png", "path/to/40x_1.png"]
}
```

### 5.3 Recommended split protocol

To avoid patch-level leakage, the recommended split mode is slide-level split:

```text
Train / Validation / Test are separated by slide or WSI identity.
No identical slide or patch should appear in more than one split.
```

If patient-level split is available, it is recommended for stricter evaluation.

---

## 6. Feature Extraction

Visual features can be extracted using a frozen ViT encoder.

Example:

```bash
python scripts/extract_vit_features.py \
  --image_root data/patches \
  --output_dir outputs/features \
  --backbone vit_base_patch16_224 \
  --batch_size 64
```

Expected outputs:

```text
outputs/features/feature_bank.pt
outputs/features/path_to_idx.json
```

`feature_bank.pt` stores visual features, and `path_to_idx.json` maps image paths to feature indices.

---

## 7. Cross-resolution Feature Modeling

The cross-resolution module is used to improve consistency among features from different magnification levels.

Example:

```bash
python scripts/train_alignment.py \
  --feature_bank outputs/features/feature_bank.pt \
  --path_to_idx outputs/features/path_to_idx.json \
  --qa_jsonl data/example_qa.jsonl \
  --save_dir outputs/alignment
```

Typical evaluation metrics include:

```text
P2C R@1
C2P R@1
Mean R@1
Mean R@5
Pair AUC
Pair AP
Pair accuracy
```

---

## 8. Pseudo-semantic Descriptor Construction

Pseudo-semantic descriptors are generated from weak semantic cues and converted into structured schema-level labels.

The construction process includes:

```text
pseudo-caption generation
schema parsing
label normalization
group-level stable aggregation
question-aware masking
```

Example:

```bash
python scripts/build_pseudo_semantic_descriptors.py \
  --qa_jsonl data/example_qa.jsonl \
  --schema data/schema_candidate_labels.json \
  --output_jsonl outputs/descriptors/descriptors.jsonl
```

A descriptor may include fields such as:

```text
architecture_main
stroma_type
distribution_pattern
background_finding
lumen_pattern
gland_pattern
stromal_background
20x_soft_score
20x_margin
20x_entropy_proxy
```

The exact schema can be modified according to the dataset and task definition.

---

## 9. VQA Training

Train the multi-scale VQA model:

```bash
python scripts/train_multiscale_vqa.py \
  --qa_jsonl data/example_qa.jsonl \
  --feature_bank outputs/features/feature_bank.pt \
  --path_to_idx outputs/features/path_to_idx.json \
  --descriptor_jsonl outputs/descriptors/descriptors.jsonl \
  --save_dir outputs/checkpoints/full_model \
  --batch_size 64 \
  --epochs 12 \
  --lr 1e-4 \
  --hidden_dim 512 \
  --seed 42
```

The model predicts answers from a closed candidate answer set.

---

## 10. Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate_vqa.py \
  --checkpoint outputs/checkpoints/full_model/best.pt \
  --qa_jsonl data/example_qa.jsonl \
  --feature_bank outputs/features/feature_bank.pt \
  --path_to_idx outputs/features/path_to_idx.json \
  --output_dir outputs/metrics/full_model
```

Main metrics:

```text
Accuracy
Macro-F1
Question-type accuracy
Question-type Macro-F1
Answer-type accuracy
Answer-type Macro-F1
```

---

## 11. Baselines and Ablation Studies

The framework supports several baselines.

### Raw ViT feature-fusion baseline

This baseline does not use cross-resolution modeling or pseudo-semantic descriptors.

```bash
python scripts/run_ablation_baselines.py \
  --mode raw_vit_avg \
  --qa_jsonl data/example_qa.jsonl \
  --feature_bank outputs/features/feature_bank.pt \
  --path_to_idx outputs/features/path_to_idx.json \
  --save_dir outputs/baselines/raw_vit_avg
```

Supported modes:

```text
raw_vit_avg
raw_vit_concat
alignment_only_avg
alignment_only_concat
descriptor_removed
question_only
visual_only
```

### Input-shuffling controls

To test whether the model relies on valid visual and semantic correspondence:

```bash
python scripts/run_ablation_baselines.py \
  --mode descriptor_shuffled \
  --qa_jsonl data/example_qa.jsonl \
  --save_dir outputs/controls/descriptor_shuffled
```

Common controls:

```text
descriptor_shuffled
descriptor_soft_shuffled
visual_shuffled
```

---

## 12. Bootstrap Confidence Interval

Bootstrap confidence intervals can be computed for stable reporting.

```bash
python scripts/bootstrap_ci.py \
  --runs \
    full_model=outputs/checkpoints/full_model \
    raw_vit_avg=outputs/baselines/raw_vit_avg \
    alignment_only=outputs/baselines/alignment_only_concat \
  --baseline_name full_model \
  --n_boot 2000 \
  --seed 2026 \
  --out_dir outputs/metrics/bootstrap
```

The output includes:

```text
Accuracy 95% CI
Macro-F1 95% CI
Delta Accuracy 95% CI
Delta Macro-F1 95% CI
```

---

## 13. External No-retraining Evaluation

The framework supports no-retraining external evaluation. In this setting, the model trained on the source dataset is directly evaluated on external datasets without additional fine-tuning.

Example:

```bash
python scripts/evaluate_vqa.py \
  --checkpoint outputs/checkpoints/full_model/best.pt \
  --qa_jsonl data/external_qa.jsonl \
  --feature_bank outputs/external_features/feature_bank.pt \
  --path_to_idx outputs/external_features/path_to_idx.json \
  --output_dir outputs/external_eval
```

This protocol can be used to evaluate cross-dataset robustness and generalization.

---

## 14. Example Results

Example internal VQA results:

```text
Full model Accuracy: 71.51%
Full model Macro-F1: 67.09%
```

Example baseline comparison:

```text
Raw ViT avg-pooling Accuracy: 65.75%
Raw ViT concat Accuracy: 64.36%
Alignment-only concat Accuracy: 68.96%
```

These values are provided only as an example. Actual results may vary depending on dataset, split protocol, feature extractor, and training configuration.

---

## 15. Visualization

The repository can generate figures such as:

```text
cross-resolution retrieval heatmap
question-type performance bar plot
answer-type performance bar plot
pseudo-semantic construction quality plot
baseline comparison plot
bootstrap confidence interval table
```

Example:

```bash
python scripts/plot_cross_resolution_heatmap.py
```

---

## 16. Reproducibility

For reproducible experiments, please record:

```text
random seed
dataset split
feature extractor checkpoint
training configuration
evaluation configuration
software environment
```

Recommended settings:

```text
seed: 42
batch size: 64
epochs: 12
optimizer: AdamW
learning rate: 1e-4
hidden dimension: 512
```

---

## 17. Limitations

This project focuses on multi-scale pathology image understanding and weakly supervised visual-semantic enhancement. Several limitations remain:

* The quality of multi-scale patch pairing affects model performance.
* Weak semantic descriptors may contain noise.
* External datasets may have different staining, tissue preparation, or annotation protocols.
* Deployment on edge devices may require additional compression, quantization, or inference acceleration.

---


Please replace the citation with the final paper information after publication.

---

## 18. License

This project is released for academic research purposes. Please check the license file before commercial use.

Recommended license options:

```text
MIT License
Apache License 2.0
CC BY-NC 4.0 for non-commercial research
```

---

## 19. Contact

For questions or suggestions, please open an issue or contact the project maintainer.

```text
Maintainer: Liu Xiaodan
Email:xiaodanliu43@gmail.com
```
