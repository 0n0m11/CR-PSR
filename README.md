# CR-PSR
# Multi-scale Pathology VQA Framework

This repository provides a multi-scale pathology visual question answering (VQA) framework based on visual representation learning, cross-scale feature modeling, weakly supervised semantic enhancement, and robust evaluation. The project is designed for whole-slide-image-derived pathology patches under multiple magnification levels, including 5×, 10×, 20×, and 40×.

The framework aims to improve the ability of deep learning models to jointly understand global tissue architecture, local structural morphology, and fine-grained visual patterns in pathology images.

---

## Overview

Digital pathology images contain rich information at different magnification levels. Low magnification provides global tissue structure, while high magnification reveals local cellular and morphological details. Directly using single-scale visual features often leads to unstable performance when the model needs to reason across different resolutions.

This project builds a multi-scale pathology VQA pipeline with the following components:

Multi-scale patch-level visual feature extraction
Cross-resolution visual representation modeling
Weakly supervised pseudo-semantic descriptor construction
Question-aware visual-semantic feature fusion
Closed-set VQA prediction
Baseline comparison, ablation study, and robustness evaluation
Cross-dataset inference evaluation

---

## Features

* Multi-scale pathology image modeling (5× / 10× / 20× / 40×)
* Vision Transformer based feature extraction
* Cross-scale visual representation learning
* Weak semantic enhancement
* Visual Question Answering (VQA)
* Ablation and robustness evaluation
* Cross-dataset generalization testing

---

## Project Structure

```text
.
├── configs/
├── data/
├── scripts/
├── src/
├── outputs/
├── docs/
├── requirements.txt
└── README.md
```

---

## Installation

Create a Python environment:

```bash
conda create -n pathology_vqa python=3.10 -y
conda activate pathology_vqa
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Prepare multi-scale pathology patches and the corresponding VQA annotations.

Example:

```text
data/
├── patches/
├── qa_train.jsonl
├── qa_val.jsonl
└── qa_test.jsonl
```

---

## Training

Train the VQA model:

```bash
python scripts/train_multiscale_vqa.py \
  --config configs/train.yaml
```

---

## Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate_vqa.py \
  --config configs/eval.yaml
```

Metrics include:

* Accuracy
* Macro-F1
* Question-type accuracy
* Cross-dataset performance

---

## Baselines

The repository includes several baseline settings:

* Raw visual feature baseline
* Feature fusion baseline
* Visual-only baseline
* Semantic-enhanced baseline
* Input-shuffling control experiments

These baselines are provided for reproducibility and fair comparison.

---

## Visualization

Supported visualizations include:

* Cross-resolution retrieval heatmaps
* Performance comparison plots
* Ablation study figures
* Robustness evaluation results

---

---

## License

This project is released for academic research purposes.

---

## Contact

For questions or suggestions, please open an issue.

