# Multimodal Graph Modality Imbalance

> A minimal reproducible benchmark for **modality imbalance** in multiview graph learning.

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Experiment Results](#experiment-results)
- [Method Overview](#method-overview)
- [License](#license)

## Background

In multimodal / multiview graph learning, different **views** (modalities) of a graph may have different quality or availability. For example, one view might have missing edges or corrupted features — a problem known as **modality imbalance**.

This project studies a simple but practical setting on the **Cora** citation network:

| View | Source | Description |
|------|--------|-------------|
| **View A** | Citation graph | Original paper citation links (who cited whom) |
| **View B** | kNN feature graph | Edges built from node feature similarity |

We simulate modality imbalance by randomly dropping edges in View B at evaluation time, and test whether **Modality Dropout** (randomly removing View B during training) can improve robustness.

## Project Structure

```
├── src/
│   ├── train_baseline.py      # Dual-view GCN baseline (both views fully available)
│   └── train_imbalance.py     # Modality imbalance benchmark (miss_b + moddrop)
├── scripts/
│   └── run_exp.py             # Automated experiment runner (multiple seeds & settings)
├── results/
│   ├── day1_baseline.md       # Baseline result log
│   ├── day2_modality_imbalance.md
│   └── day2_modality_imbalance_auto.md  # Auto-generated experiment table
├── data/
│   └── Planetoid/Cora/        # Cora dataset (auto-downloaded)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+, [Conda](https://docs.conda.io/) recommended.

```bash
# Create and activate environment
conda create -n mmgi python=3.10 -y
conda activate mmgi

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Clone and enter the repo
git clone https://github.com/<your-username>/multimodal-graph-modality-imbalance.git
cd multimodal-graph-modality-imbalance
```

## Usage

### 1. Baseline (Dual-view GCN, both views complete)

```bash
python -u src/train_baseline.py --epochs 200
```

### 2. Modality Imbalance Benchmark

```bash
# No imbalance (equivalent to baseline)
python -u src/train_imbalance.py --epochs 200 --miss_b 0.0 --moddrop 0.0

# 50% View-B edges missing at eval time (no defense)
python -u src/train_imbalance.py --epochs 200 --miss_b 0.5 --moddrop 0.0

# 50% View-B edges missing at eval + Modality Dropout during training
python -u src/train_imbalance.py --epochs 200 --miss_b 0.5 --moddrop 0.5
```

### 3. Run Full Experiment Suite (5 seeds × 3 settings)

```bash
python scripts/run_exp.py
```

Results will be saved to `results/day2_modality_imbalance_auto.md`.

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--k` | int | 10 | Number of neighbors for kNN graph (View B) |
| `--epochs` | int | 200 | Training epochs |
| `--hid` | int | 64 | GCN hidden dimension |
| `--lr` | float | 0.01 | Learning rate |
| `--wd` | float | 5e-4 | Weight decay (L2 regularization) |
| `--seed` | int | 7 | Random seed |
| `--miss_b` | float | 0.0 | Edge drop probability for View B at **eval** time |
| `--moddrop` | float | 0.0 | Probability of dropping **entire** View B during **training** |

## Experiment Results

**Dataset:** Cora (2708 nodes, 5429 edges, 7 classes)  
**Epochs:** 200 | **Seeds:** 1–5

| Setting | miss_b | moddrop | seed1 | seed2 | seed3 | seed4 | seed5 | **Avg** |
|---------|-------:|--------:|------:|------:|------:|------:|------:|--------:|
| baseline | 0.00 | 0.00 | 0.8020 | 0.8080 | 0.8010 | 0.8120 | 0.8090 | **0.8064** |
| missing-B | 0.50 | 0.00 | 0.8100 | 0.8010 | 0.8010 | 0.8090 | 0.7950 | **0.8032** |
| +moddrop | 0.50 | 0.50 | 0.8090 | 0.7910 | 0.8070 | 0.8030 | 0.8100 | **0.8040** |

### Key Observations

- **Modality imbalance hurts (slightly):** Dropping 50% of View-B edges causes a ~0.3% accuracy drop (0.8064 → 0.8032).
- **Modality Dropout helps (slightly):** Training with `moddrop=0.5` partially recovers performance under missing View B (0.8032 → 0.8040).
- **Baseline fusion is already somewhat robust:** The naive 50/50 average fusion degrades gracefully — View A alone carries most of the signal.

## Method Overview

```
                    ┌───────────────┐
                    │  Node Features │  (x: [2708, 1433])
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
     ┌────────────────┐         ┌────────────────┐
     │   View A (GCN) │         │   View B (GCN) │
     │  Citation Graph │         │   kNN Graph    │
     │  GCN→ReLU→Drop │         │  GCN→ReLU→Drop │
     │  GCN→logits_a  │         │  GCN→logits_b  │
     └───────┬────────┘         └───────┬────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
              0.5 × logits_a + 0.5 × logits_b
                        │
                        ▼
                   Prediction (7 classes)
```

**Modality Dropout (training only):** Each epoch, with probability `moddrop`, View B's edges are entirely removed, forcing the model to rely solely on View A.

## Next Steps

- [ ] Test more missing rates (0.1, 0.3, 0.7, 0.9)
- [ ] Replace naive average fusion with learned / attention-based fusion
- [ ] Add more datasets (CiteSeer, PubMed)
- [ ] Compare with existing multimodal robustness methods

## License

[MIT](LICENSE) © 2026 Bingxin Li