# Multimodal Graph Modality Imbalance

A minimal reproducible benchmark for **modality imbalance** in **multiview / multimodal graph learning**.

## Goal
This project studies a simple but practical setting:

- View-A: citation graph
- View-B: kNN graph built from node features

We simulate **modality imbalance** by randomly dropping edges in View-B at evaluation time, and test whether a simple training strategy (**Modality Dropout**) can improve robustness.

## Environment
- Python 3.10
- PyTorch / PyTorch Geometric
- CPU only

## Files
- `src/train_baseline.py`: basic dual-view GCN baseline
- `src/train_imbalance.py`: missing-modality benchmark with `miss_b` and `moddrop`
- `results/week1_baseline.md`: baseline result log
- `results/week2_modality_imbalance.md`: imbalance experiment results

## Main Settings
- Dataset: Cora
- View-A: citation graph
- View-B: kNN graph
- Model: DualGCN
- Imbalance setting: missing edges in View-B

## Quick Start
Baseline:
```bash
python -u src/train_baseline.py --epochs 5
```

Imbalance benchmark:
```bash
python -u src/train_imbalance.py --epochs 50 --miss_b 0.0 --moddrop 0.0
python -u src/train_imbalance.py --epochs 50 --miss_b 0.5 --moddrop 0.0
python -u src/train_imbalance.py --epochs 50 --miss_b 0.5 --moddrop 0.5
python -u src/train_imbalance.py --epochs 50 --miss_b 0.0 --moddrop 0.5
```

## Current Results
See:
results/week1_baseline.md
results/week2_modality_imbalance.md

## Next Step
run with more epochs / more random seeds
add a stronger fusion strategy
compare robustness under different missing rates