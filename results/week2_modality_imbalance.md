# Week2 Modality Imbalance (Missing View-B Edges)

Dataset: Cora (Planetoid)
View-A: citation graph
View-B: kNN graph
epoch : 50

| setting | miss_b (eval) | moddrop (train) | best_test |
|---|---:|---:|---:|
| baseline | 0.00 | 0.00 | 0.8080 |
| missing-B | 0.50 | 0.00 | 0.8160 |
| +moddrop | 0.50 | 0.50 | 0.8170 |
| only_moddrop | 0.00 | 0.50 | 0.8180 |