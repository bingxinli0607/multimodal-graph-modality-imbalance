# Week2 Modality Imbalance (Missing View-B Edges)

## test

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

Dataset: Cora (Planetoid)  
View-A: citation graph  
View-B: kNN graph  
epoch: 50  
seeds: 1, 2, 3  

| setting | miss_b (eval) | moddrop (train) | seed1 | seed2 | seed3 | avg |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.00 | 0.8020 | 0.8080 | 0.8010 | 0.8037 |
| missing-B | 0.50 | 0.00 | 0.8160 | 0.8210 | 0.8000 | 0.8123 |
| +moddrop | 0.50 | 0.50 | 0.8070 | 0.8040 | 0.8070 | 0.8060 |

## Observation

Under the current 3-seed, 50-epoch setting, introducing 50% missing edges in View-B did not significantly degrade performance.
Moreover, simple Modality Dropout did not outperform the missing-B baseline.

This suggests that the current dual-view average fusion may already have some robustness under this setup, or that dropping View-B edges partly removes noisy connections.
However, this conclusion is still preliminary because the experiments were conducted with a small number of seeds and a relatively short training schedule.
Future work will use more seeds, more epochs, and multiple missing rates for a more reliable comparison.