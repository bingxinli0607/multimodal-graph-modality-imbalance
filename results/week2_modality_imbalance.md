# Day2 Modality Imbalance (Missing View-B Edges)

## test

### first(no_seed)
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

### second(not_fixed_drop_edge)
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

#### Observation

Under the fixed evaluation setting (`fixed_drop_edge`), introducing 50% missing edges in View-B did not significantly hurt performance on Cora.
Moreover, adding simple Modality Dropout did not outperform the missing-B baseline in this initial single-seed experiment.

A possible explanation is that dropping part of View-B edges may remove noisy or less useful kNN connections, so the model can still rely on View-A and the remaining View-B structure.
However, this is still a preliminary result. More seeds and more missing rates are needed before drawing a reliable conclusion.

### third(fixed_drop_edge)

Dataset: Cora (Planetoid)  
View-A: citation graph  
View-B: kNN graph  
epoch: 200  
seeds: 1

| setting | miss_b (eval) | moddrop (train) | seed1 |
|---|---:|---:|---:|
| baseline | 0.00 | 0.00 | 0.8020  |
| missing-B | 0.50 | 0.00 | 0.8100 |
| +moddrop | 0.50 | 0.50 | 0.8090  |

#### Observation

Under the fixed evaluation setting (`fixed_drop_edge`), introducing 50% missing edges in View-B did not significantly hurt performance on Cora.
Moreover, adding simple Modality Dropout did not outperform the missing-B baseline in this initial single-seed experiment.

A possible explanation is that dropping part of View-B edges may remove noisy or less useful kNN connections, so the model can still rely on View-A and the remaining View-B structure.
However, this is still a preliminary result. More seeds and more missing rates are needed before drawing a reliable conclusion.
