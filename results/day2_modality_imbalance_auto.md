# Day2 Modality Imbalance (Missing View-B Edges)

Dataset: Cora (Planetoid)  
View-A: citation graph  
View-B: kNN graph  
epoch: 200  
seeds: 1, 2, 3, 4, 5  

| setting | miss_b (eval) | moddrop (train) | seed1 | seed2 | seed3 | seed4 | seed5 | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.00 | 0.8020 | 0.8080 | 0.8010 | 0.8120 | 0.8090 | 0.8064 |
| missing-B | 0.50 | 0.00 | 0.8100 | 0.8010 | 0.8010 | 0.8090 | 0.7950 | 0.8032 |
| +moddrop | 0.50 | 0.50 | 0.8090 | 0.7910 | 0.8070 | 0.8030 | 0.8100 | 0.8040 |

## Observation

Under the fixed evaluation setting, introducing 50% missing edges in View-B led to a slight performance drop compared with the full-view baseline (0.8032 vs. 0.8064 on average).

Adding simple Modality Dropout slightly improved the missing-B setting (0.8040 vs. 0.8032), but it still did not surpass the full-view baseline.

This suggests that the current dual-view average fusion already has some robustness under moderate View-B corruption, while simple Modality Dropout provides only limited gains in the current setting.

However, the conclusion is still preliminary and should be validated with more missing rates, stronger fusion strategies, and more extensive experiments.