Ablation: pass identities / program features / both (direct target)

| metric | context | protocol | mode | model | mae | rmse | r2 | spearman | macro_f1 | residual_r2 | per_program_spearman_median | per_program_ndcg10_mean | mae_diff_vs_pairmean | mae_diff_ci_lo | mae_diff_ci_hi | wilcoxon_p_holm | programs_better | programs_worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text_size | C0 | kfold | pass | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | pass | HistGradientBoosting | 0.0096 | 0.0426 | 0.2890 | 0.3141 | 0.7200 | 0.0000 | 0.3345 | 0.9040 | -0.0000 | -0.0000 | -0.0000 | 0.0005 | 17.0000 | 84.0000 |
| text_size | C0 | kfold | features | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | features | HistGradientBoosting | 0.0129 | 0.0506 | -0.0038 | 0.0221 | 0.3299 | -0.4117 | -- | 0.6317 | 0.0033 | 0.0025 | 0.0041 | 0.0000 | 20.0000 | 81.0000 |
| text_size | C0 | kfold | both | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | both | HistGradientBoosting | 0.0096 | 0.0416 | 0.3225 | 0.2986 | 0.6703 | 0.0472 | 0.3114 | 0.9140 | -0.0000 | -0.0005 | 0.0004 | 0.6197 | 51.0000 | 50.0000 |
| text_size | C1 | kfold | pass | PairMeanBaseline | 0.0049 | 0.0332 | 0.0183 | 0.0649 | 0.4479 | 0.0000 | 0.0582 | 0.6958 | -- | -- | -- | -- | -- | -- |
| text_size | C1 | kfold | pass | HistGradientBoosting | 0.0049 | 0.0332 | 0.0183 | 0.0617 | 0.4479 | 0.0000 | 0.0583 | 0.6958 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | 101.0000 | 0.0000 |
| text_size | C1 | kfold | features | PairMeanBaseline | 0.0049 | 0.0332 | 0.0183 | 0.0649 | 0.4479 | 0.0000 | 0.0582 | 0.6958 | -- | -- | -- | -- | -- | -- |
| text_size | C1 | kfold | features | HistGradientBoosting | 0.0057 | 0.0337 | -0.0106 | 0.0306 | 0.3575 | -0.0294 | -- | 0.6297 | 0.0008 | 0.0002 | 0.0016 | 0.7310 | 57.0000 | 44.0000 |
| text_size | C1 | kfold | both | PairMeanBaseline | 0.0049 | 0.0332 | 0.0183 | 0.0649 | 0.4479 | 0.0000 | 0.0582 | 0.6958 | -- | -- | -- | -- | -- | -- |
| text_size | C1 | kfold | both | HistGradientBoosting | 0.0057 | 0.0352 | -0.1034 | 0.1076 | 0.4623 | -0.1240 | 0.0625 | 0.7260 | 0.0008 | 0.0002 | 0.0016 | 0.1420 | 52.0000 | 49.0000 |
| text_size | C2 | kfold | pass | PairMeanBaseline | 0.0018 | 0.0392 | 0.0080 | 0.0199 | 0.4072 | 0.0000 | 0.0560 | 0.6964 | -- | -- | -- | -- | -- | -- |
| text_size | C2 | kfold | pass | HistGradientBoosting | 0.0018 | 0.0392 | 0.0080 | 0.0170 | 0.4072 | 0.0000 | 0.0488 | 0.6964 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | 87.0000 | 14.0000 |
| text_size | C2 | kfold | features | PairMeanBaseline | 0.0018 | 0.0392 | 0.0080 | 0.0199 | 0.4072 | 0.0000 | 0.0560 | 0.6964 | -- | -- | -- | -- | -- | -- |
| text_size | C2 | kfold | features | HistGradientBoosting | 0.0023 | 0.0395 | -0.0040 | -0.0183 | 0.3336 | -0.0121 | -- | 0.6810 | 0.0005 | 0.0001 | 0.0009 | 0.3135 | 54.0000 | 47.0000 |
| text_size | C2 | kfold | both | PairMeanBaseline | 0.0018 | 0.0392 | 0.0080 | 0.0199 | 0.4072 | 0.0000 | 0.0560 | 0.6964 | -- | -- | -- | -- | -- | -- |
| text_size | C2 | kfold | both | HistGradientBoosting | 0.0038 | 0.0506 | -0.6516 | 0.0299 | 0.4056 | -0.6649 | 0.0495 | 0.7139 | 0.0019 | 0.0007 | 0.0038 | 0.0000 | 31.0000 | 70.0000 |
