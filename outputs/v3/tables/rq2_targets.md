Direct target (compare with residual in rq2_models)

| metric | context | protocol | model | mae | rmse | r2 | spearman | macro_f1 | residual_r2 | per_program_spearman_median | per_program_ndcg10_mean | mae_diff_vs_pairmean | mae_diff_ci_lo | mae_diff_ci_hi | wilcoxon_p_holm | programs_better | programs_worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text_size | C0 | kfold | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | RandomForest | 0.0097 | 0.0403 | 0.3629 | 0.3023 | 0.6293 | 0.1040 | 0.3341 | 0.9204 | 0.0001 | -0.0003 | 0.0005 | 1.0000 | 48.0000 | 53.0000 |
| text_size | C0 | kfold | HistGradientBoosting | 0.0096 | 0.0416 | 0.3225 | 0.2986 | 0.6703 | 0.0472 | 0.3114 | 0.9140 | -0.0000 | -0.0005 | 0.0004 | 1.0000 | 51.0000 | 50.0000 |
