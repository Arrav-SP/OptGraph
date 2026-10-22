Ablation: program representation (residual target)

| metric | context | protocol | reps | model | mae | rmse | r2 | spearman | macro_f1 | residual_r2 | per_program_spearman_median | per_program_ndcg10_mean | mae_diff_vs_pairmean | mae_diff_ci_lo | mae_diff_ci_hi | wilcoxon_p_holm | programs_better | programs_worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text_size | C0 | kfold | hand | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | hand | HistGradientBoosting | 0.0100 | 0.0417 | 0.3173 | 0.3026 | 0.6650 | 0.0399 | 0.3274 | 0.9178 | 0.0004 | -0.0002 | 0.0010 | 0.5681 | 49.0000 | 52.0000 |
| text_size | C0 | kfold | funcprops | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | funcprops | HistGradientBoosting | 0.0099 | 0.0406 | 0.3544 | 0.3010 | 0.6679 | 0.0920 | 0.3152 | 0.9182 | 0.0002 | -0.0003 | 0.0007 | 0.7877 | 48.0000 | 53.0000 |
| text_size | C0 | kfold | opcodes | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | opcodes | HistGradientBoosting | 0.0097 | 0.0392 | 0.3963 | 0.3064 | 0.6628 | 0.1510 | 0.3317 | 0.9233 | 0.0000 | -0.0004 | 0.0005 | 0.9824 | 51.0000 | 50.0000 |
| text_size | C0 | kfold | ir2vec | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | ir2vec | HistGradientBoosting | 0.0099 | 0.0406 | 0.3523 | 0.3006 | 0.6658 | 0.0891 | 0.3286 | 0.9181 | 0.0003 | -0.0002 | 0.0007 | 0.3720 | 48.0000 | 53.0000 |
| text_size | C0 | kfold | hand+funcprops+opcodes+ir2vec | PairMeanBaseline | 0.0096 | 0.0426 | 0.2890 | 0.3514 | 0.7200 | 0.0000 | 0.3778 | 0.9040 | -- | -- | -- | -- | -- | -- |
| text_size | C0 | kfold | hand+funcprops+opcodes+ir2vec | HistGradientBoosting | 0.0097 | 0.0396 | 0.3846 | 0.2993 | 0.6604 | 0.1345 | 0.3169 | 0.9198 | 0.0001 | -0.0004 | 0.0006 | 0.9312 | 52.0000 | 49.0000 |
| dyn_instructions | C0 | kfold | hand | PairMeanBaseline | 0.0077 | 0.0334 | 0.3720 | 0.3097 | 0.6913 | 0.0000 | 0.3544 | 0.8904 | -- | -- | -- | -- | -- | -- |
| dyn_instructions | C0 | kfold | hand | HistGradientBoosting | 0.0070 | 0.0297 | 0.5033 | 0.2331 | 0.6821 | 0.2091 | 0.2604 | 0.8832 | -0.0007 | -0.0015 | 0.0002 | 0.0726 | 46.0000 | 32.0000 |
| dyn_instructions | C0 | kfold | funcprops | PairMeanBaseline | 0.0077 | 0.0334 | 0.3720 | 0.3097 | 0.6913 | 0.0000 | 0.3544 | 0.8904 | -- | -- | -- | -- | -- | -- |
| dyn_instructions | C0 | kfold | funcprops | HistGradientBoosting | 0.0071 | 0.0313 | 0.4480 | 0.2491 | 0.7021 | 0.1211 | 0.2966 | 0.8724 | -0.0006 | -0.0016 | 0.0004 | 0.0687 | 42.0000 | 36.0000 |
| dyn_instructions | C0 | kfold | opcodes | PairMeanBaseline | 0.0077 | 0.0334 | 0.3720 | 0.3097 | 0.6913 | 0.0000 | 0.3544 | 0.8904 | -- | -- | -- | -- | -- | -- |
| dyn_instructions | C0 | kfold | opcodes | HistGradientBoosting | 0.0066 | 0.0273 | 0.5818 | 0.2407 | 0.6746 | 0.3341 | 0.2466 | 0.8874 | -0.0012 | -0.0020 | -0.0002 | 0.0006 | 54.0000 | 24.0000 |
| dyn_instructions | C0 | kfold | ir2vec | PairMeanBaseline | 0.0077 | 0.0334 | 0.3720 | 0.3097 | 0.6913 | 0.0000 | 0.3544 | 0.8904 | -- | -- | -- | -- | -- | -- |
| dyn_instructions | C0 | kfold | ir2vec | HistGradientBoosting | 0.0076 | 0.0312 | 0.4533 | 0.2105 | 0.6699 | 0.1295 | 0.2543 | 0.8896 | -0.0001 | -0.0010 | 0.0008 | 0.7405 | 43.0000 | 35.0000 |
| dyn_instructions | C0 | kfold | hand+funcprops+opcodes+ir2vec | PairMeanBaseline | 0.0077 | 0.0334 | 0.3720 | 0.3097 | 0.6913 | 0.0000 | 0.3544 | 0.8904 | -- | -- | -- | -- | -- | -- |
| dyn_instructions | C0 | kfold | hand+funcprops+opcodes+ir2vec | HistGradientBoosting | 0.0066 | 0.0275 | 0.5743 | 0.2391 | 0.6882 | 0.3222 | 0.2511 | 0.8893 | -0.0011 | -0.0020 | -0.0001 | 0.0007 | 50.0000 | 28.0000 |
| instruction_count | C0 | kfold | hand | PairMeanBaseline | 0.0084 | 0.0364 | 0.5716 | 0.6112 | 0.8454 | 0.0000 | 0.6250 | 0.9585 | -- | -- | -- | -- | -- | -- |
| instruction_count | C0 | kfold | hand | HistGradientBoosting | 0.0077 | 0.0345 | 0.6168 | 0.5632 | 0.8073 | 0.1055 | 0.5876 | 0.9665 | -0.0007 | -0.0012 | -0.0003 | 0.0001 | 73.0000 | 28.0000 |
| instruction_count | C0 | kfold | funcprops | PairMeanBaseline | 0.0084 | 0.0364 | 0.5716 | 0.6112 | 0.8454 | 0.0000 | 0.6250 | 0.9585 | -- | -- | -- | -- | -- | -- |
| instruction_count | C0 | kfold | funcprops | HistGradientBoosting | 0.0077 | 0.0341 | 0.6243 | 0.5648 | 0.8037 | 0.1230 | 0.5836 | 0.9637 | -0.0007 | -0.0012 | -0.0003 | 0.0001 | 66.0000 | 35.0000 |
| instruction_count | C0 | kfold | opcodes | PairMeanBaseline | 0.0084 | 0.0364 | 0.5716 | 0.6112 | 0.8454 | 0.0000 | 0.6250 | 0.9585 | -- | -- | -- | -- | -- | -- |
| instruction_count | C0 | kfold | opcodes | HistGradientBoosting | 0.0075 | 0.0315 | 0.6797 | 0.5612 | 0.8015 | 0.2522 | 0.5860 | 0.9661 | -0.0009 | -0.0013 | -0.0005 | 0.0000 | 74.0000 | 27.0000 |
| instruction_count | C0 | kfold | ir2vec | PairMeanBaseline | 0.0084 | 0.0364 | 0.5716 | 0.6112 | 0.8454 | 0.0000 | 0.6250 | 0.9585 | -- | -- | -- | -- | -- | -- |
| instruction_count | C0 | kfold | ir2vec | HistGradientBoosting | 0.0079 | 0.0335 | 0.6376 | 0.5671 | 0.7955 | 0.1541 | 0.5918 | 0.9668 | -0.0005 | -0.0010 | -0.0001 | 0.0031 | 60.0000 | 41.0000 |
| instruction_count | C0 | kfold | hand+funcprops+opcodes+ir2vec | PairMeanBaseline | 0.0084 | 0.0364 | 0.5716 | 0.6112 | 0.8454 | 0.0000 | 0.6250 | 0.9585 | -- | -- | -- | -- | -- | -- |
| instruction_count | C0 | kfold | hand+funcprops+opcodes+ir2vec | HistGradientBoosting | 0.0072 | 0.0316 | 0.6774 | 0.5735 | 0.8075 | 0.2470 | 0.5984 | 0.9666 | -0.0012 | -0.0016 | -0.0008 | 0.0000 | 78.0000 | 23.0000 |
