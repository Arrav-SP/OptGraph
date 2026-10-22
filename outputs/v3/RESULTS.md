# OptGraph v3: headline results

Generated 2026-10-22T01:59:15+00:00. Machine-readable: `results_v3.json`.

- Programs: 101 kept of 124 ({'testsuite': 47, 'polybench': 30, 'optgraph': 24}); 23 dropped.
- Experiments: 146955 (0 failed), 1367982 interaction values.

## RQ1

For instruction_count, the pair alone explains 9.8% of the variance of I over 101 programs, the terms that involve the context but not the program 15.2%, and all terms involving the program 75.0% (pair x program 26.2%, pair x program x context 48.1%). For text_size, the pair alone explains 5.7% of the variance of I over 101 programs, the terms that involve the context but not the program 11.1%, and all terms involving the program 83.2% (pair x program 27.6%, pair x program x context 54.7%). For dyn_instructions, the pair alone explains 11.5% of the variance of I over 78 programs, the terms that involve the context but not the program 22.9%, and all terms involving the program 65.6% (pair x program 21.6%, pair x program x context 42.2%). Across contexts (dyn_instructions), the per-program rank correlation of the pair interactions has a median between -0.01 and -0.01 depending on the context pair. Across contexts (instruction_count), the per-program rank correlation of the pair interactions has a median between 0.14 and 0.55 depending on the context pair. Across contexts (text_size), the per-program rank correlation of the pair interactions has a median between 0.00 and 0.01 depending on the context pair. Synergy turns into antagonism (or back) between two contexts for 0.6% to 0.9% of all (program, pair) cells for instruction_count; most cells are weak in both. Order matters (dyn_instructions, |I(A,B)-I(B,A)| above the threshold) for 4.3% in C0, 1.1% in C1, 0.4% in C2 of the unordered pairs. Order matters (instruction_count, |I(A,B)-I(B,A)| above the threshold) for 17.1% in C0, 12.7% in C1, 11.5% in C2 of the unordered pairs. Order matters (text_size, |I(A,B)-I(B,A)| above the threshold) for 9.2% in C0, 4.0% in C1, 1.8% in C2 of the unordered pairs.

## RQ2

Under leave-one-program-out, program conditioning helps significantly in 1 of 9 metric/context settings. dyn_instructions, C0: the program-conditioned model is not significantly better than the pair-mean baseline (MAE 0.0071 vs 0.0077; better on 42.0 and worse on 36.0 held-out programs; Holm p = 0.1; residual R² 0.217). dyn_instructions, C1: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0021 vs 0.0016; better on 38.0 and worse on 40.0 held-out programs; Holm p = 0.019; residual R² -0.072). dyn_instructions, C2: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0005 vs 0.0004; better on 24.0 and worse on 54.0 held-out programs; Holm p = < 1e-4; residual R² -0.012). instruction_count, C0: the program-conditioned model beats the pair-mean baseline (MAE 0.0078 vs 0.0084; better on 68.0 and worse on 33.0 held-out programs; Holm p = 0.00012; residual R² 0.120). instruction_count, C1: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0088 vs 0.0086; better on 55.0 and worse on 46.0 held-out programs; Holm p = 0.91; residual R² 0.033). instruction_count, C2: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0090 vs 0.0061; better on 51.0 and worse on 50.0 held-out programs; Holm p = 0.15; residual R² -0.689). text_size, C0: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0099 vs 0.0096; better on 49.0 and worse on 52.0 held-out programs; Holm p = 0.35; residual R² 0.056). text_size, C1: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0061 vs 0.0049; better on 46.0 and worse on 55.0 held-out programs; Holm p = 0.0076; residual R² -0.220). text_size, C2: the program-conditioned model does not beat the pair-mean baseline (MAE 0.0036 vs 0.0018; better on 30.0 and worse on 71.0 held-out programs; Holm p = < 1e-4; residual R² -0.551).

| metric | context | protocol | pairmean_mae | hgb_mae | best_model | best_mae | hgb_residual_r2 | p_holm | better | worse | supported |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dyn_instructions | C0 | kfold | 0.0077 | 0.0070 | HistGradientBoosting | 0.0070 | 0.2091 | 0.0726 | 46.0000 | 32.0000 | False |
| dyn_instructions | C0 | lopo | 0.0077 | 0.0071 | HistGradientBoosting | 0.0071 | 0.2168 | 0.1050 | 42.0000 | 36.0000 | False |
| dyn_instructions | C1 | kfold | 0.0016 | 0.0021 | HistGradientBoosting | 0.0021 | -0.0966 | 0.0258 | 35.0000 | 43.0000 | False |
| dyn_instructions | C1 | lopo | 0.0016 | 0.0021 | HistGradientBoosting | 0.0021 | -0.0721 | 0.0191 | 38.0000 | 40.0000 | False |
| dyn_instructions | C2 | kfold | 0.0004 | 0.0006 | HistGradientBoosting | 0.0006 | -0.0497 | 0.0000 | 23.0000 | 55.0000 | False |
| dyn_instructions | C2 | lopo | 0.0004 | 0.0005 | HistGradientBoosting | 0.0005 | -0.0119 | 0.0000 | 24.0000 | 54.0000 | False |
| instruction_count | C0 | kfold | 0.0084 | 0.0077 | HistGradientBoosting | 0.0077 | 0.1055 | 0.0001 | 73.0000 | 28.0000 | True |
| instruction_count | C0 | lopo | 0.0084 | 0.0078 | HistGradientBoosting | 0.0078 | 0.1200 | 0.0001 | 68.0000 | 33.0000 | True |
| instruction_count | C1 | kfold | 0.0086 | 0.0088 | HistGradientBoosting | 0.0088 | 0.0179 | 0.6197 | 55.0000 | 46.0000 | False |
| instruction_count | C1 | lopo | 0.0086 | 0.0088 | HistGradientBoosting | 0.0088 | 0.0326 | 0.9123 | 55.0000 | 46.0000 | False |
| instruction_count | C2 | kfold | 0.0061 | 0.0083 | LinearRidge | 0.0081 | -0.4498 | 0.1792 | 50.0000 | 51.0000 | False |
| instruction_count | C2 | lopo | 0.0061 | 0.0090 | LinearRidge | 0.0085 | -0.6889 | 0.1457 | 51.0000 | 50.0000 | False |
| text_size | C0 | kfold | 0.0096 | 0.0100 | HistGradientBoosting | 0.0100 | 0.0399 | 0.5681 | 49.0000 | 52.0000 | False |
| text_size | C0 | lopo | 0.0096 | 0.0099 | HistGradientBoosting | 0.0099 | 0.0560 | 0.3542 | 49.0000 | 52.0000 | False |
| text_size | C1 | kfold | 0.0049 | 0.0060 | RandomForest | 0.0056 | -0.1459 | 0.0059 | 43.0000 | 58.0000 | False |
| text_size | C1 | lopo | 0.0049 | 0.0061 | RandomForest | 0.0056 | -0.2203 | 0.0076 | 46.0000 | 55.0000 | False |
| text_size | C2 | kfold | 0.0018 | 0.0037 | RandomForest | 0.0023 | -0.6322 | 0.0000 | 24.0000 | 77.0000 | False |
| text_size | C2 | lopo | 0.0018 | 0.0036 | RandomForest | 0.0023 | -0.5515 | 0.0000 | 30.0000 | 71.0000 | False |

## RQ3

Objective text_size, budget 16: geomean reduction vs -O0 is 20.0% (predicted graph, greedy), 19.4% (program-agnostic graph), 23.3% (measured graph), 9.9% (median random), 25.6% (measured search), against 6.1% for -O2 and -11.5% for -O3. Objective dyn_instructions, budget 16: geomean reduction vs -O0 is 27.7% (predicted graph, greedy), 37.5% (program-agnostic graph), 36.7% (measured graph), 25.6% (median random), 40.1% (measured search), against 40.1% for -O2 and 40.1% for -O3. Predicted vs program-agnostic graph: significantly better in 1 of 12 objective/budget/builder settings (Holm p < 0.05); median geomean ratio 1.0765, wins/ties/losses summed 280/225/431. Predicted graph vs median random sequence: significantly better in 11 of 12 settings.

| objective | k | metric | a | b | programs | gmean_ratio_a_over_b | wins | ties | losses | wilcoxon_p_holm | supported |
|---|---|---|---|---|---|---|---|---|---|---|---|
| text_size | 8 | text_size | graph_predicted_greedy | graph_agnostic_greedy | 78 | 0.9996 | 25 | 39 | 14 | 0.4606 | False |
| text_size | 8 | text_size | graph_predicted_greedy | random_median | 78 | 0.9063 | 72 | 4 | 2 | 0.0000 | True |
| text_size | 8 | text_size | graph_predicted_beam | graph_agnostic_beam | 78 | 0.9870 | 28 | 39 | 11 | 0.0339 | True |
| text_size | 8 | text_size | graph_predicted_beam | random_median | 78 | 0.8937 | 72 | 5 | 1 | 0.0000 | True |
| text_size | 12 | text_size | graph_predicted_greedy | graph_agnostic_greedy | 78 | 1.0120 | 29 | 17 | 32 | 0.8716 | False |
| text_size | 12 | text_size | graph_predicted_greedy | random_median | 78 | 0.9206 | 68 | 4 | 6 | 0.0000 | True |
| text_size | 12 | text_size | graph_predicted_beam | graph_agnostic_beam | 78 | 1.0139 | 25 | 19 | 34 | 0.4606 | False |
| text_size | 12 | text_size | graph_predicted_beam | random_median | 78 | 0.9194 | 67 | 4 | 7 | 0.0000 | True |
| text_size | 16 | text_size | graph_predicted_greedy | graph_agnostic_greedy | 78 | 0.9928 | 28 | 35 | 15 | 0.4606 | False |
| text_size | 16 | text_size | graph_predicted_greedy | random_median | 78 | 0.8886 | 68 | 1 | 9 | 0.0000 | True |
| text_size | 16 | text_size | graph_predicted_beam | graph_agnostic_beam | 78 | 0.9840 | 32 | 33 | 13 | 0.1082 | False |
| text_size | 16 | text_size | graph_predicted_beam | random_median | 78 | 0.8850 | 68 | 1 | 9 | 0.0000 | True |
| dyn_instructions | 8 | dyn_instructions | graph_predicted_greedy | graph_agnostic_greedy | 78 | 1.1738 | 18 | 6 | 54 | 0.0001 | False |
| dyn_instructions | 8 | dyn_instructions | graph_predicted_greedy | random_median | 78 | 0.9471 | 61 | 2 | 15 | 0.0000 | True |
| dyn_instructions | 8 | dyn_instructions | graph_predicted_beam | graph_agnostic_beam | 78 | 1.1600 | 18 | 3 | 57 | 0.0001 | False |
| dyn_instructions | 8 | dyn_instructions | graph_predicted_beam | random_median | 78 | 0.9521 | 60 | 2 | 16 | 0.0000 | True |
| dyn_instructions | 8 | runtime | graph_predicted_greedy | graph_agnostic_greedy | 52 | 1.1588 | 16 | 0 | 36 | 0.0339 | False |
| dyn_instructions | 8 | runtime | graph_predicted_greedy | random_median | 52 | 0.8962 | 38 | 0 | 14 | 0.0043 | True |
| dyn_instructions | 8 | runtime | graph_predicted_beam | graph_agnostic_beam | 52 | 1.1223 | 22 | 0 | 30 | 0.2535 | False |
| dyn_instructions | 8 | runtime | graph_predicted_beam | random_median | 52 | 0.8733 | 41 | 0 | 11 | 0.0001 | True |
| dyn_instructions | 12 | dyn_instructions | graph_predicted_greedy | graph_agnostic_greedy | 78 | 1.1566 | 18 | 8 | 52 | 0.0012 | False |
| dyn_instructions | 12 | dyn_instructions | graph_predicted_greedy | random_median | 78 | 0.9486 | 58 | 0 | 20 | 0.0003 | True |
| dyn_instructions | 12 | dyn_instructions | graph_predicted_beam | graph_agnostic_beam | 78 | 1.1469 | 18 | 9 | 51 | 0.0022 | False |
| dyn_instructions | 12 | dyn_instructions | graph_predicted_beam | random_median | 78 | 0.9568 | 57 | 0 | 21 | 0.0011 | True |
| dyn_instructions | 12 | runtime | graph_predicted_greedy | graph_agnostic_greedy | 52 | 1.1434 | 19 | 0 | 33 | 0.1397 | False |
| dyn_instructions | 12 | runtime | graph_predicted_greedy | random_median | 52 | 0.8828 | 44 | 0 | 8 | 0.0001 | True |
| dyn_instructions | 12 | runtime | graph_predicted_beam | graph_agnostic_beam | 52 | 1.1096 | 20 | 0 | 32 | 0.6265 | False |
| dyn_instructions | 12 | runtime | graph_predicted_beam | random_median | 52 | 0.8661 | 45 | 0 | 7 | 0.0001 | True |
| dyn_instructions | 16 | dyn_instructions | graph_predicted_greedy | graph_agnostic_greedy | 78 | 1.1573 | 20 | 8 | 50 | 0.0051 | False |
| dyn_instructions | 16 | dyn_instructions | graph_predicted_greedy | random_median | 78 | 0.9725 | 49 | 3 | 26 | 0.0339 | True |
| dyn_instructions | 16 | dyn_instructions | graph_predicted_beam | graph_agnostic_beam | 78 | 1.1392 | 21 | 9 | 48 | 0.0536 | False |
| dyn_instructions | 16 | dyn_instructions | graph_predicted_beam | random_median | 78 | 0.9738 | 49 | 2 | 27 | 0.0657 | False |
| dyn_instructions | 16 | runtime | graph_predicted_greedy | graph_agnostic_greedy | 52 | 1.1335 | 18 | 0 | 34 | 0.0637 | False |
| dyn_instructions | 16 | runtime | graph_predicted_greedy | random_median | 52 | 0.9213 | 39 | 0 | 13 | 0.0339 | True |
| dyn_instructions | 16 | runtime | graph_predicted_beam | graph_agnostic_beam | 52 | 1.0969 | 24 | 0 | 28 | 0.8716 | False |
| dyn_instructions | 16 | runtime | graph_predicted_beam | random_median | 52 | 0.8995 | 41 | 0 | 11 | 0.0080 | True |

Rules: supported for a metric/context if the LOPO main model (HistGradientBoosting, residual target) has a lower per-program MAE than the program-agnostic pair mean, Wilcoxon signed-rank over programs with Holm-corrected p < 0.05 (Holm over the learned models of that configuration).

supported for an objective/budget/builder if sequences from the predicted program-conditioned graph give a lower objective value than sequences from the program-agnostic graph, Wilcoxon on per-program log ratios with Holm-corrected p < 0.05 (Holm over all key comparisons).
