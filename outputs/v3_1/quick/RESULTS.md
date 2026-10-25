# OptGraph v3.1 results

Generated 2026-10-22T09:20:47+00:00. Objective: IR instruction count / -Oz.

## Pre-registered hypotheses (Holm across the four)

| hypothesis | claim | programs | effect | p (one-sided) | p (Holm) | supported |
|---|---|---|---|---|---|---|
| H1 | Claim 1 headroom: oracle graph vs static graph, E = 10 | 3 | 1.0000 | 1 | 1 | False |
| H2 | Claim 2 drift: rho(S0, S2) < 0.5 | 1 | 0.4289 | n/a | 1 | False |
| H3 | Claim 3: LOPO model MAE < static-graph MAE | 3 | -0.1208 | 1 | 1 | False |
| H4 | Claim 4: re-prediction vs strongest other method, E = 10 | 3 | 1.0021 | 1 | 1 | False |

H4 compared against the strongest other method at E = 10: **predict_once**.

Everything else in the report is exploratory.
