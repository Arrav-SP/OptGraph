# OptGraph v3.1 results

Generated 2026-10-22T11:24:59+00:00. Objective: IR instruction count / -Oz.

## Pre-registered hypotheses (Holm across the four)

| hypothesis | claim | programs | effect | p (one-sided) | p (Holm) | supported |
|---|---|---|---|---|---|---|
| H1 | Claim 1 headroom: oracle graph vs static graph, E = 10 | 5 | 0.9913 | 0.438 | 1 | False |
| H2 | Claim 2 drift: rho(S0, S2) < 0.5 | 5 | 0.2606 | 0.0312 | 0.125 | False |
| H3 | Claim 3: LOPO model MAE < static-graph MAE | 5 | -0.0726 | 0.688 | 1 | False |
| H4 | Claim 4: re-prediction vs strongest other method, E = 10 | 5 | 1.0723 | 0.781 | 1 | False |

H4 compared against the strongest other method at E = 10: **pan_style**.

Everything else in the report is exploratory.
