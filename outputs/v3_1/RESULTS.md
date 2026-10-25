# OptGraph v3.1 results

Generated 2026-10-25T06:47:11+00:00. Objective: IR instruction count / -Oz.

## Pre-registered hypotheses (Holm across the four)

| hypothesis | claim | programs | effect | p (one-sided) | p (Holm) | supported |
|---|---|---|---|---|---|---|
| H1 | Claim 1 headroom: oracle graph vs static graph, E = 10 | 129 | 0.9486 | 1.3e-09 | 2.59e-09 | True |
| H2 | Claim 2 drift: rho(S0, S2) < 0.5 | 123 | 0.2922 | 2.68e-21 | 8.03e-21 | True |
| H3 | Claim 3: LOPO model MAE < static-graph MAE | 129 | 0.5042 | 1.02e-21 | 4.1e-21 | True |
| H4 | Claim 4: re-prediction vs strongest other method, E = 10 | 129 | 1.1067 | 1 | 1 | False |

H4 compared against the strongest other method at E = 10: **ga**.

Everything else in the report is exploratory.
