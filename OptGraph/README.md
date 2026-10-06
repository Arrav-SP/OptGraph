# OptGraph

**Program-Conditioned Quantification and Prediction of LLVM Optimization Pass Interactions**

OptGraph measures how pairs of LLVM optimization passes interact on a given
program, represents those interactions as a directed weighted graph per
program, and tests whether the interaction can be predicted for programs the
model has never seen, from the structure of the unoptimized IR alone.

**Research question.** To what extent do interactions between compiler
optimization passes depend on the structural characteristics of the input
program, and can these program-conditioned interactions be learned and
accurately predicted for previously unseen programs?

Pass interaction, phase ordering, optimization graphs and ML-guided
compilation are all established topics. What this project examines is whether
the *magnitude and direction* of pairwise interactions vary systematically
with program structure, and whether that variation is predictable.

## Pipeline

```
C program ──clang -O0──> LLVM IR (P) ──┬──> structural features (original IR only)
                                       │
                                       └──> opt:  P→A   P→B   P→A→B   P→B→A
                                                    │
                                              IR measurements
                                                    │
                              I(A,B,P) = Δ(B | A,P) − Δ(B | P)
                                                    │
                    ┌───────────────────────────────┼───────────────────────────┐
              per-program graph             interaction dataset          similarity analysis
                                                    │
                                 ML: features + pass pair → I   (split by program)
```

## Setup (WSL / Ubuntu)

All commands are run inside WSL from the `OptGraph/` directory.

```bash
sudo apt update && sudo apt install -y clang llvm libc6-dev python3-venv python3-pip
python3 -m venv ~/.venvs/optgraph
source ~/.venvs/optgraph/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
```

The virtual environment is kept in the WSL home directory on purpose: this
repository sits under `/mnt/c/.../OneDrive/`, where a venv would be slow and
would be synced to the cloud. `python3 -m venv .venv` also works if you move the
repository into the Linux filesystem (e.g. `~/OptGraph`), which is also the
thing to do if the experiments ever feel slow.

Developed with Ubuntu 26.04 (WSL2), Clang/LLVM 21.1.8, Python 3.14.
`environment.json` records the exact versions of the last run.

## Running

```bash
python run_demo.py                                   # ~10 s end-to-end demo, one program, one pair
python run_demo.py --program mix_01_matrix_mul --pass-a licm --pass-b instcombine

python scripts/run_experiments.py --quick            # 6 programs x 5 passes  -> data/quick/
python scripts/run_experiments.py --full --verify-outputs   # 24 programs x 12 passes (~95 s)
python scripts/build_graph.py                        # outputs/graphs/
python scripts/train_model.py                        # outputs/plots/, outputs/models/, outputs/reports/
python scripts/analyze_similarity.py                 # hypothesis check
python scripts/summarize_results.py                  # descriptive statistics
bash scripts/run_all.sh                              # all of the above, in order
```

`run_experiments.py` also accepts `--programs`, `--program`, `--passes`,
`--pass-a/--pass-b`, `--tag` and `--workers`. `--verify-outputs` links and runs
every optimized IR and fails the experiment if the program's output changes.

## Definitions

**Metrics** (counts over function bodies of the textual IR): instruction count,
basic blocks, branches, conditional branches, loads, stores, PHIs, calls. They
are analysed separately; they are never merged into one score. Runtime is not
used: the benchmarks finish in microseconds and timing would be noise.

**Improvement.** `(baseline − optimized) / baseline`; positive = the count went down.

**Interaction.** For metric *m*:

```
Δ(B | P)    = ( m(P)    − m(B(P))    ) / m(P)
Δ(B | A, P) = ( m(A(P)) − m(B(A(P))) ) / m(P)
I(A, B, P)  = Δ(B | A, P) − Δ(B | P)
```

Both terms are fractions of the *original* program's metric, so `I` is the
pair's departure from additivity. `I > +0.005` synergy, `I < −0.005`
antagonism, otherwise weak (threshold in `config/passes.yaml`).
`I(A,B,P) − I(B,A,P) = (m(A(B(P))) − m(B(A(P)))) / m(P)`: the asymmetry of the
graph is exactly the phase-ordering effect of the pair.

**Zero baseline.** If `m(P) = 0` (the PHI count of `-O0` IR is always 0) the
normalised score is undefined. The row keeps the absolute interaction
(`interaction_abs`), is labelled `UNDEFINED`, and is excluded from modelling.

**Leakage control.** Model inputs are the `f_*` columns (features of the
original IR) and the pass identities. Train/validation/test are split **by
program**; leave-one-program-out cross-validation is reported as well.

## Layout

```
config/passes.yaml          passes (exact opt pipeline strings), thresholds, quick subset
benchmarks/<category>/      24 deterministic C programs in 5 categories
src/llvm_pipeline.py        the only place that builds clang/opt command lines
src/metrics.py              IR text parser and metric definitions
src/features.py             structural features of the original IR
src/experiments.py          P, P→A, P→A→B runner with per-experiment logging
src/interaction.py          interaction score and dataset construction
src/graph.py                per-program and aggregate graphs
src/model.py                models, program-level splits, leave-one-program-out
src/evaluation.py           metrics, plots, similarity analysis
scripts/                    command-line entry points
run_demo.py                 end-to-end demonstration
data/                       program_features.csv, interaction_dataset.csv, results/
experiments/<tag>/<id>/     every IR file plus metadata.json
outputs/                    graphs/, plots/, models/, reports/, *.md summaries
```

## Results and documents

- `outputs/review2_summary.md` – status and results for Review 2
- `outputs/research_summary.md` – full method, results, limitations
- `outputs/viva_questions.md` – 40 questions with short answers
- `outputs/reports/` – every number quoted in the documents, as CSV/JSON
