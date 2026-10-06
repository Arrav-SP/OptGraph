# OptGraph Study Guide (from zero)

Read it top to bottom once, then re-read parts 4, 6 and 9. Every number is real
output from this project.

---

## Part 1. The one-paragraph version

A compiler turns your C code into machine code. In the middle it runs many small
cleanup steps called **optimization passes**. Each pass shrinks or speeds up the
code. The catch: **the order matters**. Running pass A then pass B can give a
different result than B then A. This project **measures** that effect for every
pair of 12 passes on 24 small programs, draws it as a **graph**, and then trains
a **machine learning model** to see whether the effect can be **predicted** from
the program's structure, for programs the model has never seen.

---

## Part 2. Background from absolute zero

### 2.1 Compiler
A program that translates source code (C) into something the CPU runs.

### 2.2 LLVM and Clang
- **LLVM**: a toolkit for building compilers.
- **Clang**: the C compiler built on LLVM.
- **opt**: the LLVM tool that runs optimization passes on a file.

### 2.3 LLVM IR (Intermediate Representation)
Clang does not go straight from C to machine code. It first makes **IR**, a
simple assembly-like language that is the same for every CPU. It is stored in
`.ll` text files. Example, `int y = x + 0;` looks like:

```
%4 = load i32, ptr %2      ; read x from memory
%5 = add nsw i32 %4, 0     ; x + 0
store i32 %5, ptr %3       ; write y to memory
```

Passes work on IR. We count things in IR (instructions, loads...) to measure
how good a pass was.

### 2.4 Key IR vocabulary
- **Instruction**: one line of IR (add, load, store, br, call...).
- **Basic block**: a straight run of instructions with no jumps in the middle.
  It starts at a label and ends with a jump or return.
- **CFG (control flow graph)**: blocks as dots, jumps between them as arrows.
- **SSA (static single assignment)**: every variable is assigned exactly once.
  If a variable has different values on different paths, a **PHI** instruction
  picks the right one. SSA makes optimizations easy to write.
- **alloca / load / store**: a variable that lives in memory (alloca =
  reserve space, store = write, load = read).

### 2.5 -O0 and why we use it
`-O0` means "no optimization". The IR is then a very literal translation: every
variable is in memory (lots of alloca/load/store). That is a **clean baseline**
to measure passes against. Clang also tags -O0 functions `optnone`
("don't optimize me"), so we pass `-Xclang -disable-O0-optnone`; otherwise
`opt` would skip everything and nothing would change.

### 2.6 The 12 passes (what each does, simply)
| Pass | What it does |
|---|---|
| **mem2reg** | Moves variables from memory into registers (SSA values). Removes alloca/load/store. Biggest effect at -O0. |
| **sroa** | Splits structs/arrays into separate scalars, then does mem2reg-like promotion. Overlaps with mem2reg. |
| **instcombine** | Local algebra: `x+0 → x`, `x*4 → x<<2`, merges instructions. |
| **simplifycfg** | Cleans control flow: merges blocks, removes unreachable ones, folds constant branches. |
| **early-cse** | Quick removal of repeated identical computations. |
| **gvn** | Global value numbering: finds values computed twice and reuses them, also removes redundant loads. |
| **dce** | Deletes instructions whose result is never used. |
| **adce** | Aggressive version of dce. |
| **reassociate** | Reorders `(a+5)+(b+7)` so constants group together and fold. |
| **licm** | Loop-invariant code motion: moves work that doesn't change per iteration out of the loop. |
| **loop-rotate** | Reshapes a loop (while to do-while form) so other loop passes work. |
| **loop-unroll** | Copies a loop body several times. Makes code **bigger** on purpose. |

(dce and adce changed almost nothing on our programs, since at this point
nothing is dead; they show up as isolated nodes in the graph.)

---

## Part 3. The research idea

**Pass interaction**: how much A changes what B can do.

**Research question (say it exactly):** To what extent do interactions between
compiler optimization passes depend on the structural characteristics of the
input program, and can these program-conditioned interactions be learned and
accurately predicted for previously unseen programs?

**Hypothesis:** programs with similar structure have similar interactions.

**What is NOT new** (never claim it): pass interaction, phase ordering,
optimization graphs, ML-guided compilation. **Our angle:** we check whether the
interactions *vary systematically with program structure* and whether that is
*predictable for unseen programs*.

**Phase ordering** = the problem of finding the best order of passes. Our work
studies the pairwise interactions that cause that problem.

---

## Part 4. The experiment and the formula (learn this perfectly)

For one program **P** and one ordered pair of passes **(A, B)** we make 5 IR files:

1. **P** (baseline)
2. **P → A** (A alone)
3. **P → B** (B alone)
4. **P → A → B** (B run on A's output)
5. **P → B → A** (A run on B's output)

and count instructions in each. Call the counts m(P), m(A), m(B), m(AB).

**Improvement (Δ):** (before − after) / baseline. Positive = got smaller.

- **Δ(B | P)** = (m(P) − m(B)) / m(P). What B achieves on the raw program.
- **Δ(B | A,P)** = (m(A) − m(AB)) / m(P). What B *adds* after A already ran.
- **Interaction I(A,B,P) = Δ(B | A,P) − Δ(B | P).**

Both are divided by m(P) (the original) so they are on the same scale.

- I > +0.005 → **SYNERGY** (A makes B better)
- I < −0.005 → **ANTAGONISM** (A makes B worse; often because A already did B's job)
- else → **WEAK**

### Worked example (from the demo, loop_04_invariant, A=gvn, B=instcombine)
m(P)=91, m(gvn)=62, m(instcombine)=75, m(gvn→instcombine)=34.

- Δ(B|P) = (91−75)/91 = **17.58%**
- Δ(B|A,P) = (62−34)/91 = **30.77%**
- I = 30.77 − 17.58 = **+13.19% → synergy**

Reverse (A=instcombine, B=gvn): m(instcombine)=75, m(gvn)=62, m(inst→gvn)=55.
- Δ(gvn|P) = (91−62)/91 = 31.87%
- Δ(gvn|instcombine,P) = (75−55)/91 = 21.98%
- I = **−9.89% → antagonism**

Final sizes 34 vs 55: **order matters.** That is why I(A,B) ≠ I(B,A) and why
the graph is **directed**. (Fact: I(A,B) − I(B,A) = (m(BA) − m(AB)) / m(P).)

**Zero baseline:** if m(P)=0 you can't divide. This happens for PHI count (0
at -O0). Those rows are labelled UNDEFINED and kept out of the ML. Never silently NaN.

**Metrics measured** (each analysed separately): instruction_count,
basic_blocks, branches, conditional_branches, loads, stores, phis, calls.
The ML uses **instruction_count**.

---

## Part 5. The graph

- **Nodes** = the 12 passes. **Edge A→B** = weight I(A,B,P).
- Green = synergy, red = antagonism, thickness ∝ |I|. Weak edges not drawn.
- One graph per program (24) + one **aggregate** (average over programs).
- Stats per pass: weighted in/out-degree (sum of incoming/outgoing weights),
  positive/negative degree (counts), betweenness (how often a node lies on
  shortest paths).
- Out-degree high = "good enabler" (e.g. loop-rotate +0.21, licm +0.16).
  Very negative = mem2reg, sroa (they "use up" opportunities).

---

## Part 6. The machine learning

**Task:** predict I(A,B,P) as a number.
**Input:** 30 structural features of the ORIGINAL IR + which pass is A, which is
B (one-hot encoding = a 0/1 column per pass) + the ordered pair as one-hot.
**Output:** a number (the predicted interaction).

**Features (30):** counts (instructions, blocks, functions, branches, loads,
stores, calls, PHIs, int/float/arith/memory ops, returns, CFG edges, loop
count, loop depth, allocas, GEPs, compares, casts) + ratios (load fraction,
etc., average block size, cyclomatic complexity = edges − nodes + 2).

**Data leakage (critical):** one program produces 132 rows (12×11 pairs). If
some rows of a program are in training and others in testing, the model has
effectively seen the test program and the score is fake. **So we split by
program.** Test programs are completely unseen.

Two protocols:
1. **Hold-out:** 16 train / 4 validation / 4 test programs.
   Test = mem_01_array_copy, mem_02_array_init, mem_04_struct_fields, mix_01_matrix_mul.
2. **Leave-one-program-out (LOPO):** 24 rounds; each program is the test once.
   More stable with few programs.

**Models:**
- **Mean baseline:** always guess the average. The "dumb" floor.
- **Pair-mean baseline:** guess the average score of that exact (A,B) pair seen
  in training. Knows nothing about the program. **The most important
  baseline**: if the real model can't beat it, program features don't help.
- **Ridge (linear regression):** a weighted sum. Simple, interpretable.
- **Random forest:** many decision trees averaged. Handles non-linear stuff, little tuning.
- **Gradient boosting:** trees built one after another, each fixing the previous errors.
- No deep learning: 24 programs is far too little.

**Metrics explained:**
- **MAE** = average |predicted − actual|. Lower better.
- **RMSE** = like MAE but punishes big errors more.
- **R²** = fraction of variance explained. 1 perfect, 0 = as good as guessing
  the mean, negative = worse than the mean.
- **Pearson/Spearman** = correlation (linear / rank). Higher better.
- **Classification:** turn scores into synergy/weak/antagonism and compute
  **accuracy**, and **precision/recall/F1** (macro = average over the 3
  classes). Accuracy alone can mislead when classes are unbalanced (61% weak).
- **Confusion matrix:** table of actual vs predicted class.

**Ablation:** same model with different inputs: A = pass identities only,
B = program features only, C = both. Tests whether program features add anything.

### Results (memorise)
LOPO (3,168 predictions):
| Model | MAE | R² | Accuracy | F1 |
|---|---|---|---|---|
| Mean | 0.0551 | −0.001 | 0.243 | 0.130 |
| Pair-mean | 0.0243 | 0.529 | 0.882 | 0.850 |
| Ridge | 0.0367 | 0.492 | 0.474 | 0.477 |
| Gradient boosting | 0.0229 | 0.566 | 0.740 | 0.713 |
| **Random forest** | **0.0211** | **0.567** | **0.905** | **0.876** |

Hold-out test (4 programs): pair-mean MAE 0.0239 / R² 0.508; random forest
MAE 0.0250 / R² 0.194 (but accuracy 0.913 vs 0.875).

Ablation: passes only MAE 0.0243; features only 0.0565 (useless alone); both 0.0211.

**Honest conclusion (say this, don't oversell):** the *pair identity* explains
most of the predictability (56.7% of variance). Program features give a small
gain (13% lower MAE in LOPO, better than the baseline on 20 of 24 programs) but
on the small 4-program hold-out the baseline wins. So: promising, not conclusive.

### Descriptive findings
- 3,168 observations: 61% weak, 24% antagonism, 15% synergy.
- 27 of 132 ordered pairs flip between synergy and antagonism across programs
  → proof that interactions depend on the program.
- Order matters in 34% of (program, pair) cases.
- Strongest: licm→instcombine +0.21 (synergy in all 24), mem2reg↔sroa −0.42
  (both do the same job), gvn→loop-unroll −0.39.

### Similarity analysis (tests the hypothesis)
Each program has a structure vector and an "interaction vector" (its 132
scores). Compare similarity (cosine) between every program pair: Spearman
ρ = 0.114, permutation (Mantel) p = 0.048. Right direction, weak. Not a proof.

---

## Part 7. File-by-file

### Root
- **run_demo.py** — the live demo. Checks environment, picks a program, runs P/A/B/AB/BA, prints features and interaction, draws the graph, does an ML prediction. Uses only `experiments/demo` and `outputs/demo`.
  Options: `--list` (show programs and passes), `--program ID`, `--pass-a X --pass-b Y`,
  `--file path/to/any.c` (run on a program that is not in the benchmark suite, e.g. one the
  professor writes; the ML step is then a true unseen-program prediction), `--metric loads`.
- **examples/custom_example.c** — a template for `--file`.
- **README.md** — setup and commands. **requirements.txt** — Python packages. **environment.json** — versions recorded by the environment check. **.gitignore** — what git should skip.

### config/
- **passes.yaml** — the 12 passes and their exact `opt` pipeline strings (e.g. licm = `loop-mssa(licm)`), settings (seed 42, threshold 0.005, primary metric), and the "quick" subset. Change passes here, no code edits.

### benchmarks/ (24 C programs, 5 folders)
arithmetic (5), loops (5), memory (4), control_flow (5), mixed (5). Each is small, deterministic, no input, no undefined behavior (checked with sanitizers). Program ID = file name without `.c`.

### src/ (the library)
- **utils.py** — paths, loads the YAML, finds benchmarks, `run_cmd` (runs a program and captures output/failure without crashing), `Workspace` (where each run keeps files).
- **llvm_pipeline.py** — the ONLY place that builds clang/opt commands: C→IR, verify IR, run a pass, link and run an executable, environment info, smoke test. If LLVM syntax changes, edit here.
- **metrics.py** — parses the text IR and counts instructions/blocks/branches/loads/stores/PHIs/calls; `improvement()` formula (returns None if baseline is 0).
- **features.py** — the 30 features from the baseline IR. Loop count/depth come from LLVM's own `print<loops>`.
- **experiments.py** — the core runner: for one program builds baseline, every single pass, every ordered pair; logs status/command/stdout/stderr/time for each; one failure never stops the rest; can check outputs still equal the original.
- **interaction.py** — the formula: turns measurements into Δ values, interaction score and SYNERGY/ANTAGONISM/WEAK/UNDEFINED; joins in program features (`f_` columns).
- **graph.py** — builds per-program and aggregate NetworkX graphs, degree/betweenness stats, draws pictures.
- **model.py** — one-hot design matrix, the models, the baselines, the by-program split, leave-one-program-out.
- **evaluation.py** — MAE/RMSE/R²/correlations, classification metrics, all plots, similarity analysis.

### scripts/ (command-line tools)
- **check_environment.py** — prints versions, tests C→IR→pass, validates all 12 passes, writes environment.json.
- **run_experiments.py** — `--quick` or `--full`; runs everything, writes `data/` CSVs.
- **extract_features.py** — writes only the features CSV.
- **build_graph.py** — makes the graphs and stats.
- **train_model.py** — trains/evaluates everything, makes the plots, saves the model.
- **analyze_similarity.py** — hypothesis test and plot.
- **summarize_results.py** — descriptive stats (the 61/24/15%, 34%, etc.).
- **run_all.sh** — runs all of the above. **validate_clean.sh** — wipes quick/demo and re-runs.
- **_bootstrap.py** — tiny helper so scripts can `import src`.

### data/
- **program_features.csv** — 24 rows, one per program.
- **interaction_dataset.csv** — 25,344 rows (program × ordered pair × metric).
- **results/measurements.csv, experiment_log.csv, run_config.json** — raw measurements, full log of every command, run settings.

### experiments/full/<program>/
`baseline.ll`, `gvn.ll`, `gvn_then_instcombine.ll`, ... plus `metadata.json`. Open two and compare to SEE what a pass does.

### outputs/
- **graphs/** 25 PNGs. **plots/** 9 PNGs (predicted vs actual, residuals, distribution, model comparison, feature importance, confusion matrix, ablation, similarity). **models/** saved model. **reports/** all numbers as CSV/JSON.
- **review2_summary.md, research_summary.md, viva_questions.md** — the documents.

---

## Part 8. How the data flows (trace one row)

1. `benchmarks/loops/loop_04_invariant.c` → clang → `baseline.ll` (91 instructions).
2. `features.py` reads baseline.ll → 91 instr, 10 blocks, 2 loops...
3. `experiments.py` runs opt for gvn, instcombine, gvn→instcombine, instcombine→gvn.
4. `metrics.py` counts instructions in each: 62, 75, 34, 55.
5. `interaction.py` computes +13.19% and −9.89%, appends the features.
6. That row goes in `interaction_dataset.csv`.
7. `graph.py` uses it as an edge; `model.py` as a training example.

---

## Part 9. Weak spots (admit them before they ask)

- 24 small hand-written programs; only 4 test programs; 3 of them are memory programs.
- Only static IR counts, no runtime or binary size.
- "Smaller = better" is wrong for loop-unroll, so its scores (to −2.01) mean "more unrolled".
- Negative can mean redundancy (mem2reg/sroa), not harmful conflict.
- -O0 baseline makes memory promotion dominate.
- I changed model settings once after seeing results (added pair one-hot, deeper boosting). Disclosed.
- Hypothesis result is weak (ρ=0.114).

---

## Part 10. Tonight's plan

1. Part 2 + 4 (30 min): redo the worked example by hand without looking.
2. Run the demo twice, read each section aloud (10 min).
3. Open `experiments/full/loop_04_invariant/instcombine.ll` and `baseline.ll` side by side (10 min).
4. Look at `aggregate_interaction_graph.png` and explain 3 edges (10 min).
5. Part 6 results table + honest conclusion (20 min).
6. Skim `viva_questions.md` and answer aloud (30 min).
7. Open each src file just far enough to say what it does (30 min).
