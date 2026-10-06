# Viva Questions — OptGraph

Numbers are from the full run (24 programs, 12 passes, LLVM 21.1.8).

**1. What is OptGraph?**
A framework that measures how pairs of LLVM passes interact on each program, stores the result as a directed weighted graph per program, and tries to predict that graph for unseen programs.

**2. What is the research question?**
How much do pass interactions depend on the structure of the input program, and can they be predicted for programs not seen before?

**3. Why LLVM?**
It is modular: each optimization is a separate pass that `opt` can run alone and in any order on a well-defined IR. It is also open and widely used in research.

**4. What is LLVM IR?**
LLVM's typed, RISC-like intermediate representation in SSA form. Front ends produce it, passes transform it, back ends turn it into machine code.

**5. What is an optimization pass?**
A unit that analyses or transforms the IR while preserving the program's behaviour, for example removing dead instructions.

**6. Give examples of passes.**
mem2reg, sroa, instcombine, simplifycfg, early-cse, gvn, dce, adce, reassociate, licm, loop-rotate, loop-unroll. These are the 12 we use.

**7. What is pass interaction?**
The effect of pass B changes depending on whether pass A ran first. The pair's combined effect is not the sum of the individual effects.

**8. What is synergy?**
I > 0: B achieves more after A than alone. Example: licm→instcombine is synergistic in all 24 programs (mean +0.21).

**9. What is antagonism?**
I < 0: B achieves less after A than alone, because A consumed or blocked B's opportunities. Example: mem2reg and sroa (−0.42): both promote the same allocas.

**10. What is Δ?**
The relative improvement of a metric: (before − after) / baseline. Positive means the count went down.

**11. Explain the interaction formula.**
I(A,B,P) = Δ(B|A,P) − Δ(B|P). First term: what B gains when run after A. Second: what B gains alone. Both are divided by the original program's metric so they are comparable.

**12. Why A→B?**
To measure B's benefit on the program already transformed by A, which is the first term of the formula.

**13. Why B→A?**
Because order matters. I(B,A,P) is a different quantity. In our data the order changes the result in 34% of cases.

**14. Why is the graph directed?**
I(A,B) ≠ I(B,A) in general. Their difference equals the difference in final size between the two orders, divided by the baseline.

**15. What are the graph nodes?**
Optimization passes.

**16. What are the graph edges?**
Edge A→B has weight I(A,B,P). Positive is synergy, negative antagonism, near zero weak (not drawn).

**17. Why program-conditioned?**
The same pair gives different scores on different programs. 43% of the score variance is between programs for a fixed pair, and 27 of 132 pairs change sign across programs. So there is one graph per program.

**18. What program features are extracted?**
30: counts of instructions, blocks, functions, branches, loads, stores, calls, PHIs, integer/float/arithmetic/memory ops, returns, CFG edges, loops, loop depth, allocas, GEPs, compares, casts; plus ratios, average block size and cyclomatic complexity.

**19. Why these features?**
They are cheap, interpretable, and describe what passes act on: memory traffic (mem2reg, gvn), control flow (simplifycfg), loops (licm, unroll), arithmetic (instcombine, reassociate).

**20. What is SSA?**
Static Single Assignment: every variable is assigned exactly once; PHI nodes merge values at control-flow joins. It makes def-use relations explicit.

**21. What is a basic block?**
A straight-line sequence of instructions with one entry and one exit, ending in a terminator such as a branch or return.

**22. What is CFG?**
Control Flow Graph: basic blocks as nodes, possible jumps as edges.

**23. What does GVN do?**
Global Value Numbering: finds computations that produce the same value and removes the redundant ones, including redundant loads.

**24. What does DCE do?**
Dead Code Elimination: removes instructions whose results are never used and that have no side effects.

**25. What does LICM do?**
Loop-Invariant Code Motion: moves computations that give the same result on every iteration out of the loop.

**26. What is the ML input?**
Features of the original, unoptimized IR plus one-hot encodings of pass A, pass B and the ordered pair.

**27. What is the ML output?**
The predicted interaction score I(A,B,P) for instruction count. Classes (synergy/weak/antagonism) are derived by thresholding.

**28. Why Random Forest?**
It handles non-linear effects and mixed one-hot and numeric inputs, needs little tuning, works on small data and gives feature importances. It had the lowest validation MAE.

**29. Why not deep learning?**
Only 24 programs and 3,168 rows. A neural network would overfit and be harder to interpret.

**30. How do you prevent data leakage?**
Features come only from the original IR, never from optimized IR. Splits are by program, so no program is in both training and test.

**31. Why split by program?**
One program produces 132 rows. A random row split would put the same program on both sides and the model could memorise it, inflating the scores.

**32. What are your baselines?**
Mean predictor; pair-mean predictor (average score of that pair in training, no program information); ridge regression. The pair-mean baseline is the important one.

**33. How do you evaluate?**
MAE, RMSE, R², Pearson, Spearman; accuracy and macro precision/recall/F1 for the three classes. Two protocols: 16/4/4 program hold-out and leave-one-program-out.

**34. What if the ML model performs poorly?**
That is still an answer. Our result is modest: random forest MAE 0.0211 vs 0.0243 for the pair-mean baseline in leave-one-program-out, but the baseline wins on the 4-program hold-out. Program features help a little, not decisively.

**35. What is novel?**
Not pass interaction or phase ordering themselves. We quantify how much pairwise interactions vary with program structure and test whether they are predictable for unseen programs.

**36. What are the limitations?**
Small hand-written benchmarks; only static IR counts; a smaller count is treated as better even for loop-unroll; negative scores mix interference with redundancy; `-O0` baseline makes memory promotion dominant.

**37. What is future scope?**
Standard suites (PolyBench, cBench), runtime and code size, richer features, using predicted graphs to choose pass orders, three-pass interactions.

**38. Why not simply use -O3?**
-O3 is one fixed pass order for all programs. We study why and when pass effects depend on order and on the program, which -O3 does not reveal.

**39. Why not test every possible pass sequence?**
The number of sequences grows factorially. Pairs are the smallest unit that shows interaction: 132 ordered pairs instead of 12! orderings.

**40. What is the difference between phase ordering and your work?**
Phase ordering searches for a good sequence. We measure and model the pairwise interactions that cause the phase-ordering problem, per program. A predicted graph could later guide ordering.

## Extra questions likely for this implementation

**41. Why does -O0 IR need `-disable-O0-optnone`?**
Clang marks -O0 functions `optnone`, and every pass skips such functions. Without the flag no pass would change anything.

**42. What happens when the baseline metric is zero?**
The normalised score is undefined. We keep the absolute change, label the row UNDEFINED and exclude it from modelling. It only happens for PHI count.

**43. How do you know the optimized programs are still correct?**
Each optimized IR is linked, run, and its output compared to the baseline's. All 3,480 matched.

**44. Did the hypothesis hold?**
Weakly. Structural and interaction similarity correlate at ρ = 0.114 (permutation p = 0.048) over 276 program pairs. Right direction, small effect, small sample.

**45. How much of the interaction is explained by the pass pair alone?**
56.7% of the variance. The remaining 43.3% differs between programs.
