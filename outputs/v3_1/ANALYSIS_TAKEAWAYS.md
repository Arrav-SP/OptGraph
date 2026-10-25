## Takeaways for the paper (written 25 Oct 2026 from the numbers below)

1. **The sequence builder, not the model, limits re-prediction.**
   - Graph-driven builders barely improve with more budget: from E = 1 to E = 200 the median gain is 0.0% for repredict, 0.4% for predict_once, 1.1% for oracle_graph and 2.6% for static_graph. Search methods gain 14-35%.
   - Even the oracle graph, which uses the program's own measured interactions, is not better than GA at E = 10: ratio 1.052, p = 0.26.
   - The model is not the weak part. It ranks single-pass effects better than the static graph at every snapshot (Spearman 0.90 vs 0.83 at S0, 0.77 vs 0.71 at S2-S4). Yet repredict equals static_graph at E = 10 (0.998, p = 0.35).
   - The information is there; a pairwise graph builder cannot turn it into sequences.
2. **At one compile, graph guidance wins** (exploratory; pre-registered Claim 4 at E = 10 stays unsupported).
   - At E = 1, predict_once beats the strongest search method (pan_style) on 89 of 129 programs: ratio 0.911, p = 3.6e-8.
   - repredict also wins at E = 1: 0.938, p = 0.015. The oracle graph wins with 0.891, p = 3.5e-10.
   - The advantage is gone by E = 5. This gives a niche: single-shot, no-search compilation.
3. **Claim 1's headroom carries to code size on two ISAs.**
   - Oracle vs static graph at E = 10: x86-64 .text 0.950 (p = 4.1e-6), AArch64 .text 0.930 (p = 1.5e-7). At E = 200, AArch64 .text is 0.910 (p = 2.5e-12).
   - For executed instructions (0.921, CI 0.78-1.01, p = 0.13) and runtime (0.921, CI 0.81-1.00, p = 0.032, 89 programs) the direction holds but the evidence is weaker.
   - Claim it for size; call runtime suggestive.
4. **Caveat on the objective.**
   - IR instruction count is a size proxy. GA at E = 200 gives about 25% smaller .text than -O3, but 19-20% more executed instructions and runtime.
   - GA has 3% fewer IR instructions than -Oz yet 17% larger x86-64 .text. Part of that gap is likely backend attributes: -Oz IR carries optsize/minsize, which the fixed backend honours, while searched sequences start from IR without them.
   - The paper should present the objective as IR size and state this gap.
5. **Drift is concentrated in early simplification, then interactions stabilise.**
   - Median ρ is about 0.3 between raw IR (S0) and every later snapshot.
   - Between later snapshots ρ is higher: 0.58 for S1 vs S2, 1.00 for S2 vs S3 (32/129 programs have byte-identical IR there), and 0.78 for S2/S3 vs S4.
   - Interactions are sparse: 81% of pairs are near zero at S0, rising to 88% at S4.
   - So graphs measured on raw IR describe later states poorly.

**Suggested framing:** a characterisation paper.
- Program-specific headroom in code size on two ISAs (1, 3).
- Drift concentrated early (5).
- Interactions are predictable (pre-registered H3, R² 0.50).
- An honest negative result: pairwise-graph builders plateau and the bottleneck is the builder (1), with a win only at a single compile (2).
- Future work: use predictions inside search (seeding or mutation), not as a fixed graph.
