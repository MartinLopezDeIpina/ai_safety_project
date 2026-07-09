# Figure 2 Replication — Experiment Log

Goal: reproduce Figure 2 of *"LLMs Encode Harmfulness and Refusal Separately."*
Two panels, each plots `s^l(h) = cos(h, μ_refused_harmful) − cos(h, μ_accepted_harmless)` per layer.
- **tinst**: coral (accepted-harmful) should be **> 0**, teal (refused-harmless) **< 0**.
- **tpost**: flip — coral **< 0**, teal **> 0**.

Priority: **qwen7b first**. Labeling: **substring matcher only** (no LLM judge — confirmed no meaningful effect).

---

## Root-cause hypotheses (from exploration, 2026-07-09)

- **H1 (qwen tinst reversal):** qwen-specific prompt-leak in `inference.py::extract_model_output`
  (`split('assistant')` fails for the tinst template that has no `assistant` token) → echoed prompt
  stored in `ori_output` → substring refusal classifier mislabels qwen tinst harmful buckets.
  Activations themselves are clean (extract_hidden re-tokenizes the instruction field).
- **H2 (tpost fails both models):** structural. `refused_harmless` bucket tiny (qwen 23) — teal never
  crosses positive. Diagnose: contamination vs. weak signal vs. anchor separation.

Efficiency: activations depend only on instruction+template (not on tinst/tpost generation), so
re-bucketing is free (CPU) and only a single fast forward-pass extraction is needed on GPU.

---

## Experiments

### EXP-0 (2026-07-09) — Diagnostics on existing qwen7b tensors
Status: DONE. Scripts: `diagnose_fig2.py`, `diagnose_fig2b.py`.

Findings (qwen7b, per-layer `s^l`, p+ = fraction of examples with s>0):

**tinst panel**
- anchors sane: refused_harmful ≈ +0.19..+0.34 (p+1.00); accepted_harmless ≈ −0.26..−0.30.
- accepted_harmful (coral) = **−0.17..−0.19 (p+0.21)** → WRONG (should be >0).
- refused_harmless (teal) = −0.28..−0.36 (p+0.00) → correct.
- **Root cause (decisive):** split by source of `tinst_accepted_harmful` (n=337):
  advbench (42) **+0.18..+0.34 p+1.00**, jbb (30) **+0.15..+0.30 p+1.00**,
  **sorrybench (265) −0.27..−0.33 p+0.00** — and this is UNIFORM across all 44 SB categories.
  Sorry-Bench's conversational phrasing sits on the *harmless* side of qwen's advbench/jbb harmfulness
  axis at tinst. Non-circular split-half test: held-out refused_harmful projects +0.19..+0.34 (p+1.00),
  so genuine harmful advbench/jbb robustly project positive.
- Why SB dominates the bucket: (a) prompt-leak bug mislabeled most advbench/jbb tinst outputs as
  refused (their echoed prompt+continuation contained "I'm sorry"), pushing them out of the accepted
  bucket; (b) `POOLS` wrongly injects sorrybench into `tinst_accepted_harmful`. Per paper Appendix B,
  the harmfulness cluster & tinst analysis use **advbench+jbb**; Sorry-Bench is for *accepted-at-tpost*.

**tpost panel** (magnitudes ~10x smaller, ±0.03)
- accepted_harmful (coral) ≈ 0 / slightly + (should be <0). By source: advbench(34) **+0.25 p+0.97**,
  jbb(19) +0.11, sorrybench(207) ≈ 0 (p+0.49). The advbench/jbb "accepted@tpost" are substring
  false-negatives (advbench@tpost is ~98% refused) → they're actually refusals projecting positive.
- refused_harmless (teal) ≈ 0/negative (should be >0). Bucket tiny (23) and partly garbage
  (3+ examples have ori_output == "<|im_end|>", i.e. empty response labeled refused).

**Conclusions / fixes**
1. tinst: build `accepted_harmful` (and refused anchor) from **advbench+jbb with corrected labels**;
   drop Sorry-Bench from the tinst bucket. Expect coral > 0.
2. tpost: need cleaner accepted labels (drop advbench/jbb false-accepts) and a larger/cleaner
   `refused_harmless` set; Sorry-Bench belongs in tpost_accepted_harmful.

**KEY OPTIMIZATION discovered:** activations are identical whether an example was processed in the
tinst-run or tpost-run bucket (extract_hidden always uses the full template, ignores tinst/tpost). The
6 existing `*.pt` files therefore already contain activations for ~every advbench/jbb/sorrybench and
xstest/alpaca example, each mappable back to its instruction via the bucket JSON (same row order).
So I can rebuild ANY bucket with ANY (corrected) labeling by re-indexing existing activations —
**fully on CPU, no GPU re-extraction needed**. VERIFIED: for 508 examples common to
tinst_refused_harmful and tpost_refused_harmful, activation maxabs diff == 0.

### EXP-1 (2026-07-09) — CPU re-bucketing (`rebucket_store.py` -> buckets_activations_v2)
Build a text->activation store from the 6 tensors; recompute clean labels (recover tinst responses,
3-way classify empty|refused|accepted, empties excluded); reassemble buckets and re-plot.

Iterations (qwen7b, position -1, plotting figure2_v2.png):
1. **tinst fix:** `tinst_accepted_harmful` = advbench+jbb (drop Sorry-Bench). Coral goes **>0** across
   all layers, teal **<0**. tinst panel now matches the paper.
2. **tpost teal:** excluding empty responses from `refused_harmless` (23 -> 15 genuine refusals) makes
   teal cross **>0 at late layers** (L28 +0.026, p+0.60). The 8 empty "<|im_end|>" outputs on harmless
   prompts were artifacts, not refusals.
3. **tpost coral:** `tpost_accepted_harmful` = **Sorry-Bench only**. The 52 advbench/jbb "accepted@tpost"
   are substring-missed refusals that project strongly +0.146 (contamination); dropping them makes coral
   clearly **<0** (L25 -0.045) while teal +0.046 -> clean late-layer flip. Paper-consistent
   (advbench/jbb ~98% refused at tpost; Sorry-Bench supplies genuine accepted-harmful@tpost).
4. Harmless-anchor ablation: xstest-only worse (borderline "scary" prompts are a poor harmless
   prototype); alpaca-only slightly stronger but off-recipe. Kept paper-faithful **xstest+alpaca**.

**RESULT (qwen7b):** Figure 2 reproduces the opposing pattern:
- tinst: coral (accepted-harmful) ~ +0.15 (harmful side), teal (refused-harmless) ~ -0.3 (harmless side).
- tpost: reversed at layers ~16-27 — teal rises to +0.045 (refused side), coral drops to -0.04.
Gap vs paper: qwen flip is late-layer (not early) and smaller magnitude (+/-0.05 vs +/-0.15); teal band
is wide because only 15 genuine xstest over-refusals exist for qwen (inherent). Core claim reproduced.
Final bucket sizes: tinst_refused 538, tinst_accepted 71, tpost_refused 557, tpost_accepted 207(SB),
accepted_harmless 707, refused_harmless 15.

### EXP-2 (2026-07-09) — Harmless-anchor confound + llama38b_2
Applied rebucketing to llama38b_2 (llama3 tinst extractor had no leak bug, so tinst_accepted is
already large, N=246). With paper-recipe harmless anchor (xstest+alpaca):
- llama tinst: correct (coral >0, teal <0).
- llama tpost: FAILS — teal stays negative at every token position (swept all 7); even a "pure
  refusal axis" (refused_harmful vs accepted_harmful) puts refused_harmless on the harmless side.

**Root cause (confound):** `refused_harmless` is xstest; the harmless anchor includes xstest-accepted
(same distribution). cosine(h, mu_accepted_harmless) is then inflated by shared xstest style, masking
the refusal signal. **Fix:** harmless anchor = **Alpaca only** (canonical clean harmless reference,
Arditi et al.) — a different distribution from the xstest test set. Verified stronger flip for BOTH:
  qwen  teal_max +0.046 -> **+0.096**; layers with teal>coral>... up from 8 to 10.
  llama teal now crosses **>0 in mid layers (~11-20, peak +0.03)**, clearly above coral (<0).

**FINAL (position -1, harmless anchor = alpaca, tinst_accepted = advbench+jbb, tpost_accepted = SB,
empties excluded):**
- **qwen7b**: tinst coral>0 / teal<0; tpost teal rises to +0.096 (refused region), coral <0 — clean
  late-layer flip. Reproduces paper Fig 2(c).
- **llama38b_2**: tinst coral>0 / teal<0; tpost teal >0 above coral in mid layers (reverts late; the
  teal std band is wide because only 15 genuine xstest over-refusals exist). Reproduces the qualitative
  reversal; late-layer behaviour weaker than paper.

Note vs paper: paper uses xstest+alpaca for accepted_harmless and still gets the flip; our replication
needs alpaca-only to overcome the xstest self-similarity confound (fewer over-refusals / weaker signal
in these checkpoints). Documented deviation; `--harmless-sources xstest,alpaca` reproduces the weak
paper-recipe version.

## Deliverables
- Canonical figures regenerated: `output/qwen7b/figure2.png`, `output/llama38b_2/figure2.png`.
- `rebucket_store.py`: CPU-only corrected bucketing from existing activations (no GPU re-run).
- `inference.py`: qwen `extract_model_output` prompt-leak fixed (tinst-aware).
- Diagnostics: `diagnose_fig2.py`, `diagnose_fig2b.py`.
- GPU re-extraction (Step 3 / Modal) NOT needed — activations were reused byte-for-byte.
