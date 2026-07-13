# CLAUDE.md

Guidance for Claude Code agents working in this repository.

## What this repo is

This repo **replicates** the paper *"LLMs Encode Harmfulness and Refusal Separately"* (Zhao et al.,
NeurIPS 2025). The paper's thesis: LLMs encode **harmfulness** at the last instruction token
`t_inst` and **refusal** at the last post-instruction token `t_post`, as two separable directions.
The full paper is in `paper.pdf`; the original project README is `README.md`.

There are two layers:

- **`src/`** — the paper's original reference code (inference, evaluation, hidden-state extraction,
  steering). Multi-model (Llama-2 / Llama-3 / Qwen-2).
- **`src/experiments_replication/`** — **our replication pipeline**, layered on top of `src/`. This is
  where essentially all of our work lives.

## Golden rules (read first)

1. **All replication work goes in `src/experiments_replication/`.** Layer new code on top of `src/`;
   do not restructure or "tidy" `src/`.

2. **The paper reference scripts are read-only reference:** `src/inference.py`, `src/eval.py`,
   `src/extract_hidden.py`, `src/utils.py`. Our pipeline imports and subprocess-invokes them; treat
   them as canonical and don't refactor them.
   - **One intentional exception, already made:** `src/eval.py::easy_eval` carries a local extension
     (`_is_genuine_refusal` / `_REFUSE_TO_ACT`, `src/eval.py:265-299`). It special-cases xstest
     over-refusal rows (those carrying both `focus` and `type` fields) so that empathetic
     "sorry to hear…" replies are not miscounted as refusals. Harmful rows still use the byte-for-byte
     original substring matcher. **Leave this in place** — do not "clean it up."

3. **The GPU stages have already been run** for `qwen7b` and `llama38b`. Do **not** re-run
   `infer` / `eval` / `acts` unless you are deliberately regenerating from scratch. Normal iteration is
   `stages=("fig", "fig3")` — pure CPU.

4. **Activation `.pt` files are git-ignored** (`.gitignore` excludes `**/activations/**`) because they
   are too large for GitHub; they are shared out-of-band. They must be present under
   `output/<model><size>/datasets_outputs/activations/` for `gen_buckets` / `fig` / `fig3` to run.
   Everything upstream of them (generations, classified generations, figures) *is* committed.

## The pipeline — `src/experiments_replication/main.py`

Entry point: `main(model, model_size, left, right, stages, bucket_config)` (`main.py:40`).
There is **no argparse** — configure by editing the `main(...)` call in the `__main__` block, then run:

```bash
cd src/experiments_replication
python main.py
```

The current `__main__` runs: `main("qwen", "7b", stages=("fig","fig3"), bucket_config="bucket_config_clean.json")`.

Stages run in order; each is gated by membership in the `stages` tuple:

| Stage | Function (file:line) | Cost | What it does |
|-------|----------------------|------|--------------|
| `infer` | `run_all_inference` (`_00_run_inference.py:106`) | **GPU** | Shells to `src/inference.py` for the 8 dataset×config runs → `generations/`. |
| `eval` | `evaluate` (`_00_run_inference.py:121`) | CPU | `easy_eval(..., mode="refusal")` labels each row (`'5'`=accepted, `'0'`=refused), splits into `classified_generations/`. |
| `acts` | `compute_all_activations` (`_01_compute_activations.py:83`) | **GPU** | Shells to `src/extract_hidden.py --extract_only 1` per classified file → one `.pt` in `activations/`. |
| `gen_buckets` | `gen_buckets` (`dynamic_bucket_formation.py:183`) | CPU | Composes `.pt` sources into train/test clusters per the bucket config. |
| `fig` | `plot_figure2` (`_02_3.1_figure2.py:104`) | CPU | Figure 2. |
| `fig3` | `plot_figure3` (`_03_3.2_figure3.py:109`) | CPU | Figure 3. |

Notes:
- Requesting `fig` and/or `fig3` **implicitly runs `gen_buckets` first** — buckets are built once and
  held in memory (`main.py:55-57`).
- The two figure modules are loaded dynamically (`main.py:12-22`) because their filenames
  (`_02_3.1_figure2.py`, `_03_3.2_figure3.py`) are not valid Python identifiers.
- Model / size values: `qwen` + `7b` or `0.5b`; `llama3` + `8b`; `llama` (= Llama-2) + `7b`.
  The directory name is `f"{model}{model_size}"`, e.g. `qwen7b`, `llama38b`.

## Directory / output layout

```
output/<model><size>/
  datasets_outputs/
    generations/            # 8 files, from `infer`  (committed)
    classified_generations/ # 16 files, from `eval`  (committed)
    activations/            # 16 *.pt, from `acts`   (GIT-IGNORED, shared out-of-band)
  figure2<suffix>.png       # from `fig`   (committed)
  figure3<suffix>.png       # from `fig3`  (committed)
```

The figure `<suffix>` is derived from the bucket-config filename by `_fig_path` (`main.py:25-37`):
`bucket_config.json` → no suffix (`figure2.png`), `bucket_config_clean.json` → `figure2_clean.png`,
`bucket_config_alt.json` → `figure2_alt.png`. This lets parallel configs coexist without clashing.

Input datasets live in `data/`: `advbench.json`, `jbb.json`, `sorry-badq.json` (harmful),
`xstest-harmless.json`, `alpaca_data_instruction.json` (harmless), plus `catqa_*` and persuasion sets.

## The two token positions & `gentinst` / `gentpost` (the paper's core idea)

- **`t_inst`** = token index **1**, encodes **harmfulness**. **`t_post`** = token index **-1**, encodes
  **refusal**. (`POSITION_INDEX = {"tinst": 1, "tpost": -1}`, `dynamic_bucket_formation.py:35`.)
- Each **harmful** dataset is generated **twice** (`RUNS`, `_00_run_inference.py:91-103`):
  - `*_gentinst` — `do_not_use_last_inst_tok=1`: the post-instruction template tokens are removed from
    the prompt, which yields **more accepted** completions (the model isn't cued to refuse). Used to
    study the `t_inst` position.
  - `*_gentpost` — `do_not_use_last_inst_tok=0`: normal full-template generation. Used for `t_post`.
- **Harmless** datasets (alpaca, xstest) run **once** (gentpost config), so they have no gen-token in
  their names.
- **Key efficiency fact:** `src/extract_hidden.py` always re-tokenizes with the *full* template, so
  **every** `.pt` tensor has shape `(L, N, T, H)` and holds **both** `t_inst` and `t_post` positions,
  regardless of which generation config produced the row. The `gentinst`/`gentpost` distinction only
  affects *which responses were generated* (hence which rows land in the accepted vs refused split),
  not which token positions are stored. **This is why re-bucketing is a free CPU re-index and needs no
  GPU re-extraction.**

The 16 activation files per model:
```
advbench_gentinst_{accepted,refused}.pt   advbench_gentpost_{accepted,refused}.pt
jbb_gentinst_{accepted,refused}.pt        jbb_gentpost_{accepted,refused}.pt
sorrybench_gentinst_{accepted,refused}.pt sorrybench_gentpost_{accepted,refused}.pt
alpaca_{accepted,refused}.pt              xstest_{accepted,refused}.pt
```
(3 harmful × 2 gen-configs × 2 labels = 12, plus 2 harmless × 2 labels = 4.)

## The bucketing system — `dynamic_bucket_formation.py` + `bucket_config*.json`

A **cluster** (aka bucket) is the activations for one behavioral category — {accepted, refused} ×
{harmful, harmless} — at one token position. That's 8 cluster names: `<category>_{tinst,tpost}`.
`gen_buckets` returns `{"train": {name → tensor}, "test": {name → tensor}}`, where each returned tensor
is `(L, N, H)` (the token axis is already sliced to the cluster's position).

Config JSON fields:
- `test_ratio` (float) — fraction of each **shared** source's rows assigned to test.
- `seed` (int) — base seed for the deterministic per-source split.
- `bucket_train`, `bucket_test` — lists of entries `[cluster_name, [source_stems], position]`, where
  `source_stems` are `activations/*.pt` filenames without the extension, and `position` is
  `"tinst"`/`"tpost"`.

**Split rule** (`build_splits` + `_split_indices`, md5-seeded per source so the split is stable
regardless of processing order):
- a source in **both** `bucket_train` and `bucket_test` → **ratio-split** (`test_ratio`);
- a source **only in train** → all its rows go to train;
- a source **only in test** → all its rows go to test.

This is how `refused_harmless` is **held out of train** (empty source list) yet still populated in test.
Corrupt zero-norm activation rows (which would make cosine `0/0 = NaN`) are dropped with a warning
(`dynamic_bucket_formation.py:156-161`).

Three configs are checked in:
- `bucket_config.json` — default.
- `bucket_config_alt.json` — paper-faithful "ALT" anchors; `refused_harmless` present in both splits.
- `bucket_config_clean.json` — **what `main.py` currently runs.** accepted-harmful @`t_inst` from
  advbench+jbb, @`t_post` from sorrybench; harmless anchored on **alpaca only**; `refused_harmless` =
  `xstest_refused`, in test only.

If no config file is found, module constants `BUCKET_TRAIN` / `BUCKET_TEST`
(`dynamic_bucket_formation.py:52-73`) are the fallback default.

**Why these anchors** (see `diagnose/EXPERIMENT_LOG.md` for the full derivation): Sorry-Bench's
conversational phrasing sits on the *harmless* side of the advbench/jbb harmfulness axis at `t_inst`, so
it's excluded from accepted-harmful@`t_inst` and used for accepted-harmful@`t_post` instead. And because
`refused_harmless` is xstest, using xstest in the harmless anchor creates a self-similarity confound
that masks the refusal signal — hence the alpaca-only harmless anchor in the clean recipe.

Standalone inspection: `python dynamic_bucket_formation.py <model> <size>` prints each split's cluster
names and `(L, N, H)` shapes.

## What Figures 2 and 3 compute

- **Figure 2** (`_02_3.1_figure2.py`): per-layer score
  `s^l(h) = cos(h, μ_refused_harmful) − cos(h, μ_accepted_harmless)`, drawn as two panels
  (`t_inst`, `t_post`). Anchors μ come from the **train** clusters; the two plotted lines —
  accepted-harmful (coral, solid) and refused-harmless (teal, dash-dot) — from **test**. Expected
  pattern: at `t_inst`, coral > 0 and teal < 0; at `t_post` the two flip.
- **Figure 3** (`_03_3.2_figure3.py`): scatter of per-instruction `Δ_harmful` (x, from `t_inst`) vs
  `Δ_refuse` (y, from `t_post`), one point per instruction, colored by behavioral category. The 4
  anchors come from **train**; points from **test**, paired per instruction (`Δ_harmful` from
  `<base>_tinst`, `Δ_refuse` from `<base>_tpost`).

## Supporting files & notes

- `src/experiments_replication/diagnose/` — earlier one-off tooling plus `EXPERIMENT_LOG.md` (the record
  of how the clean/alt configs were derived and validated). `rebucket_store.py` is the predecessor of
  `dynamic_bucket_formation.py`.
- `src/experiments_replication/modal_run.py` — remote/Modal GPU runner for the extraction stages.
- `src/experiments_replication/prueba.py`, `prueba2.py` — **uncommitted scratch** (token-sequence /
  reasoning-trace probing for Qwen, tied to ongoing "thinking evaluation" exploration). Not part of the
  pipeline; do not treat them as canonical.



# Paper must-known parts

2 Experimental Setup
In this section, we describe the setup in our following experiments.
Models. We focus on widely-used instruct models (also called chat models). They have gone
through several stages of training to fulfill users’ harmless requests and refuse harmful ones [Ouyang
et al., 2022]. In our experiments, we use three widely-adopted open-source models: LLAMA2-
CHAT-7B [Touvron et al., 2023], LLAMA3-INSTRUCT-8B [Meta AI, 2024] and QWEN2-INSTRUCT-
7B [Yang et al., 2024]. Experiments on these models are run on A100-40GB GPUs.
Prompting templates. These instruct models all have their own chat templates for instruction
tuning. For example, Llama2-chat has the following template, “[INST] {user’s instruction}
[/INST]”. We refer to all special tokens after the user’s instruction as post-instruction tokens (e.g.,
[/INST] for Llama2-chat). If not explained, we use the default prompting templates of the tested
models. The exact templates of each model are shown in Table 4 in the Appendix.
Hidden states extraction. Decoder-only Transformers [Vaswani et al., 2017] are the backbone
of mainstream LLMs. Through each layer l ∈ [1, L] in a Transformer model, the hidden state for a
token xt in the input sequence x is updated with self-attention modules that associate xt with tokens
x1:t and a multi-layer perception:
hl
t(x) = hl−1
t (x) + Attnl(xt) + MLPl(xt). (1)
We focus on the residual stream activation hl(xt) of a token position t for an input sequence x at
a certain layer l. Due to self-attention, this hl(xt) contains information on tokens before xt and
itself. In addition, hl(xt) also encodes plans about future tokens that the model will predict in its
response [Pal et al., 2023]. We consider two token positions: (1) Instruction tinst: the last token
of the user’s instruction. (2) Post-instruction tpost-inst: the last token of the post-instruction tokens.
Previous work [Arditi et al., 2024, Zheng et al., 2024, Yu et al., 2025] has focused on tpost-inst. But
both token positions capture information from the entire input instruction. The only difference is
whether they include the special post-instruction tokens. We examine the position tinst because we
find that LLMs may accept a harmful instruction at tinst yet successfully refuse it at tpost-inst, which
implies refusal may be specifically encoded at tpost-inst (see details in Section 3.1). Unless otherwise
specified, accepting or refusing examples refer to model behaviors at the tpost-inst position using the
default prompting template.
Datasets. We employ a wide range of public datasets. For harmful instructions, we use Ad-
vbench [Zou et al., 2023b], JBB [Chao et al., 2024], and Sorry-Bench [Xie et al., 2025] which contain
naive harmful requests. For harmless instructions, we follow previous work [Arditi et al., 2024] to use
ALPACA, an instruction finetuning dataset [Taori et al., 2023]. We also consider harmless prompts
leading to over-refusal [Röttger et al., 2023, Shi et al., 2024, Cui et al., 2024], where the model’s
refusal mechanism is so strong that it will refuse benign requests. For this category, we use examples
from Xstest [Röttger et al., 2023]. See Appendix B for further details about the datasets.
3
Jailbreak methods. We consider jailbreak methods that make LLMs accept harmful instructions.
We employ three different types of jailbreak methods. (1) Adversarial suffixes (GCG specifically
[Zou et al., 2023b]): A sequence of learnable suffix tokens that are optimized to elicit acceptance
responses. (2) Persuasion [Zeng et al., 2024]: Persuasion techniques are applied to rephrase naive
harmful instructions to persuade LLMs to accept them. (3) Adversarial prompting templates [Yu et al.,
2023]: Harmful instructions are inserted into carefully constructed jailbreak prompting templates.
Examples of these jailbreak methods are shown in Table 11 in the Appendix.
Refusal rate. Instruct models are usually finetuned to return certain fixed phrases to refuse users’
prompts, e.g., “Sorry, I cannot”. To evaluate the models’ refusal rate, we follow the conven-
tion [Zou et al., 2023b, Arditi et al., 2024, Zhou et al., 2025] to compile a set of common refusal
substrings. In Section 3.5, the rate is computed based on the refusal token “No”. If the model’s
response contains one of the refusal substrings, we classify it as refusal; otherwise, it is classified as
non-refusal. We calculate the refusal rate out of all the test examples.
3 Decoupling Harmfulness from Refusal
In this section, we investigate the hidden states of harmful/harmless prompts at two different token
positions, the last token of the instruction tinst and the last token of the post-instruction tokens tpost-inst.
This is motivated by our first observation (Section 3.1) that removing all the post-instruction tokens
will reduce LLMs’ refusal behaviors (Table 1). Next, we demonstrate that harmfulness and refusal
may be encoded separately at these two token positions, since the hidden states at tinst form clusters
based on the instruction’s harmfulness, while the hidden states at tpost-inst form clusters based on
whether the instruction is refused (Section 3.2). Then, we quantify the correlation between these
harmfulness and refusal clusters, and find they are not always strongly correlated (Section 3.3). Next,
we show that steering with the harmfulness direction can also lead to refusal behaviors (Section 3.4).
Finally, we show that steering with the harmfulness direction and the refusal direction will lead to
opposite behaviors in our designed reply inversion task, providing causal evidence that LLMs encode
harmfulness and refusal separately (Section 3.5).
Refusal Rate (%) w/ post-instruction tokens w/o post-instruction tokens
LLAMA2-CHAT-7B 100.0 85.3
LLAMA3-INSTRUCT-8B 96.0 58.9
QWEN2-INSTRUCT-7B 98.0 81.3
Table 1: Refusal rates of harmful instructions when prompting with and without post-instruction
special tokens in the prompting template. The refusal rate drops dramatically without post-instruction
special tokens.
3.1 Removing post-instruction tokens weakens refusal abilities
Observation. We find that LLMs can refuse harmful instructions at tpost-inst while accepting
them at tinst. In other words, the refusal ability of harmful instructions can be weakened by
removing the post-instruction special tokens in the prompting template. As shown in Table 1,
all the tested LLMs have a lower refusal rate of harmful instructions in Advbench [Zou
et al., 2023b] when prompted without post-instruction tokens. Examples model outputs are
shown in Figure 19 in the Appendix. Past work [Jiang et al., 2025] has shown that different
prompting templates can weaken the refusal ability of LLMs. Our results further indicate the
importance of post-instruction tokens in generating refusal replies. Those results imply that
LLMs may not formulate refusal signals until the post-instruction tokens are passed to the models.
Our findings also support the hypothesis of template-anchored safety alignment [Leong et al., 2025]
that LLMs overly depend on post-instruction tokens in the prompting template to form their decisions
of refusal.
Hypothesis. Both tpost-inst and tinst contain the information of the whole input instruction (due to
self-attention in Transformers [Vaswani et al., 2017]), but LLMs’ refusal behaviors are much stronger
4
0 10 20 30
Layers
0.125
0.100
0.075
0.050
0.025
0.000
0.025
0.050
0.075
0.100
sl(hl)
refused harmful
accepted harmless
token position tinst
accepted harmful
refused harmless0 10 20 30
Layers
0.4
0.3
0.2
0.1
0.0
0.1
sl(hl)
refused harmful
accepted harmless
token position tinst
accepted harmful
refused harmless0 5 10 15 20 25
Layers
0.08
0.06
0.04
0.02
0.00
0.02
0.04
0.06
0.08
sl(hl)
refused harmful
accepted harmless
token position tinst
accepted harmful
refused harmless0 10 20 30
Layers
0.3
0.2
0.1
0.0
0.1
sl(hl)
refused harmful
accepted harmless
token position tpost-inst
accepted harmful
refused harmless(a) LLAMA3-INSTRUCT-8B0 10 20 30
Layers
0.4
0.2
0.0
0.2
sl(hl)
refused harmful
accepted harmless
token position tpost-inst
accepted harmful
refused harmless (b) LLAMA2-CHAT-7B0 5 10 15 20 25
Layers
0.15
0.10
0.05
0.00
0.05
sl(hl)
refused harmful
accepted harmless
token position tpost-inst
accepted harmful
refused harmless (c) QWEN2-INSTRUCT-7B
Figure 2: The internal clustering of hidden states extracted at tinst (the first row) and tpost-inst (the
second row) exhibit opposing patterns. The red region: Cl
refused harmful (the cluster of refused harmful
instructions). The green region: Cl
accepted harmless (the cluster of accepted harmless instructions). At
each token position, we collect hidden states of two special misbehaving cases: accepted harmful
instructions (the red line) and refused harmless instructions (the green line) to see which cluster
these two cases fall in. The first row: At tinst, across layers, accepted harmful instructions (the red
line) mostly fall in Cl
refused harmful (the red region), while refused harmless instructions (the green line)
mostly fall in Cl
accepted harmless (the green region). This implies that the clustering may be based on
whether the instruction is harmful or harmless, regardless of whether it is refused or accepted. The
second row: At tpost-inst, the clustering behavior is reversed. Now the accepted harmful instructions
(the red line) fall in Cl
accepted harmless (the green region), while refused harmless instructions (the green
line) fall in Cl
refused harmful (the red region). This implies that at tpost-inst, the clustering may be based on
whether the instruction is refused or accepted, regardless of whether it is harmful or harmless. In
Section 3.5, we further provide causal evidence supporting that harmfulness/ harmlessness features
are encoded at tinst, while refusal/acceptance features are encoded at tpost-inst.
at tpost-inst. We then ask: What is encoded at tinst? Is that different from the refusal signals encoded
at tpost-inst? We hypothesize that: at tinst, the hidden states of harmful instructions may encode
harmfulness, and then at tpost-inst, the hidden states will encode explicit refusal signals for the model
to generate the rejection responses. We verify our hypothesis in Section 3.2 by analyzing how the
hidden states of different instructions (harmful but accepted, and harmless but rejected) form clusters
at different token positions.
3.2 Hidden states cluster by harmfulness at tinst, and by refusal at tpost-inst
Motivated by the different refusal behaviors with and without tpost-inst, we extract hidden states at tinst
and tpost-inst to examine what each position encodes. As hidden states often form distinct clusters based
on the input features they encode [Zheng et al., 2024, Marks and Tegmark, 2023, Tigges et al., 2023],
we analyze how harmful/ harmless instructions that lead to different models’ behaviors form clusters at
tinst and tpost-inst. Specifically, we ask: Is the clustering in the latent space based on,(1) the instruction’s
harmfulness/ harmlessness or (2) its refusal/acceptance? To investigate this question, we first compute
the clusters of hidden states for instructions with desired model behaviors (refused harmful instructions
and accepted harmless instructions). We then analyze the misbehaving instructions (accepted but
harmful instructions, and refused but harmless instructions) to see which cluster they fall in. For
instance, if the hidden states of accepted but harmful instructions are closer to the cluster of refused
harmful instructions than that of accepted harmless instructions, it suggests that the instruction’s
harmfulness/harmlessness plays a more important role in the clustering than its refusal/acceptance.
5
Instruction clustering. We first collect the hidden states of accepted harmful instructions, refused
harmful instructions, accepted harmless instructions, and refused harmless instructions at tinst and
tpost-inst respectively (data used are detailed in Section 2). Then, at each layer l, we compute
the cluster of refused harmful instructions (Cl
refused harmful), and the cluster of accepted harmless
instructions (Cl
accepted harmless) at the studied token position on the training set. The cluster centers are
the mean of these instructions’ hidden states and are denoted as μl
harmful refused and μl
harmless accepted.
To decide which cluster a test instruction x belongs to at each layer l, we calculate the cosine
similarity between its hidden states hl and the two cluster centers, cos_sim(hl, μl
refused harmful) and
cos_sim(hl, μl
accepted harmless). Then we calculate the following:
sl(hl) = cos_sim(hl, μl
refused harmful) − cos_sim(hl, μl
accepted harmless). (2)
If sl(hl) > a, hl ∈ Cl
refused harmful; If sl(hl) < a, then hl ∈ Cl
accepted harmless. We by default set the
threshold a as 0 in this work, which has an intuitive mathematical interpretation: hl is assigned to the
cluster whose center it is closer to. However, the oracle value for a in LLMs may not necessarily
be 0, as internal clusters are likely to be fuzzy and overlapped. We leave further investigation on
estimating the oracle as future work. We then compute the average sl(hl) for all the misbehaving
accepted harmful instructions and refused harmless instructions at each token position to see, on
average, which cluster these misbehaving examples are closer to. This allows us to assess whether
clustering is primarily driven by the refusal/acceptance feature or the harmful/harmless feature.
At tinst, hidden states primarily form clusters by harmfulness; at tpost-inst, hidden states form
cluster by refusal. The results of different models are shown in Figure 2. We find that at tinst, harm-
fulness plays a more decisive role in clustering, while at tpost-inst, refusal plays a more important role in
clustering. For example, at tinst (the first row of Figure 2), for all three models tested, across all layers,
the hidden states of accepted harmful instructions (the red solid line) mainly fall in the Cl
refused harmful
cluster, and refused harmless instructions (the green dashed line) mainly fall in the Cl
accepted harmless
cluster (the green region). These results suggest that at tinst, the clustering is driven more by the
harmfulness feature of the instructions than by whether they were refused. However, at tpost-inst (the
second row of Figure 2), the clustering behavior is reversed. The hidden states of refused harmless
instructions (the green dashed line) fall in Cl
refused harmful (the red region), and the accepted harmful
instructions (the red solid line) fall in Cl
accepted harmless (the green region). These results suggest that the
clustering is driven more by whether the instruction was accepted or refused, regardless of whether
it was actually harmful or harmless. Apart from the two positions tinst and tpost-inst, we also study the
clustering patterns at more token positions and perform similar layer-wise analysis in Appendix F.1.
We find that clustering based on the harmfulness of instructions is the most evident at tinst.
Additionally, in both Llama3 and Qwen2 models, we observe a similar layer-wise pattern that at
tpost-inst, refused harmless instructions (the green dashed line) start to fall in Cl
refused harmful (the red
region) and accepted harmful instructions (the red solid line) in Cl
accepted harmless (the green region)
in early layers, indicating the early emergence of strong refusal signals. However, such strong
refusal signals do not appear until the later layers for Llama2. We suspect this is due to Llama2’s
limited capabilities, requiring deeper layers to make refusal decisions. We leave it as future work to
understand the role of each layer. Overall, our results demonstrate at tinst and tpost-inst, the internal
clustering of instructions exhibit opposing patterns, which suggests that harmfulness may be encoded
at tinst, while refusal may be encoded at tpost-inst.
3.3 Correlation between beliefs of harmfulness and refusal
In this section, we quantitatively analyze the correlation between the belief of harmfulness and the
belief of refusal. We interpret the LLM’s belief as reflected by which cluster the hidden state of
an instruction falls into in the latent space. We find that sometimes the model may internally
recognize the correct level of harmfulness in input instructions, yet still exhibit incorrect refusal
or acceptance behaviors. Formally, following the clustering analysis in Section 3.2, we define the
cluster formed by harmful instructions at position tinst as the harmfulness cluster Cl
harmful in layer l,
whose center is denoted as μl, tinst
harmful. Similarly, we denote the harmlessness cluster at tinst as Cl
harmless
and its center as μl, tinst
harmless. Then at tpost-inst, we denote the clusters formed by refused and accepted
instructions as Cl
refusal and Cl
accept respectively, whose centers are μl, tinst
refusal and μl, tinst
accept. For an input
6
instruction x whose hidden state at token t in layer l is hl
t, its belief of harmfulness and refusal is
defined respectively as
∆harmful = Avg(sl(hl
tinst )) = 1
L
LX
l=1
(cos_sim(hl
tinst , μl, tinst
harmful) − cos_sim(hl
tinst , μl, tinst
harmless)), (3)
∆refuse = Avg(sl(hl
tpost-inst )) = 1
L
LX
l=1
(cos_sim(hl
tpost-inst , μl, tpost-inst
refuse ) − cos_sim(hl
tpost-inst , μl, tpost-inst
accept )).
(4)0.2 0.1 0.0 0.1 0.2
harmful
0.3
0.2
0.1
0.0
0.1
0.2
0.3
refuse
accepted harmful
refused harmless
accepted harmless
refused harmful
Figure 3: Correlation between the
model’s beliefs of harmfulness and re-
fusal on Llama2. Each point is a sam-
pled instruction. We show that refusing
an instruction is not necessarily aligned
with the model’s internal belief of harm-
fulness. For example, refused harmless
instructions have negative harmful be-
lief scores, indicating that the model in-
ternally considers them as not harmful,
even though it behaviorally refuses them.
We then compute the belief for a random sample of
each category in the test set. The results on Llama2 are
shown in Figure 3. Accepted harmless instructions (green
squares) and refused harmful instructions (red dots) have
a relatively strong positive correlation between the beliefs
of harmfulness ∆harmful and refusal ∆refusal. However, we
find that refused harmless instructions have significantly
low belief scores in harmfulness. This suggests that, al-
though the model over-refuses these harmless instructions
behaviorally, it still internally deems these instructions
as harmless. This result also supports the prior hypothesis
[Röttger et al., 2023, Bianchi et al., 2023] that refusal can
be triggered by some shallow syntax features despite the
harmlessness of user instructions. In the case of accepted
harmful instructions, the belief of harmfulness ∆harmful
remains positive for many examples, indicating that LLMs
internally view them as harmful despite accepting them.
Overall, our results suggest that refusal is generally cor-
related with harmfulness in LLMs. However, there exist
cases where refusing (or accepting) an instruction does not
align with the model’s internal perception of harmfulness.

B Data
In Section 3.2, we need to find refused and accepted harmful instructions at the respective token
positions tinst and tpost-inst to investigate the clustering patterns. Refused harmful instructions are
sampled from Advbench [Zou et al., 2023b] and JBB [Chao et al., 2024]. As for accepted harmful
instructions, we aggregate Advbench, JBB and Sorry-Bench [Xie et al., 2025] to find examples.
This is because at tpost-inst, almost all examples from Advbench and JBB will be rejected by the
models. To find sufficient harmful instructions that will bypass refusal at tpost-inst, we also employ
Sorry-Bench, which contains a 44-class safety taxonomy across four domains. Some sub-categories
of harmful instructions are shown to bypass the refusal of LLMs frequently. In comparison, we can
more easily find accepted harmful examples at the tinst position because the refusal rate drops greatly
when prompting without post-instruction tokens, as shown in Table 1.
In further sections, we mainly sample harmful instructions refused at tpost-inst position from Advbench
and JBB to compute the center of the harmfulness cluster μl, tinst
harmful and that of the refusal cluster
μl, tpost-inst
refuse . In Section 5, we also add harmful instructions that were accepted at tpost-inst from Advbench
and JBB, or refused harmful examples from SorryBench to the sampling pool for constructing μl, tinst
harmful
for the latent guard model. We find that incorporating these examples improves the classification
performance. We note that such a latent detector can fail when tested on out-of-domain examples
(Appendix H) which we consider as a limitation in application.
For harmless instructions, we run through Xstest [Röttger et al., 2023] for each model to find refused
harmless instructions, which are then held out for testing. The Xstest dataset is carefully constructed
with harmless instructions that include certain keywords (e.g., “kill”, “strangle”), which may lead
LLMs to mistakenly refuse the instruction. Examples of harmless instructions leading to refusal are
shown in Table 5. We aggregate the rest of harmless and accepted examples in Xstest with examples
from ALPACA instruction finetuning dataset as the set of accepted harmless instructions. We also
sample from harmless instructions accepted at tpost-inst to compute the center of harmlessness cluster,
i.e., μl, tinst
harmless.
Prompt Model Response
How do I strangle options to maximise my money? I can’t provide information on how to strangle op-
tions. Is there anything else I can help you with?
What’s the easiest way to sever my employment
with a company?
I can’t provide information on how to sever your
employment with a company. If you’re considering
leaving your job, I suggest speaking with a career
counselor or a trusted mentor to discuss your op-
tions and create a plan that’s right for you. Is there
anything else I can help you with?
Table 5: Examples of refused harmless instructions and corresponding responses by Llama3 when
we prompt with the default template.
17

F Analysis on More Token Positions
In this section, we investigate LLMs’ hidden states extracted from token positions other than tinst and
tpost-inst. We extract tokens starting from the position right before tinst until tpost-inst as illustrated in
Figure 13. We first compare the clustering patterns of hidden states extracted at each token position.
We then study the steering performance with directions extracted at each token position. Our results
suggest that tinst position encodes harmfulness the most evidently.
F.1 Clustering at different token positions
Following implementations in Section 3.2, at each token position, we compute sl(hl) by Equation 2
to measure whether the hidden states of test cases hl at layer l fall in the cluster of rejected harmful
instructions or accepted harmless instructions. Larger positive sl(hl) means these hidden states are
closer to the cluster of refused harmful instructions. In Figure 14, we average the layer-wise sl(hl) in
the middle layers (9 to 20) as they tend to be more responsible for handling harmfulness information
inside LLMs (evidenced by observation that the steering performance reaches peak in the middle
layers in experiments of Section 3.4 and Section 3.5). If a token position encodes harmfulness, then
the clustering of examples in the latent space should reflect the shared harm-related features instead
of the refusal-related features. Specifically, the hidden states of accepted harmful instructions should
fall within the red region (the cluster of refused harmful instructions), while the refused harmless
instructions should fall within the green region (the cluster of accepted harmless instructions). Among
all the token positions tested, only tinst demonstrates this desired clustering pattern. Full layer-wise
results are shown in Figure 15.
22
0 5 10 15 20 25 30
Layers
0.6
0.4
0.2
0.0
sl(hl)
refused harmful
accepted harmless
Accepted Harmful
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4
position=5 (tpost-inst)
0
1
2
3
4
5
6
7
Position Index(a) LLAMA3-INSTRUCT-8B0 5 10 15 20 25 30
Layers
0.2
0.1
0.0
0.1
0.2
0.3
0.4
sl(hl)
refused harmful
accepted harmless
Refused Harmless
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4
position=5 (tpost-inst)
0
1
2
3
4
5
6
7
Position Index (b) LLAMA3-INSTRUCT-8B0 5 10 15 20 25 30
Layers
0.4
0.3
0.2
0.1
0.0
0.1
sl(hl)
refused harmful
accepted harmless
Accepted Harmful
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4 (tpost-inst)
0
1
2
3
4
5
6
Position Index
(c) LLAMA2-CHAT-7B0 5 10 15 20 25 30
Layers
0.4
0.3
0.2
0.1
0.0
0.1
0.2
sl(hl)
refused harmful
accepted harmless
Refused Harmless
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4 (tpost-inst)
0
1
2
3
4
5
6
Position Index (d) LLAMA2-CHAT-7B0 5 10 15 20 25
Layers
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.1
sl(hl)
refused harmful
accepted harmless
Accepted Harmful
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4 (tpost-inst)
0
1
2
3
4
5
6
Position Index
(e) QWEN2-INSTRUCT-7B0 5 10 15 20 25
Layers
0.2
0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
sl(hl)
refused harmful
accepted harmless
Refused Harmless
position=-2
position=-1
position=0 (tinst)
position=1
position=2
position=3
position=4 (tpost-inst)
0
1
2
3
4
5
6
Position Index (f) QWEN2-INSTRUCT-7B
Figure 15: Clustering at different token positions
23
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
Layer
0
20
40
60
80
100
Acceptance rate
pos -2
pos -1
pos 0 (tinst)
pos 1
pos 2
pos 3
pos 4 (tpost-inst)(a) Qwen20
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
Layer
0
20
40
60
80
100
Acceptance rate
pos -2
pos -1
pos 0 (tinst)
pos 1
pos 2
pos 3
pos 4
pos 5 (tpost-inst) (b) Llama3
Figure 16: Steering harmless instruction in the reply inversion task with directions between harmful
instructions and harmless instructions extracted at different token positions. If the direction encodes
harmfulness, our intervention should trigger the model to flip ‘No’ to ‘Certainly’ as the model may
perceive the harmless instruction as harmful, leading to increased acceptance rate.
F.2 Directions extracted at different token positions
We provide further evidence by extracting steering directions at different token positions for the reply
inversion task. Specifically, we extract directions from the cluster of harmless instructions to the
cluster of harmful instructions at each token position following Section 3.4. We then apply those
directions to the tokens before the inversion question to assess how strongly each direction raises the
LLM’s perception of harmfulness. Results are shown in Figure 16. For both Qwen2 and Llama3,
the steering direction extracted at tinst (i.e., position 0) achieves the strongest intervention effect: the
model is more likely to interpret the originally harmless instructions as harmful, thereby triggering an
acceptance response in the reply inversion task.
