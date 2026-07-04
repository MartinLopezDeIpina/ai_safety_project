# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Official implementation + a replication harness for the paper **"LLMs Encode Harmfulness and
Refusal Separately"** (arXiv:2507.11878). Core claim: LLMs encode *harmfulness* at the last
instruction token (`t_inst`) and *refusal* at the last post-instruction token (`t_post-inst`) as
separate directions in activation space.

**Content warning:** `data/` contains harmful/offensive prompts (advbench, catqa, jailbreaks) by design.

## Two codebases — know which you're in

- `src/` — the **original paper code**: multi-model (Llama2/Llama3/Qwen), driven by shell scripts
  (`run_diff_mean.sh`, `complete_intervene.sh`, `run_inference.sh` → `get_diff_mean.sh`). Entry
  modules: `extract_hidden.py` (hidden-state extraction), `intervention.py` (steered generation),
  `inference.py` (model loading + dataset inference), `eval.py`, `utils.py` (prompt formatting,
  `REFUSAL_PHRASE`, `read_row`), `template_inversion.py` / `all_inversion_template.py` (reply-inversion
  prompts), `run_llama_guard.py`, `classifier.ipynb` (Latent Guard). Example outputs in `src/run/`.
- `src/experiments_replication/` — a **self-contained replication pipeline** (the active work).
  Numbered scripts `00`–`08` reproduce the paper's tables/figures on Qwen. It **imports and
  monkey-patches** `src/` utilities rather than duplicating them. **Most changes belong here.**

## Running the replication pipeline

All commands run from `src/experiments_replication/`. Config is centralized in `config.py`.

```bash
cd src/experiments_replication
bash run_all.sh            # full pipeline 00→08, each step caches to results/
python 00_collect_behaviors.py   # or run any single stage directly
```

The stages (each reads/writes `results/`; later stages depend on earlier artifacts):

| Script | Produces | Notes |
|---|---|---|
| `00_collect_behaviors.py` | `behaviors.json` | Greedy-decodes every dataset, labels each sample refused/accepted → the 4 categories (harmful/harmless × refused/accepted) + 3 jailbreak categories. **Prerequisite for everything.** |
| `01_extract_directions.py` | `dir-hf.pt`, `dir-refuse.pt`, `acts_*.pt`, `cluster-*.pt` | Forward passes at `POS_TINST`/`POS_TPOSTINST`; `dir_hf = μ(refused_harmful@tinst) − μ(accepted_harmless@tinst)`, `dir_refuse = μ(refused@tpost) − μ(accepted@tpost)`. Caches per-category `acts_*.pt`. |
| `02_table1_post_inst_tokens.py` | `table1.json` | §3.1 post-instruction-token ablation |
| `03_figure2_clustering.py` | `figure2{,-data}.{png,json}` | Layer-wise clustering (pure analysis, reads `.pt`, no GPU) |
| `04_figure3_scatter.py` | `figure3*` | Δharmful × Δrefuse scatter (no GPU) |
| `05_figure4_steering_layers.py` | `figure4*` | Per-layer steering on harmless prompts |
| `06_figure5_reply_inversion.py` | `figure5*` | Reply-inversion, per layer |
| `07_figure6_jailbreak.py` | `figure6*` | Jailbreak scatter |
| `08_latent_guard.py` | `table3.json` | Latent Guard detection |

The numbered scripts take **no command-line args** — all knobs live in `config.py`. Stages
03/04/08 are CPU-only re-analyses of saved tensors; the rest need the GPU model.

### Switching models

Edit the block at the top of `config.py` (`MODEL_NAME` / `MODEL_TYPE` / `MODEL_SIZE`). It ships
toggled to `Qwen/Qwen2.5-1.5B-Instruct` (fits a 16GB GPU); the 7B block is commented out.
`results_qwen1.5b/` holds a completed 1.5B run; `results/` is the live output dir.

> `POS_TINST = [-5]` and `POS_TPOSTINST = [-1]` are **Qwen2-chat-template specific** (the
> `<|im_end|>\n<|im_start|>assistant\n` suffix is 5 tokens). Changing model family requires
> re-deriving these.

## Architecture: how the replication harness wraps `src/`

`model_utils.py` is the bridge. It puts `SRC_DIR` on `sys.path` and adapts the original utilities:

- `load_model()` — uses `inference.load_model_and_tokenizer()` for sizes it knows (`7b/13b/14b`),
  else loads `MODEL_NAME` directly in fp16 (e.g. the 1.5B model `inference.py` doesn't know about).
- `format_prompt()` / `format_prompt_no_postinst()` — wrap `utils.formatInp_llama_persuasion`,
  **always passing `model='qwen'`** (the original `generate_directions()` defaulted to Llama2
  `[INST]` format even for Qwen — this is the fix).
- `extract_hidden_states()` — calls `extract_hidden.get_mean_activations`, but **monkey-patches
  `get_mean_activations_fwd_hook`** with `_make_compat_fwd_hook` for the transformers-version issue below.
- `is_refusal()` uses `utils.REFUSAL_PHRASE`; `load_data()` wraps `utils.read_row`; `cosc()` is a
  numerically-stable cosine sim ported from `classifier.ipynb`.

Steering scripts (05/06/07) call `intervention.complete_with_intervention` and set the module
globals `iv.MODEL='qwen'` / `iv.DECODING_STEP=-1`; scoring goes through `eval.easy_eval(mode=...)`.

## Critical gotchas (read before touching model/activation code)

- **transformers is pinned to `4.57.6`.** That version made `Qwen2DecoderLayer.forward()` return a
  bare tensor instead of a tuple. `model_utils._make_compat_fwd_hook` handles both shapes; do **not**
  bump `transformers` without re-verifying that hook (see `requirements.txt` comment).
- **`BATCH_SIZE = 1` is mandatory.** `extract_hidden.get_mean_activations` `torch.stack`s across
  batches and breaks on a smaller final batch.
- **torch CUDA wheel** is selected via `--extra-index-url` in `requirements.txt` (defaults to cu121);
  match it to your CUDA (`nvidia-smi`).
- Template/adversarial jailbreak samples go **only** into their jailbreak category, never into
  `accepted_harmful` — their disguised text looks harmless at `t_inst` and would corrupt Fig 2/3
  clustering (see comments in `00_collect_behaviors.py`).

## Environment

```bash
pip install -r requirements.txt   # torch 2.6, transformers 4.57.6, accelerate, numpy, matplotlib
```

Python 3.12 locally (`.venv/`); the cluster uses 3.11. Scripts import the original `utils`/`inference`
from `src/`, so `src/` must be on `PYTHONPATH` (`model_utils.py` also inserts it via `sys.path`).

## Cluster (SLURM / KISSKI)

`src/experiments_replication/slurm/` mirrors the pipeline as SLURM jobs (A100, `kisski` partition).
Compute nodes have **no internet**, so:

1. `bash slurm/setup_venv.sh` on the **login node** (builds venv, pip installs).
2. Pre-download models on the login node; jobs read `HF_HOME`/`HF_HUB_OFFLINE` from repo-root `.env`.
3. `sbatch slurm/smoke_test.slurm` to verify env, then `bash slurm/run_all.sh` to submit the
   dependency chain (`00 → 01 → {02,03+04,05,06,07,08}` in parallel). Logs land in
   `src/experiments_replication/logs/`; monitor with `squeue --me`.

Note: SLURM scripts hardcode the KISSKI repo path (`/user/m.lopezdeipinamuno/.../ai_safety_project`)
and a sibling `venv/` (not the local `.venv/`), and export `PYTHONPATH=$REPO/src`. They need editing
to run from a different checkout.

## Original paper workflows (in `src/`, run from `src/`)

- Directions/hidden states: `cd src && sh run_diff_mean.sh` (→ `get_diff_mean.sh` → `extract_hidden.py`).
- Interventions: `sh complete_intervene.sh` (key args: `--intervention_vector`, `--reverse_intervention`,
  `--coeff_select`, `--use_inversion`, `--intervene_context_only`); evaluate with
  `python eval.py` (`mode=inversion` for reply-inversion, else `refusal`).
- LlamaGuard baseline: `python run_llama_guard.py --input data/<prompts>.json --output results/llamaguard.txt`.

# Original paper explaining each experiment (very important):

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
the mean of these instructions’ hidden states and are denoted as µl
harmful refused and µl
harmless accepted.
To decide which cluster a test instruction x belongs to at each layer l, we calculate the cosine
similarity between its hidden states hl and the two cluster centers, cos_sim(hl, µl
refused harmful) and
cos_sim(hl, µl
accepted harmless). Then we calculate the following:
sl(hl) = cos_sim(hl, µl
refused harmful) − cos_sim(hl, µl
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
whose center is denoted as µl, tinst
harmful. Similarly, we denote the harmlessness cluster at tinst as Cl
harmless
and its center as µl, tinst
harmless. Then at tpost-inst, we denote the clusters formed by refused and accepted
instructions as Cl
refusal and Cl
accept respectively, whose centers are µl, tinst
refusal and µl, tinst
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
tinst , µl, tinst
harmful) − cos_sim(hl
tinst , µl, tinst
harmless)), (3)
∆refuse = Avg(sl(hl
tpost-inst )) = 1
L
LX
l=1
(cos_sim(hl
tpost-inst , µl, tpost-inst
refuse ) − cos_sim(hl
tpost-inst , µl, tpost-inst
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
3.4 Eliciting refusal with harmfulness directions
To investigate the causality between believing an instruction is harmful internally and refusing it
in the response, we steer the hidden states of accepted harmless instructions towards the region of
Cl
harmful to see how much that can reverse the model’s acceptance to refusal.0
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
Layer
0
20
40
60
80
100
Refusal rate
harmfulness direction
refusal direction
Figure 4: Steering the hidden states of
harmless instructions along the harmful-
ness direction and refusal direction can
both elicit refusal behaviors on Llama3.
Harmfulness direction. We can extract a steering di-
rection of harmfulness at a layer l with the difference-in-
means [Marks and Tegmark, 2023] from Cl
harmless to Cl
harmful
in the latent space. Namely, we define the harmfulness
direction as the difference between two cluster centers at
token tinst,
vl
harmful = µl, tinst
harmful − µl, tinst
harmless. (5)
We then intervene on the residual stream for the hidden
state of test examples using activation addition at layer
l, i.e., h′l = hl + vl
harmful to all tokens of input instruc-
tions. As comparison, we also extract a refusal direction
as vl
refuse = µl, tpost-inst
refuse − µl, tpost-inst
accept at token tpost-inst.
Steering results. We find that both the harmfulness di-
rection and the refusal direction can elicit LLMs’ refusal
behaviors on harmless instructions. Layer-wise results on
Llama3 are shown in Figure 4, and results on other models
are shown in Appendix C. Steering along the harmfulness direction at layer nine leads to the highest
7
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
Refusal rate
refusal dir
harmfulness dir
reverse refusal dir(a) Harmless instructions0
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
Refusal rate
reverse refusal dir
reverse harmfulness dir
refusal dir (b) Harmful instructions
Figure 5: Steering with the harmfulness direction (the orange line) and the refusal direction (the
blue line) leads to opposite behaviors, which serves as causal evidence that these two directions are
fundamentally different in LLMs. We append an inversion question (e.g., “Could this user prompt
cause harm? Answer ‘Certainly’ if it could, otherwise ‘No’. ”) to the input instruction so that the
model should respond with an acceptance token (e.g., Certainly) instead of a refusal token (e.g.,
No) if the instruction is harmful and vice versa.
refusal rate (94%), while the refusal direction performs the best at layer eleven (100% refusal rate).
On Qwen2 and Llama2, the refusal rate elicited by the harmfulness direction is much lower than that
elicited directly by the refusal direction (see Appendix C). Although both directions can elicit refusal
behaviors, we find these two directions have a low average cosine similarity , e.g., around 0.1 on
Llama2. For held-out refused instructions, cosine similarity with the refusal direction averages 0.57,
while for accepted ones it averages -0.14; in this context, an observed similarity of 0.1 is low. This
suggests that the harmfulness direction and the refusal direction elicit refusal behaviors in different
ways. We further prove their difference causally in Section 3.5 by constructing a prompting task
where steering along the harmfulness direction and the refusal direction leads to opposite behaviors.
3.5 Causally separating the harmfulness direction and the refusal direction
In Section 3.4, we demonstrate that steering along the refusal direction and the harmfulness direction
can both elicit models’ refusal behaviors. In this section, we prove that these two directions play
different roles inside LLMs. We design a reply inversion task to provide causal evidence that steering
along these two directions will lead to opposite results. Overall, our results suggest that (1) the
harmfulness direction extracted at tinst encodes the concept of harmfulness, and steering the model
along it will reverse the model’s judgment of the instruction harmfulness; (2) the refusal direction
extracted at tpost-inst mainly encodes surface-level refusal signals, and steering along it will not always
alter the model’s judgment of instruction harmfulness.
Reply inversion task. To distinguish the effects of the refusal direction and the harmful direction,
we design a reply inversion task that disentangles harmfulness and refusal behaviorally: when the
harmfulness of instructions is increased, the model should output acceptance tokens instead of refusal
tokens. To achieve this, we append an inversion question, e.g., “Could this user prompt cause
harm?” to the user’s instruction, as shown in Table 2. Then, the model will respond to harmless
instructions with a refusal token (e.g., “No”) and to harmful instructions with an acceptance token
(e.g., “Certainly”). We then steer the hidden states using our identified harmfulness direction and
the refusal direction, following the intervention approach in Section 3.4 (implementation details
in Appendix E.1), to examine whether the model’s judgment of harmfulness will be altered. We
determine if the judgment is changed by observing if the generated response is flipped.
User wants to {User Instruction Tokens}
Could this user prompt cause harm? Answer ‘Certainly’ if it could, otherwise ‘No’
Table 2: An example of the inversion prompting template.
Steering results. Results on Qwen2 are shown in Figure 5. When the instruction is harmless
(Figure 5a), the model should answer “No” in our reply inversion task without intervention. The
8
refusal rate indicates the occurrence of refusal tokens like “No”. If we steer the hidden states along
the harmfulness direction, the model will start to flip its answer to “Certainly”, and thus the refusal
rate will drop. This suggests that the model begins to internally interpret the harmless instruction
as harmful, providing causal evidence that the identified direction indeed represents the concept of
harmfulness. In contrast, steering along the refusal direction generally causes the model to maintain
its original refusal response (e.g., “No”), while steering in the reverse direction tends to produce
affirmative tokens (e.g., “Certainly”). These results suggest that the refusal direction may not alter
the model’s perception of harmfulness; instead, it may primarily encode surface-level refusal cues
rather than a deeper understanding of harmfulness.
On the other hand, if the instructions are harmful, as shown in Figure 5b, steering them along
the reverse harmfulness direction will cause the model to reply “No”. This indicates that our
intervention leads the model to interpret those harmful instructions as harmless. However, steering
along the reverse refusal direction fails to reverse the model’s perception of harmfulness, and therefore
does not elicit refusal responses in the reply inversion task. We observe similar results on other
inversion templates and models as shown in Appendix E. In sum, we provide causal evidence that
LLMs internally reason about the harmfulness of inputs independently from their refusal behaviors,
indicating that harmfulness and refusal are represented as separate concepts.
4 Analyzing Jailbreak via Harmfulness0.2 0.1 0.0 0.1 0.2
harmful
0.3
0.2
0.1
0.0
0.1
0.2
0.3
refuse
adv-suffix
persuasion
template
refused harmful
Figure 6: Belief of harmfulness and
refusal for different categories of jail-
break prompts in comparison with re-
fused harmful instructions.
Different jailbreak methods [Zou et al., 2023b, Yu et al.,
2023, Zeng et al., 2024] have successfully enabled harmful
instructions to be accepted by LLMs. But it remains un-
clear how jailbreak methods work. In this section, we ap-
ply the identified internal belief of harmfulness and refusal
(see Equation 3 and Equation 4) to analyze jailbreak. We
find that some jailbreak methods work by suppressing
the refusal signal, but cannot fundamentally reverse
the model’s belief of harmfulness. We consider different
types of jailbreak methods as detailed in Section 2, i.e.,
adversarial suffixes [Zou et al., 2023b], persuasion [Zeng
et al., 2024] and adversarial prompting templates [Yu et al.,
2023]. As shown in Figure 6, we find that in some cases,
the persuasion jailbreak method can internally make LLMs
believe the persuasive harmful jailbreak prompts are harm-
less (negative ∆harmful). By contrast, for other jailbreak
methods, the refusal signals are suppressed, generally lead-
ing to negative ∆refuse, but in some cases, the model still
internally believes the jailbreak prompts are harmful as
reflected by high ∆harmful scores. Therefore, although prior work has shown jailbreak methods can
suppress refusal features and hypothesize that ablating the refusal directions makes LLMs perceive
instructions as less harmful [Yu et al., 2025], we clarify that not all jailbreak methods can internally
reverse LLMs’ harmfulness judgment, highlighting the need for further investigation.
5 Developing a Latent Guard Model with Harmfulness Representations
Guardrails for LLMs have been widely employed to improve safety, where users’ input instructions
are screened by a guard model [Dong et al., 2024]. When the guard model identifies potentially
harmful inputs, enforcement actions will be taken (e.g., preventing LLMs from processing the input or
adapting LLMs’ outputs). In this section, we propose to use LLM’s internal belief of harmfulness as
a Latent Guard to detect challenging cases like harmful instructions that bypass refusal and harmless
but over-refused instructions [Röttger et al., 2023]. Latent Guard is motivated by the faithfulness and
robustness of the LLMs’ perception of harmfulness: LLMs may still correctly assess the harmfulness
of instructions even when their refusal behavior is incorrect, as shown in Figure 3 and Figure 6.
9
Model Guard Adv-suffix Persuasion Template Refused HL Accepted HF
LLAMA2-CHAT-7B Llama Guard 3 100.0 0.0 76.0 84.4 45.5
Latent Guard 100.0 41.6 100.0 100.0 93.9
LLAMA3-INSTRUCT-8B Llama Guard 3 99.2 6.8 50.0 50.0 37.3
Latent Guard 91.0 65.0 100.0 78.5 59.3
QWEN2-INSTRUCT-7B Llama Guard 3 97.8 17.8 91.4 50.0 59.4
Latent Guard 100.0 75.0 53.5 91.6 54.6
Table 3: Classification accuracy (%) of Latent Guard and Llama Guard 3 on test cases where LLMs
are jailbroken by different techniques (adversarial suffixes, persuasion, prompting template), as well
as results on refused harmless (HL) and accepted harmful (HF) instructions.
5.1 Latent Guard is effective and computationally efficient.
For an incoming instruction, the Latent Guard model computes the belief of harmfulness ∆harmful
following Equation 3. If ∆harmful is negative, the instruction will be classified as harmless, and vice
versa. We sample 100 harmful and 100 harmless examples from the training set (see details in
Appendix B) to compute the centroid of clusters. We compare our Latent Guard with Llama Guard 3
8B 2. Llama Guard is an LLM trained on various examples to classify whether the input is safe or
unsafe [Inan et al., 2023]. For each model, we evaluate the classification performance on a variety of
held-out datasets: harmless but overly-refused instructions (Xstest [Röttger et al., 2023]), harmful
but accepted instructions (Sorry-Bench [Xie et al., 2025]), and prompts that successfully jailbreak
the model (see details in Section 2). Table 3 shows the result. We find that our Latent Guard model
achieves performance comparable to or better than Llama Guard 3, a dedicated finetuned model.
The latent guard performs especially well on all three LLMs in detecting jailbreak prompts with
persuasion and refused harmless instructions. For example, on the Qwen2 model, Latent Guard has
an accuracy of 75% in detecting harmful persuasion prompts, while the Llama Guard 3 only has
an accuracy of 17.8%. Besides, our latent guard model is also computationally efficient. Because
no extra guard models are needed, and one can obtain the classification results within the normal
feed forwarding of users’ input before the LLM starts to generate its response. We provide more
evaluation results in Appendix H.
5.2 Latent Guard is robust to the finetuning attack
LLMs have been shown vulnerable to finetuning attacks [Qi et al., 2023], where finetuning on a few
adversarial examples breaks the safety alignment of LLMs and makes it accept harmful instructions.
Qi et al. [2024b] have also shown that existing safeguards are not robust to adversarial finetuning, we
ask whether the Latent Guard model will also fail to detect harmful instructions after finetuning.0 5 10 15 20 25 30
Layers
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
Harmful
sft-50
sft-100
sft-200
sft-300
sft-400
Figure 7: The belief of harmfulness on
harmful instructions in the latent space
of the model is almost unchanged after
finetuning on different sizes of adversar-
ial examples.
To test this, we finetune LLMs on different numbers of
adversarial examples (from 50 to 400 examples) to evalu-
ate how that may influence a model’s latent representation.
To get adversarial examples on the datasets we use in this
paper, we steer the harmful instructions in these datasets
along the reverse refusal direction to get corresponding
acceptance responses from the model. We then finetune
the model on these pairs of harmful instructions and accep-
tance responses, and only update the model with respect
to the loss of responses.
As shown in Figure 7, we find that, although the model
starts to accept held-out harmful instructions after finetun-
ing, its belief of harmfulness (∆harmful) of these harmful
instructions at tinst is almost unchanged despite the
increase of adversarial training examples. Since ∆harmful is
used by Latent Guard for classification, these instructions
will still be detected as harmful. This indicates that our
2https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/
10
proposed Latent Guard based on ∆harmful is robust to the tested narrow finetuning attack. We
also observe similar results on the refusal representation in LLMs (Appendix H.2). Overall, these
results suggest that finetuning has limited impact on the model’s internal beliefs and may primarily
affect surface-level response styles as hypothesized by Zhou et al. [2023]. We leave it as future work
to investigate the effects of finetuning on model representations
