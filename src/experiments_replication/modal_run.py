"""
Run the replication pipeline on Modal (A100) for whatever model config.py selects
(Qwen2-7B by default; uncomment the Llama-3-8B block in config.py to switch). Gated models
authenticate via an HF_TOKEN Secret read from the repo-root .env (see _read_hf_token) —
unused for the ungated Qwen default. Token positions adapt to the model automatically
(config.POST_INST_SUFFIX / model_utils.get_token_positions), so no other change is needed.

Path convention (unchanged from the original stage-00 runner): every script is flattened
into /root, so config.py sits at /root/config.py → REPO_ROOT="/" → DATA_DIR="/data"
(the behavior-datasets volume) and RESULTS_DIR="/root/results" (the behavior-results
volume). The src/ utilities (utils, inference, extract_hidden, intervention, eval,
template_inversion) are copied next to them so `import intervention` etc. resolve from
/root regardless of config.SRC_DIR.

Two remote functions:
  gen_jailbreaks   — generate data/gcg-advsuffix.json + data/pap-persuasion.json ON the
                     target model and commit them to the datasets volume (run once; slow).
  run_pipeline     — run the numbered stages as subprocesses, commit results.

Usage:
  modal run modal_run.py                     # gen datasets (if missing) + full pipeline
  modal run modal_run.py --stages 04,05,06,07,08,08b   # rerun a subset (datasets/acts cached)
  modal run modal_run.py --gen-only          # only (re)generate the jailbreak datasets
  modal run modal_run.py --skip-gen          # assume datasets already on the volume
"""
import os
import sys

import modal

app = modal.App("ai_safety")

_REPL = os.path.dirname(os.path.abspath(__file__))          # .../src/experiments_replication
_SRC = os.path.abspath(os.path.join(_REPL, ".."))           # .../src

# src/ modules the numbered scripts import (copied flat into /root)
_SRC_MODULES = ["utils.py", "inference.py", "extract_hidden.py",
                "intervention.py", "eval.py", "template_inversion.py"]


def _read_hf_token():
    """Parse HF_TOKEN from the repo-root .env so the GPU functions can download a GATED
    model (e.g. meta-llama/Meta-Llama-3-8B-Instruct when config.py is switched to Llama-3).
    Harmless/unused for the default Qwen models (ungated).

    Read only on the LOCAL machine at app-build time and bound into the submitted app; this
    module is re-imported inside the container (where no .env exists), so returns None there
    instead of raising. Handles both the main checkout and a .claude/worktrees/<name> checkout
    (the worktree has no .env of its own — fall back to the main root)."""
    root = os.path.abspath(os.path.join(_REPL, "..", ".."))          # repo/worktree root
    candidates = [root]
    if "/.claude/worktrees/" in root:
        candidates.append(root.split("/.claude/worktrees/")[0])      # main checkout root
    for d in candidates:
        p = os.path.join(d, ".env")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# Gated-model auth: real token bound locally at build; empty placeholder in-container.
_hf_token = _read_hf_token()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token} if _hf_token else {})

image = (
    modal.Image.debian_slim()
    # transformers pinned to 4.57.6 = the version the extract_hidden fwd-hook + intervention
    # were validated against (CLAUDE.md). nanogcg is intentionally NOT installed here: it is
    # imported lazily only inside gen_gcg, and pinning transformers==4.57.6 conflicts with
    # nanogcg's newer requirement. Regenerating datasets uses a separate (unpinned) launch.
    .pip_install("torch", "transformers==4.57.6", "accelerate", "numpy", "matplotlib",
                 "sentencepiece", "protobuf", "openai", "tqdm")
    # Flatten the replication scripts into /root (skip local outputs / caches).
    .add_local_dir(
        _REPL, "/root",
        ignore=["results*", "qwen*", "qweb*", "__pycache__", "*.pyc", "logs",
                "slurm", "responses_judging", "*.pt", "*.png", ".venv"],
        copy=True,
    )
)
# Copy the needed src/ modules next to them.
for _m in _SRC_MODULES:
    image = image.add_local_file(os.path.join(_SRC, _m), f"/root/{_m}", copy=True)

hf_cache    = modal.Volume.from_name("hf-hub-cache",      create_if_missing=True)
results_vol = modal.Volume.from_name("behavior-results",  create_if_missing=True)
data_vol    = modal.Volume.from_name("behavior-datasets", create_if_missing=True)

VOLUMES = {"/cache": hf_cache, "/root/results": results_vol, "/data": data_vol}

# Stage order. 08b (Llama Guard baseline) runs last and is optional/gated.
DEFAULT_STAGES = [
    "_00_collect_behaviors.py",
    "01_extract_directions.py",
    "02_table1_post_inst_tokens.py",
    "03_figure2_clustering.py",
    "04_figure3_scatter.py",
    "05_figure4_steering_layers.py",
    "06_figure5_reply_inversion.py",
    "07_figure6_jailbreak.py",
    "08_latent_guard.py",
    "08b_llama_guard_baseline.py",
]
# Map short tokens ("00","01",…,"08","08b") to filenames for --stages.
_SHORT = {s.lstrip("_").split("_")[0]: s for s in DEFAULT_STAGES}  # "04_x.py"→"04", "_00_x.py"→"00"


def _run(stage):
    import subprocess
    print(f"\n{'='*70}\n### {stage}\n{'='*70}", flush=True)
    subprocess.run([sys.executable, f"/root/{stage}"], cwd="/root", check=True)


@app.function(image=image, volumes=VOLUMES, timeout=600)
def check():
    """CPU-only preflight: verify the /root flattening + volume mounts resolve BEFORE
    spending A100 time. Checks config paths, importability of every stage's deps, and that
    the source datasets are on the volume."""
    os.chdir("/root")
    sys.path.insert(0, "/root")
    import importlib
    import config as cfg
    print("REPO_ROOT   :", cfg.REPO_ROOT)
    print("DATA_DIR    :", cfg.DATA_DIR, "exists:", os.path.isdir(cfg.DATA_DIR))
    print("RESULTS_DIR :", cfg.RESULTS_DIR, "exists:", os.path.isdir(cfg.RESULTS_DIR))
    assert cfg.DATA_DIR == "/data", f"DATA_DIR should be /data, got {cfg.DATA_DIR}"
    assert cfg.RESULTS_DIR == "/root/results", f"RESULTS_DIR wrong: {cfg.RESULTS_DIR}"
    # These import fine on CPU:
    for m in ["model_utils", "utils", "intervention", "eval", "template_inversion",
              "gen_jailbreaks"]:
        importlib.import_module(m)
        print("  import OK:", m)
    # inference.py / extract_hidden.py raise "CUDA is required" at import — only valid on
    # the GPU function; here just confirm the files are present.
    for m in ["inference.py", "extract_hidden.py"]:
        assert os.path.exists(f"/root/{m}"), f"missing {m}"
        print("  file present (GPU-only import):", m)
    for f in ["advbench.json", "persuasion_taxonomy.jsonl", "GPTFuzzer-50-adv.json",
              "sorry-badq.json", "jbb.json", "alpaca_data_instruction.json"]:
        print(f"  /data/{f}:", os.path.exists(f"/data/{f}"))
    for s in DEFAULT_STAGES:
        assert os.path.exists(f"/root/{s}"), f"missing stage file {s}"
    print("PREFLIGHT OK — all stage files present, paths and imports resolve.")


@app.function(image=image, gpu="A100", volumes=VOLUMES, secrets=[hf_secret], timeout=18000)
def gen_jailbreaks(gcg_n: int = 20, gcg_steps: int = 80, pap_n: int = 50):
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.chdir("/root")
    sys.path.insert(0, "/root")
    from gen_jailbreaks import main as gen_main
    data_vol.reload()   # pick up any partial datasets from a preempted run, to resume
    # GCG ≈ 4 s/step on the A100, so keep GCG small (early-stop halts on success); PAP is a
    # cheap LLM paraphrase, so generate more to ensure enough ACCEPTED persuasion jailbreaks.
    # commit_fn persists each suffix immediately so a worker PREEMPTION resumes, not restarts.
    gen_main("both", gcg_n=gcg_n, gcg_steps=gcg_steps, pap_n=pap_n, commit_fn=data_vol.commit)
    data_vol.commit()
    print("Committed generated jailbreak datasets to behavior-datasets volume.")


# Per-stage sentinel output (in RESULTS_DIR). If present, the stage is skipped on resume so a
# preempted pipeline continues instead of re-running stage 00's ~30-min inference. 08b has no
# sentinel (it re-merges into table3.json, idempotent) so it always runs when reached.
STAGE_SENTINEL = {
    "_00_collect_behaviors.py":      "behaviors.json",
    "01_extract_directions.py":      "dir-hf.pt",
    "02_table1_post_inst_tokens.py": "table1.json",
    "03_figure2_clustering.py":      "figure2-data.json",
    "04_figure3_scatter.py":         "figure3-data.json",
    "05_figure4_steering_layers.py": "figure4-data.json",
    "06_figure5_reply_inversion.py": "figure5-data.json",
    "07_figure6_jailbreak.py":       "figure6-data.json",
    "08_latent_guard.py":            "table3.json",
}


@app.function(image=image, volumes=VOLUMES, timeout=600)
def clear_results():
    """One-shot wipe of the results volume so a fresh pipeline run does not resume-skip stale
    artifacts from a previous run. Called ONCE by the entrypoint (never on preemption-retry, so
    in-run resume still works)."""
    import glob
    results_vol.reload()
    removed = 0
    for p in glob.glob("/root/results/*"):
        if os.path.isfile(p):
            os.remove(p); removed += 1
    results_vol.commit()
    print(f"Cleared {removed} stale files from behavior-results.")


@app.function(image=image, gpu="A100", volumes=VOLUMES, secrets=[hf_secret], timeout=18000)
def run_pipeline(stages: list[str], resume: bool = True):
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.chdir("/root")
    data_vol.reload()     # pick up gcg/pap datasets committed by a prior gen_jailbreaks run
    results_vol.reload()
    for stage in stages:
        sentinel = STAGE_SENTINEL.get(stage)
        if resume and sentinel and os.path.exists(f"/root/results/{sentinel}"):
            print(f"\n### SKIP {stage} — {sentinel} already exists (resume)")
            continue
        _run(stage)
        results_vol.commit()   # persist after each stage so a preemption resumes here
    print("Committed results to behavior-results volume.")


def _datasets_present():
    """True if both generated datasets already exist on the mounted data volume."""
    return all(os.path.exists(f"/data/{f}")
               for f in ("gcg-advsuffix.json", "pap-persuasion.json"))


@app.local_entrypoint()
def entry(stages: str = "", gen_only: bool = False, skip_gen: bool = False,
          fresh: bool = False, gcg_n: int = 20, gcg_steps: int = 80, pap_n: int = 50):
    # 1) (re)generate the GCG + PAP jailbreak datasets on the target model, unless skipped.
    #    NOTE: the datasets volume must already contain the source data (advbench.json and
    #    persuasion_taxonomy.jsonl) — run `python modal_data_setup.py` after adding the
    #    taxonomy so the volume is up to date before the first gen.
    if not skip_gen:
        gen_jailbreaks.remote(gcg_n=gcg_n, gcg_steps=gcg_steps, pap_n=pap_n)
    if gen_only:
        return

    # 2) optionally wipe stale results ONCE (so resume-skip is correct for a fresh run), then
    #    run the (sub)set of stages (resume-skips any stage whose output already exists).
    if fresh:
        clear_results.remote()
    if stages.strip():
        chosen = [_SHORT.get(tok.strip(), tok.strip()) for tok in stages.split(",")]
    else:
        chosen = DEFAULT_STAGES
    run_pipeline.remote(chosen)

    # 3) pull results (and datasets) back locally
    import subprocess
    subprocess.run(["modal", "volume", "get", "behavior-results", "/",
                    "./src/experiments_replication/results", "--force"])
    subprocess.run(["modal", "volume", "get", "behavior-datasets", "/gcg-advsuffix.json",
                    "./data/gcg-advsuffix.json", "--force"])
    subprocess.run(["modal", "volume", "get", "behavior-datasets", "/pap-persuasion.json",
                    "./data/pap-persuasion.json", "--force"])
