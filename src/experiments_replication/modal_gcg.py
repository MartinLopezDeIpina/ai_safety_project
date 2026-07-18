"""Run the nanoGCG jailbreak attack (qwen35_jailbreak/gcg_attack.py) against Qwen3.5-9B on Modal.

Modeled on modal_run.py: same debian_slim image + requirements + the mandatory Blackwell cu130 torch
override (B300 = sm_103), plus nanoGCG. One B300 container optimizes adversarial suffixes over the
first N advbench behaviors and writes one JSONL line per attempt to the `gcg-results` volume, flushing
+ committing after each behavior so a detached/timed-out run still yields whatever was found.

Usage (from the repo root):

  # Phase A — prove one works (blocking smoke test)
  modal run src/experiments_replication/modal_gcg.py \
      --mode gennothink --n-behaviors 2 --target-count 1 --gpu B300 --timeout 5400

  # Phase B — scale to >=10, detached
  modal run --detach src/experiments_replication/modal_gcg.py \
      --mode gennothink --n-behaviors 15 --target-count 10 --gpu B300 --timeout 18000 --no-wait
  # ...later:
  modal run src/experiments_replication/modal_gcg.py --mode gennothink --collect-only
"""
import os
import subprocess
import sys
import tempfile

import modal

app = modal.App("qwen35-gcg")

_HERE = os.path.dirname(os.path.abspath(__file__))            # src/experiments_replication
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root (has src/ and data/)
_LOCAL_RESULTS = os.path.join(_HERE, "qwen35_jailbreak", "results")
_PULL_SCRATCH = os.path.join(_HERE, ".pull_scratch")


def _read_hf_token():
    """Parse HF_TOKEN from repo-root .env at build time (None inside the container, where no .env
    exists). Qwen3.5 is ungated, so this is only future-proofing — mirrors modal_run.py."""
    p = os.path.join(_ROOT, ".env")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


_hf_token = _read_hf_token()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token} if _hf_token else {})

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(os.path.join(_ROOT, "requirements.txt"))
    # Blackwell (B200/B300 = sm_103) needs a CUDA-13 torch; the requirements' 2.6.0+cu121 only ships
    # to sm_90 and fails "no kernel image available". Identical override to modal_run.py:61-62.
    .pip_install("torch==2.12.1+cu130",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    # nanoGCG is not in requirements.txt — add it here. --no-deps is REQUIRED: nanogcg's PyPI metadata
    # otherwise drags in an older transformers that downgrades the pinned 5.13.0, after which the
    # container no longer recognizes Qwen3.5's native `qwen3_5` architecture. Its real runtime deps
    # (torch, transformers, tqdm) are already provided by requirements.txt, so --no-deps is safe.
    # If nanogcg's own code turns out incompatible with transformers 5.x, swap PyPI for the git head:
    #   .pip_install("git+https://github.com/GraySwanAI/nanoGCG", extra_options="--no-deps")
    # nanogcg's non-transformers deps that requirements.txt doesn't already provide (it pins
    # transformers>=4.4,<=4.47.1 — the exact thing we must NOT let it install). protobuf/torch/tqdm
    # are already present; scipy (spearmanr, imported at module load) and sentencepiece are not.
    .pip_install("scipy", "sentencepiece")
    .pip_install("nanogcg", extra_options="--no-deps")
    .add_local_dir(
        os.path.join(_ROOT, "src"), "/root/src",
        ignore=["**/output", "**/.pull_scratch", "**/__pycache__", "**/*.pyc", "*.pt", "**/models",
                "run", ".venv"],
    )
    .add_local_dir(os.path.join(_ROOT, "data"), "/root/data")
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
gcg_results = modal.Volume.from_name("gcg-results", create_if_missing=True)


@app.function(image=image, gpu="A100", volumes={"/cache": hf_cache, "/results": gcg_results},
              secrets=[hf_secret], timeout=18000)
def run_gcg(mode: str = "gennothink", n_behaviors: int = 15, target_count: int = 10,
            dataset: str = "advbench.json", num_steps: int = 250, search_width: int = 512,
            topk: int = 256, seed: int = 42, max_new_tokens: int = 200,
            verbosity: str = "INFO", run_name: str = "", early_stop: bool = False,
            use_jb_prompt: bool = False, cot_target: bool = False,
            close_think: bool = False, left: int = 0) -> str:
    """Optimize suffixes and persist to /results/<run_name>_suffixes.jsonl (committed per behavior).
    run_name defaults to mode; give distinct names to run several experiments in parallel without
    clobbering each other on the volume. Returns the results filename."""
    import traceback

    os.environ["HF_HUB_CACHE"] = "/cache"
    os.environ["PYTHONPATH"] = "/root/src"
    sys.path.insert(0, "/root/src/experiments_replication/qwen35_jailbreak")
    sys.path.insert(0, "/root/src")
    os.chdir("/root/src/experiments_replication/qwen35_jailbreak")

    import torch
    print(f"### GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
          f"| torch {torch.__version__} | arch {torch.cuda.get_arch_list() if torch.cuda.is_available() else '-'}",
          flush=True)

    out_name = f"{run_name or mode}_suffixes.jsonl"
    out_path = f"/results/{out_name}"
    try:
        from gcg_attack import run_attack
        run_attack(
            out_path=out_path,
            thinking_mode=mode,
            dataset=dataset,
            max_attempts=n_behaviors,
            target_count=target_count,
            num_steps=num_steps,
            search_width=search_width,
            topk=topk,
            seed=seed,
            max_new_tokens=max_new_tokens,
            verbosity=verbosity,
            early_stop=early_stop,
            use_jb_prompt=use_jb_prompt,
            cot_target=cot_target,
            close_think=close_think,
            left=left,
            commit_fn=gcg_results.commit,  # flush each behavior to the volume as it completes
        )
    except Exception:
        print("### attack error — committing partial results:", flush=True)
        traceback.print_exc()
    finally:
        if os.path.exists(out_path):
            gcg_results.commit()
    return out_name


@app.function(image=image, gpu="A100", volumes={"/cache": hf_cache},
              secrets=[hf_secret], timeout=1800)
def check_suffix(suffix: str, query: str = "", mode: str = "gennothink",
                 use_jb_prompt: bool = False, max_new_tokens: int = 256) -> dict:
    """Load the (cached) model and test one already-optimized suffix; returns the verdict dict."""
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.environ["PYTHONPATH"] = "/root/src"
    sys.path.insert(0, "/root/src/experiments_replication/qwen35_jailbreak")
    sys.path.insert(0, "/root/src")
    os.chdir("/root/src/experiments_replication/qwen35_jailbreak")
    from gcg_attack import verify_suffix
    return verify_suffix(suffix, query=query or None, thinking_mode=mode,
                         use_jb_prompt=use_jb_prompt, max_new_tokens=max_new_tokens)


@app.local_entrypoint()
def check(suffix: str, query: str = "", mode: str = "gennothink",
          use_jb_prompt: bool = False, max_new_tokens: int = 256, gpu: str = "A100"):
    """Manually check whether a given suffix jailbreaks a behavior. Blocking.
        modal run modal_gcg.py::check --suffix '<suffix>' --mode gennothink
    """
    res = check_suffix.with_options(gpu=gpu).remote(suffix, query, mode, use_jb_prompt, max_new_tokens)
    print("\n================ SUFFIX CHECK ================")
    print("query     :", res["query"])
    print("mode      :", res["thinking_mode"], "| jb_prompt:", res["use_jb_prompt"])
    print("suffix    :", res["suffix"])
    print(f"retok mismatch (optimize-style ids != joined-string ids): {res['retok_mismatch']} "
          f"(opt={res['n_tokens_opt']} toks, join={res['n_tokens_join']} toks)")
    print("\n--- (A) OPTIMIZE-STYLE construction (matches the loss) ---")
    print("VERDICT:", "ACCEPTED ✅" if res["accepted"] else "REFUSED ❌")
    print(res["generation"])
    print("\n--- (B) JOINED-STRING construction (the old/naive way) ---")
    print("VERDICT:", "ACCEPTED ✅" if res["accepted_joined"] else "REFUSED ❌")
    print(res["generation_joined"])
    print("=============================================")


def _pull(out_name: str):
    """Fetch /<out_name> from the gcg-results volume into qwen35_jailbreak/results/."""
    os.makedirs(_LOCAL_RESULTS, exist_ok=True)
    os.makedirs(_PULL_SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="pull_gcg_", dir=_PULL_SCRATCH)
    modal_cli = [sys.executable, "-m", "modal"]
    import shutil
    try:
        # check=False: a run that errored before writing any record leaves no file on the volume;
        # that should surface as a warning, not crash the entrypoint (the container logs hold the error).
        r = subprocess.run(modal_cli + ["volume", "get", "gcg-results", f"/{out_name}", tmp, "--force"])
        if r.returncode != 0:
            print(f"WARNING: nothing to pull for /{out_name} (the run may have produced no results).")
            return
        dst = os.path.join(_LOCAL_RESULTS, out_name)
        shutil.move(os.path.join(tmp, out_name), dst)
        print(f"pulled -> {dst}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.local_entrypoint()
def entry(mode: str = "gennothink",
          n_behaviors: int = 15,
          target_count: int = 10,
          dataset: str = "advbench.json",
          num_steps: int = 250,
          search_width: int = 512,
          topk: int = 256,
          seed: int = 42,
          max_new_tokens: int = 200,
          verbosity: str = "INFO",
          run_name: str = "",
          early_stop: bool = False,
          use_jb_prompt: bool = False,
          cot_target: bool = False,
          close_think: bool = False,
          left: int = 0,
          gpu: str = "B300",
          timeout: int = 18000,
          wait: bool = True,
          collect_only: bool = False):
    """Launch (and optionally collect) the GCG attack.

    Interactive (Phase A):
        modal run modal_gcg.py --mode gennothink --n-behaviors 2 --target-count 1 --gpu B300 --timeout 5400
    Detached (Phase B, safe to close the laptop):
        modal run --detach modal_gcg.py --mode gennothink --n-behaviors 15 --target-count 10 --no-wait
        # ...later:
        modal run modal_gcg.py --mode gennothink --collect-only
    Parallel experiments: give each a distinct --run-name so their volume files don't collide,
    and collect with the same --run-name.
    """
    out_name = f"{run_name or mode}_suffixes.jsonl"
    if collect_only:
        _pull(out_name)
        return

    fn = run_gcg.with_options(gpu=gpu, timeout=timeout)
    handle = fn.spawn(mode, n_behaviors, target_count, dataset, num_steps,
                      search_width, topk, seed, max_new_tokens, verbosity,
                      run_name, early_stop, use_jb_prompt, cot_target, close_think, left)
    if not wait:
        print(f"launched detached; collect later with:\n"
              f"  modal run src/experiments_replication/modal_gcg.py "
              f"--run-name {run_name or mode} --collect-only")
        return
    name = handle.get()
    _pull(name)
