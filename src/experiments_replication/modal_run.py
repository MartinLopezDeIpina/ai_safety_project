"""Run the replication pipeline (src/experiments_replication/main.py) on Modal.

Each --runs entry launches its own A100 container that runs main() for one model/config,
then copies the produced output/<model><size>/ tree onto a results volume. The local
entrypoint pulls every finished tree back into src/experiments_replication/output/,
auto-indexing (<name>_2, _3, …) so it never clobbers a previous local run.

Usage (from the repo root):
  modal run src/experiments_replication/modal_run.py --runs "qwen:0.5b:0:10"
  modal run src/experiments_replication/modal_run.py --runs "qwen:0.5b:0:10,qwen:7b:0:500"

Each run token is  model:size[:left[:right]]  (left defaults 0, right defaults 100000 = full).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import modal

app = modal.App("harmfulness-refusal")

_HERE = os.path.dirname(os.path.abspath(__file__))            # src/experiments_replication
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root (has src/ and data/)
_LOCAL_OUTPUT = os.path.join(_HERE, "output")


def _read_hf_token():
    """Parse HF_TOKEN from the repo-root .env at build time so a (future) gated model can be
    downloaded. Read only on the local machine; this module is re-imported in the container
    where no .env exists, so it returns None there instead of raising. Ungated for the current
    Qwen/Llama-3 configs."""
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

# Reproduce the local repo layout under /root so the scripts' HERE/SRC_DIR/DATA_DIR and their
# cwd-relative subprocess calls resolve unchanged: REPO_DIR=/root, SRC_DIR=/root/src,
# HERE=/root/src/experiments_replication, DATA_DIR=/root/data.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(os.path.join(_ROOT, "requirements.txt"))
    # Mounted at runtime (copy=False): no build step needs these files, and copying them into
    # an image layer would abort with "modified during build process" if a file's mtime shifts
    # mid-copy (an editor/linter touching a .py, or .pyc regeneration on entrypoint import).
    .add_local_dir(
        os.path.join(_ROOT, "src"), "/root/src",
        ignore=["**/output", "**/__pycache__", "**/*.pyc", "*.pt", "run", ".venv"],
    )
    .add_local_dir(os.path.join(_ROOT, "data"), "/root/data")
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("behavior-results", create_if_missing=True)


def _name(model: str, model_size: str) -> str:
    return f"{model}{model_size}"


def _cfg_stem(config: str) -> str:
    """Basename of a config path without extension, e.g. 'bucket_config.json' -> 'bucket_config'."""
    return os.path.splitext(os.path.basename(config))[0] if config else "default"


def _key(model: str, model_size: str, left: int, right: int, config: str = "") -> str:
    """Deterministic results-volume key for one run, so a detached run can be collected later and
    two configs of the same model (differing left/right or bucket config) don't collide."""
    return f"{model}{model_size}-{left}-{right}-{_cfg_stem(config)}"


@app.function(image=image, gpu="A100", volumes={"/cache": hf_cache, "/results": results_vol},
              secrets=[hf_secret], timeout=18000)
def run_one(model: str, model_size: str, left: int, right: int, stages: str,
            config: str) -> tuple[str, str]:
    """Run main() for one model/config and persist its output tree to /results/<key>. Returns
    (name, key). Output is committed in a `finally` so a late-stage error (e.g. figure3) can't
    discard an otherwise-complete run — whatever main() produced is still saved."""
    import traceback
    os.environ["HF_HUB_CACHE"] = "/cache"
    # extract_hidden.py runs `from utils import ...` with cwd=experiments_replication, so the
    # subprocess needs src/ on PYTHONPATH (it inherits os.environ).
    os.environ["PYTHONPATH"] = "/root/src"
    sys.path.insert(0, "/root/src/experiments_replication")
    sys.path.insert(0, "/root/src")
    os.chdir("/root/src/experiments_replication")

    name, key = _name(model, model_size), _key(model, model_size, left, right, config)
    try:
        from main import main
        main(model, model_size, left, right, stages=tuple(stages.split(",")),
             bucket_config=config)
    except Exception:
        print(f"### stage error in {key} — committing partial output:", flush=True)
        traceback.print_exc()
    finally:
        produced = f"/root/src/experiments_replication/output/{name}"
        if os.path.isdir(produced):
            shutil.copytree(produced, f"/results/{key}", dirs_exist_ok=True)
            results_vol.commit()
    return name, key


def _parse_runs(runs: str):
    """"qwen:0.5b:0:10,qwen:7b" -> [("qwen","0.5b",0,10),("qwen","7b",0,100000)]."""
    specs = []
    for token in runs.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(f"bad run spec {token!r}; expected model:size[:left[:right]]")
        model, size = parts[0], parts[1]
        left = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        right = int(parts[3]) if len(parts) > 3 and parts[3] else 100000
        specs.append((model, size, left, right))
    return specs


def _local_dest(name: str) -> str:
    """Non-colliding local output dir: name, else name_2, name_3, …"""
    dest = os.path.join(_LOCAL_OUTPUT, name)
    if not os.path.exists(dest):
        return dest
    i = 2
    while os.path.exists(os.path.join(_LOCAL_OUTPUT, f"{name}_{i}")):
        i += 1
    return os.path.join(_LOCAL_OUTPUT, f"{name}_{i}")


def _pull(name: str, key: str):
    """Fetch /<key>/ from the results volume into a fresh (indexed) local output dir named
    after `name` (name, else name_2, name_3, …)."""
    dest = _local_dest(name)
    os.makedirs(_LOCAL_OUTPUT, exist_ok=True)
    # Download into a scratch dir OUTSIDE output/ so a failure never leaves debris there.
    # `modal volume get vol /<key> <dir>` requires <dir> to exist and drops the tree at
    # <dir>/<key>/... — we then move just that tree into a clean output/<name>/.
    tmp = tempfile.mkdtemp(prefix=f"pull_{key}_")
    modal_cli = [sys.executable, "-m", "modal"]  # `modal` may not be on PATH
    try:
        subprocess.run(modal_cli + ["volume", "get", "behavior-results", f"/{key}", tmp, "--force"],
                       check=True)
        shutil.move(os.path.join(tmp, key), dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(modal_cli + ["volume", "rm", "behavior-results", f"/{key}", "-r"], check=False)
    print(f"pulled -> {dest}")


@app.local_entrypoint()
def entry(runs: str = "qwen:0.5b:0:10",
          stages: str = "infer,eval,acts,gen_buckets,fig,fig3",
          config: str = "bucket_config.json",
          wait: bool = True,
          collect_only: bool = False):
    """Launch (and optionally collect) the runs in `runs`.

    Interactive:  modal run modal_run.py --runs "..." --config bucket_config.json
        spawns every run in parallel, waits, and pulls each result locally. Launch different
        bucket configs in parallel with separate invocations (each --config gets its own key).

    Detached (safe to close the laptop): run fire-and-forget, collect later.
        modal run --detach modal_run.py --runs "..." --no-wait
        # ...later, when back at the machine:
        modal run modal_run.py --runs "..." --config bucket_config.json --collect-only
    Collect keys are deterministic (model+size+left+right+config), so the same --runs/--config
    retrieves exactly the runs that were launched.
    """
    specs = _parse_runs(runs)

    if collect_only:
        for model, size, left, right in specs:
            name, key = _name(model, size), _key(model, size, left, right, config)
            try:
                _pull(name, key)
            except subprocess.CalledProcessError:
                print(f"  not ready on volume yet (skipped): {key}")
        return

    handles = [(spec, run_one.spawn(*spec, stages, config)) for spec in specs]  # launch in parallel
    if not wait:
        print("\nLaunched detached; results commit to the 'behavior-results' volume as each "
              "run finishes.\nCollect them later with:\n"
              f'  modal run {os.path.relpath(__file__, os.getcwd())} '
              f'--collect-only --runs "{runs}"\n')
        return

    for spec, handle in handles:
        name, key = handle.get()                                 # blocks until this run finishes
        _pull(name, key)
