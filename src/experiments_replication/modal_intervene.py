"""Run intervention experiments (src/intervention.py) on Modal.

Each --runs entry launches its own A100 container that runs intervention.py for one
model/dataset/vector config, then copies the produced output JSON file(s) onto a results
volume. The local entrypoint pulls every finished file back into
src/experiments_replication/intervention_outputs/, auto-indexing so it never clobbers a
previous local run.

This is the Modal equivalent of slurm/intervene.slurm.

Usage (from the repo root):
  # Single run (defaults mirror intervene.slurm: advbench + hf vector, reverse=less-harm)
  modal run src/experiments_replication/modal_intervene.py \\
      --runs "qwen:7b:0:50"

  # Multiple datasets / vectors in parallel
  modal run src/experiments_replication/modal_intervene.py \\
      --runs "qwen:7b:0:50,qwen:7b:0:50" \\
      --datasets "advbench,jbb" \\
      --vectors "hf,refusal"

Each run token is  model:size[:left[:right]]  (left defaults 0, right defaults 50).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import modal

app = modal.App("harmfulness-refusal-intervene")

_HERE = os.path.dirname(os.path.abspath(__file__))            # src/experiments_replication
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root (has src/ and data/)
_LOCAL_OUTPUT = os.path.join(_HERE, "intervention_outputs")


def _read_hf_token():
    """Parse HF_TOKEN from the repo-root .env at build time so a (future) gated model can be
    downloaded. Read only on the local machine; this module is re-imported in the container
    where no .env exists, so it returns None there instead of raising."""
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

# Reproduce the local repo layout under /root so intervention.py's `from utils import ...`
# and `from template_inversion import ...` (both in src/) resolve, and cwd-relative paths
# like "data/advbench.json" work: REPO_DIR=/root, SRC_DIR=/root/src, DATA_DIR=/root/data.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(os.path.join(_ROOT, "requirements.txt"))
    # Blackwell (B200/B300) needs a CUDA-13 torch: the requirements' torch 2.6.0+cu121 only ships up
    # to sm_90, so B300 (sm_103) fails with "no kernel image available". torch 2.12.1+cu130 ships
    # sm_100 cubins, which run on sm_103 by same-major binary forward-compat. Harmless on A100.
    .pip_install("torch==2.12.1+cu130",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    # Mounted at runtime (copy=False): no build step needs these files, and copying them into
    # an image layer would abort with "modified during build process" if a file's mtime shifts
    # mid-copy. We DO need the steering vectors (.pt) this time, so don't ignore *.pt under
    # experiments_replication/ — but still skip the big model caches and prior outputs.
    .add_local_dir(
        os.path.join(_ROOT, "src"), "/root/src",
        ignore=["**/output", "**/__pycache__", "**/*.pyc", "**/models", "run", ".venv",
                "**/intervention_outputs"],
    )
    .add_local_dir(os.path.join(_ROOT, "data"), "/root/data")
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("intervention-results", create_if_missing=True)


def _name(model: str, model_size: str) -> str:
    return f"{model}{model_size}"


def _key(model: str, model_size: str, dataset: str, vector: str,
         left: int, right: int, reverse: int) -> str:
    """Deterministic results-volume key for one intervention run, so a detached run can be
    collected later and runs with different dataset/vector/reverse don't collide."""
    direction = "lessharm" if reverse else "moreharm"
    return f"{model}{model_size}-{dataset}-{vector}-{direction}-{left}-{right}"


@app.function(
    image=image, gpu="A100", volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=[hf_secret], timeout=7200
)
def run_intervention(
        model: str, model_size: str, left: int, right: int,
        dataset: str, vector: str, reverse_intervention: int,
        intervene_context_only: int, arg_key_prompt: str,
        layer_s: int, layer_e: int, coeff_select: float,
        max_token_generate: int, max_decode_step_while_intervene: int,
        use_inversion: int, inversion_prompt_idx: int
    ) -> tuple[str, str]:
    """
    Run intervention.py for one config and persist its output JSON(s) to /results/<key>.
    Returns (name, key). Output is committed in a `finally` so a late-stage error can't
    discard an otherwise-complete run — whatever intervention.py produced is still saved.
    """
    import traceback
    os.environ["HF_HUB_CACHE"] = "/cache"
    # intervention.py runs `from utils import ...` and `from template_inversion import ...`
    # with both modules in src/, so the subprocess needs src/ on PYTHONPATH.
    os.environ["PYTHONPATH"] = "/root/src"
    os.chdir("/root/src")

    import torch  # log the actual GPU this container got
    print(f"### GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
          f"| torch {torch.__version__} | arch {torch.cuda.get_arch_list() if torch.cuda.is_available() else '-'}",
          flush=True)

    name = _name(model, model_size)
    key = _key(model, model_size, dataset, vector, left, right, reverse_intervention)

    # Resolve paths inside the container.
    test_data_pth = f"/root/data/{dataset}.json"
    vector_pth = f"/root/src/experiments_replication/steering_vectors/{name}/{vector}.pt"
    out_dir = f"/root/src/experiments_replication/intervention_outputs/{key}"
    os.makedirs(out_dir, exist_ok=True)
    output_pth = f"{out_dir}/{dataset}_{'lessharm' if reverse_intervention else 'moreharm'}.json"

    cmd = [
        sys.executable, "-u", "/root/src/intervention.py",
        "--test_data_pth", test_data_pth,
        "--output_pth", output_pth,
        "--intervention_vector", vector_pth,
        "--reverse_intervention", str(reverse_intervention),
        "--intervene_context_only", str(intervene_context_only),
        "--arg_key_prompt", arg_key_prompt,
        "--model", model,
        "--model_size", model_size,
        "--left", str(left),
        "--right", str(right),
        "--layer_s", str(layer_s),
        "--layer_e", str(layer_e),
        "--coeff_select", str(coeff_select),
        "--max_token_generate", str(max_token_generate),
        "--max_decode_step_while_intervene", str(max_decode_step_while_intervene),
        "--use_inversion", str(use_inversion),
        "--inversion_prompt_idx", str(inversion_prompt_idx),
    ]

    print(f"### running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=os.environ)
    except Exception:
        print(f"### intervention error in {key} — committing partial output:", flush=True)
        traceback.print_exc()
    finally:
        # intervention.py writes output_pth with a per-layer suffix, e.g.
        #   <output_pth without .json>-intervene<layer>.json
        # so collect every JSON that landed in out_dir.
        if os.path.isdir(out_dir):
            dest = f"/results/{key}"
            os.makedirs(dest, exist_ok=True)
            for fname in os.listdir(out_dir):
                src_file = os.path.join(out_dir, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(dest, fname))
            results_vol.commit()
    return name, key


def _parse_runs(runs: str):
    """"qwen:7b:0:50,qwen:7b" -> [("qwen","7b",0,50),("qwen","7b",0,50)]."""
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
        right = int(parts[3]) if len(parts) > 3 and parts[3] else 50
        specs.append((model, size, left, right))
    return specs


def _local_dest(name: str, key: str) -> str:
    """Non-colliding local output dir: key, else key_2, key_3, …"""
    dest = os.path.join(_LOCAL_OUTPUT, key)
    if not os.path.exists(dest):
        return dest
    i = 2
    while os.path.exists(os.path.join(_LOCAL_OUTPUT, f"{key}_{i}")):
        i += 1
    return os.path.join(_LOCAL_OUTPUT, f"{key}_{i}")


def _pull(name: str, key: str):
    """Fetch /<key>/ from the results volume into a fresh (indexed) local output dir."""
    dest = _local_dest(name, key)
    os.makedirs(_LOCAL_OUTPUT, exist_ok=True)
    # Download into a scratch dir OUTSIDE intervention_outputs/ so a failure never leaves debris.
    tmp = tempfile.mkdtemp(prefix=f"pull_{key}_")
    modal_cli = [sys.executable, "-m", "modal"]  # `modal` may not be on PATH
    try:
        subprocess.run(modal_cli + ["volume", "get", "intervention-results", f"/{key}", tmp, "--force"],
                       check=True)
        shutil.move(os.path.join(tmp, key), dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"pulled -> {dest}")


@app.local_entrypoint()
def entry(
    runs: str = "qwen:7b:0:50",
    datasets: str = "advbench",
    vectors: str = "hf",
    reverse_intervention: int = 1,
    intervene_context_only: int = 1,
    arg_key_prompt: str = "bad_q",
    layer_s: int = 0,
    layer_e: int = 28,
    coeff_select: float = 1,
    max_token_generate: int = 100,
    max_decode_step_while_intervene: int = 1,
    use_inversion: int = 1,
    inversion_prompt_idx: int = 1,
    gpu: str = "A100",
    timeout: int = 7200,
    wait: bool = True,
    collect_only: bool = False,
):
    """Launch (and optionally collect) intervention runs.

    Interactive (defaults mirror slurm/intervene.slurm — advbench + hf vector, reverse=less-harm):
        modal run src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50"
        spawns the run, waits, and pulls the result JSON(s) locally.

    Multiple datasets / vectors in parallel (comma-separated, zipped with --runs):
        modal run src/experiments_replication/modal_intervene.py \\
            --runs "qwen:7b:0:50,qwen:7b:0:50" \\
            --datasets "advbench,jbb" --vectors "hf,refusal"

    Detached (safe to close the laptop): run fire-and-forget, collect later.
        modal run --detach src/experiments_replication/modal_intervene.py --runs "..." --no-wait
        # ...later, when back at the machine:
        modal run src/experiments_replication/modal_intervene.py --runs "..." --collect-only \\
            --datasets "advbench" --vectors "hf" --reverse-intervention 1

    Collect keys are deterministic (model+size+dataset+vector+direction+left+right), so the
    same --runs/--datasets/--vectors/--reverse-intervention retrieves exactly the runs launched.

    Flip to more-harm (amplify harmfulness) instead of less-harm:
        modal run src/experiments_replication/modal_intervene.py \\
            --runs "qwen:7b:0:50" --reverse-intervention 0
    """
    specs = _parse_runs(runs)
    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    vector_list = [v.strip() for v in vectors.split(",") if v.strip()]

    # Zip specs with datasets/vectors; if fewer datasets/vectors than specs, cycle the last one.
    configs = []
    for i, (model, size, left, right) in enumerate(specs):
        ds = dataset_list[i] if i < len(dataset_list) else dataset_list[-1]
        vec = vector_list[i] if i < len(vector_list) else vector_list[-1]
        configs.append((model, size, left, right, ds, vec))

    if collect_only:
        for model, size, left, right, ds, vec in configs:
            name, key = _name(model, size), _key(model, size, ds, vec, left, right,
                                                 reverse_intervention)
            try:
                _pull(name, key)
            except subprocess.CalledProcessError:
                print(f"  not ready on volume yet (skipped): {key}")
        return

    # per-invocation GPU + timeout override
    fn = run_intervention.with_options(gpu=gpu, timeout=timeout)
    handles = []
    for model, size, left, right, ds, vec in configs:
        h = fn.spawn(
            model, size, left, right, ds, vec,
            reverse_intervention, intervene_context_only, arg_key_prompt,
            layer_s, layer_e, coeff_select,
            max_token_generate, max_decode_step_while_intervene,
            use_inversion, inversion_prompt_idx,
        )
        handles.append((_name(model, size),
                        _key(model, size, ds, vec, left, right, reverse_intervention),
                        h))

    if not wait:
        print("\nLaunched detached; results commit to the 'intervention-results' volume as each "
              "run finishes.\nCollect them later with:\n"
              f'  modal run {os.path.relpath(__file__, os.getcwd())} '
              f'--collect-only --runs "{runs}" --datasets "{datasets}" --vectors "{vectors}" '
              f'--reverse-intervention {reverse_intervention}\n')
        return

    for name, key, handle in handles:
        handle.get()  # blocks until this run finishes
        _pull(name, key)
