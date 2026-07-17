"""Run intervention experiments (src/intervention.py) on Modal.

Port of the origin/interventions branch runner onto the current pipeline. Two changes from the
branch version:
  - the steering vector is passed as an argument (--vector-dir) instead of being hardcoded to
    steering_vectors/qwen7b/ — the branch left this as an explicit TODO;
  - --thinking-mode is forwarded, which routes intervention.py to formatInp_thinking (Qwen3.5).

Prefer driving this through _04_intervention.py rather than calling it directly.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import modal

app = modal.App("harmfulness-refusal-intervene")

_HERE = os.path.dirname(os.path.abspath(__file__))            # src/experiments_replication
_ROOT = os.path.dirname(os.path.dirname(_HERE))               # repo root (has src/ and data/)


def _generations_dir(name: str) -> str:
    """Local pull target: output/<model><size>/intervention/generations/ (mirrors _04's layout)."""
    return os.path.join(_HERE, "output", name, "intervention", "generations")


def _read_hf_token():
    """Parse HF_TOKEN from the repo-root .env at build time. Read only on the local machine; this
    module is re-imported in the container where no .env exists, so it returns None there."""
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

# Reproduce the local repo layout under /root so intervention.py's `from utils import ...` and
# `from template_inversion import ...` (both in src/) resolve, and cwd-relative paths work:
# REPO_DIR=/root, SRC_DIR=/root/src, DATA_DIR=/root/data.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(os.path.join(_ROOT, "requirements.txt"))
    # Blackwell (B200/B300) needs a CUDA-13 torch: the requirements' torch 2.6.0+cu121 only ships up
    # to sm_90, so B300 (sm_103) fails with "no kernel image available". torch 2.12.1+cu130 ships
    # sm_100 cubins, which run on sm_103 by same-major binary forward-compat. Harmless on A100.
    .pip_install("torch==2.12.1+cu130",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    # Mounted at runtime (copy=False): no build step needs these files, and copying them into an
    # image layer would abort with "modified during build process" if a file's mtime shifts mid-copy.
    # **/models excludes the local HF weight cache (the container re-downloads into its own).
    #
    # Unlike modal_run.py this does NOT ignore "**/output" or "*.pt" wholesale: the steering vectors
    # live at output/<name>/intervention/intervention_vectors/*.pt and are the whole input to this
    # job. The reason those blanket ignores exist is size, and the heavy tensors all sit under
    # datasets_outputs/ (activations/ and judge_activations*/ are ~3.5GB each), so that is what gets
    # excluded here instead. The vectors are ~540KB.
    .add_local_dir(
        os.path.join(_ROOT, "src"), "/root/src",
        ignore=["**/datasets_outputs/**", "**/__pycache__", "**/*.pyc", "**/models", "run",
                ".venv",
                # prior pulled generations: regenerated remotely, never read by the job
                "**/intervention/generations/**"],
    )
    .add_local_dir(os.path.join(_ROOT, "data"), "/root/data")
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("intervention-results", create_if_missing=True)


def _name(model: str, model_size: str) -> str:
    return f"{model}{model_size}"


def _key(model: str, model_size: str, dataset: str, vector: str,
         left: int, right: int, reverse: int, use_inversion: int,
         coeff: float, layer_s: int, layer_e: int, thinking_mode: str = "",
         max_decode_step: int = 1) -> str:
    """Deterministic results-volume key for one intervention run, so a detached run can be
    collected later and runs with different settings don't collide.

    coeff and the layer band are part of the key (the branch's version omitted them). Without
    them the coeff=0 control, a large-coeff check, and the real sweep all land in the SAME
    directory: intervention.py names its output per LAYER (<stem>-intervene<L>.json), so same-layer
    runs overwrite and different-layer runs silently merge into one mixed result set.
    """
    direction = "less" if reverse else "more"
    mode = thinking_mode or "instruct"
    return (f"{model}{model_size}-{mode}-{dataset}-{vector}-{direction}-{left}-{right}"
            f"-inv{use_inversion}-c{coeff:g}-L{layer_s}_{layer_e}-d{max_decode_step}")


@app.function(
    image=image, gpu="A100", volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=[hf_secret], timeout=14400
)
def run_intervention(
        model: str, model_size: str, left: int, right: int,
        dataset: str, vector: str, reverse_intervention: int,
        intervene_context_only: int, intervene_all: int, arg_key_prompt: str,
        layer_s: int, layer_e: int, coeff_select: float,
        max_token_generate: int, max_decode_step_while_intervene: int,
        use_inversion: int, inversion_prompt_idx: int, thinking_mode: str,
        batch_size: int,
    ) -> tuple[str, str]:
    """Run intervention.py for one config and persist its output JSON(s) to /results/<key>.

    Output is committed in a `finally` so a late-stage error can't discard an otherwise-complete
    run — whatever intervention.py produced is still saved.
    """
    import traceback
    os.environ["HF_HUB_CACHE"] = "/cache"
    # intervention.py runs `from utils import ...` and `from template_inversion import ...` with
    # both modules in src/, so the subprocess needs src/ on PYTHONPATH.
    os.environ["PYTHONPATH"] = "/root/src"
    os.chdir("/root/src")

    import torch  # log the actual GPU this container got
    print(f"### GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
          f"| torch {torch.__version__}", flush=True)

    name = _name(model, model_size)
    key = _key(model, model_size, dataset, vector, left, right, reverse_intervention, use_inversion,
               coeff_select, layer_s, layer_e, thinking_mode,
               max_decode_step_while_intervene)

    test_data_pth = f"/root/data/{dataset}.json"
    _iv = f"/root/src/experiments_replication/output/{name}/intervention"
    vector_pth = f"{_iv}/intervention_vectors/{thinking_mode or 'instruct'}/{vector}.pt"
    if not os.path.exists(vector_pth):
        raise FileNotFoundError(
            f"steering vector not found: {vector_pth}. Run the `vectors` stage of "
            f"_04_intervention.py first with the SAME thinking_mode, and confirm it wrote to "
            f"output/{name}/intervention/intervention_vectors/{thinking_mode or 'instruct'}/."
        )
    out_dir = f"{_iv}/generations/{key}"
    os.makedirs(out_dir, exist_ok=True)
    output_pth = f"{out_dir}/{dataset}-{'less' if reverse_intervention else 'more'}.json"

    cmd = [
        sys.executable, "-u", "/root/src/intervention.py",
        "--test_data_pth", test_data_pth,
        "--output_pth", output_pth,
        "--intervention_vector", vector_pth,
        "--reverse_intervention", str(reverse_intervention),
        "--intervene_context_only", str(intervene_context_only),
        "--intervene_all", str(intervene_all),
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
        "--batch_size", str(batch_size),
    ]
    if thinking_mode:
        cmd += ["--thinking_mode", thinking_mode]

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


@app.function(
    image=image, gpu="A100", volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=[hf_secret], timeout=14400
)
def judge_key(key: str, judge_config: str, judge_batch_size: int, thinking_mode: str,
              overwrite: bool) -> str:
    """Label an already-generated run on the results volume: /results/<key>/*.labels.json.

    Separate from run_intervention on purpose: the generations are already committed, so re-judging
    (a changed judge config, a fixed answer-extraction rule) must not require re-running hours of
    GPU generation. Reads and writes the volume in place.

    Locally the judge is the bottleneck -- judge_config_thinking.json sets thinking=true and
    max_new_tokens=4096, and a 16GB card only fits batch 4, giving ~4.6s/row (~3h for 2400 rows).
    On this GPU the config's own batch_size (64) fits, which is the point of running it here.
    """
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.environ["PYTHONPATH"] = "/root/src"
    os.chdir("/root/src/experiments_replication")
    sys.path.insert(0, "/root/src/experiments_replication")
    sys.path.insert(0, "/root/src")

    import torch
    print(f"### GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "i04", "/root/src/experiments_replication/_04_intervention.py")
    i04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i04)

    run_path = f"/results/{key}"
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"no such run on the volume: /{key}")

    try:
        i04.judge_run(run_path, judge_config, overwrite=overwrite,
                      judge_batch_size=judge_batch_size or None, thinking_mode=thinking_mode)
    finally:
        results_vol.commit()
    return key


def _local_dest(name: str, key: str) -> str:
    """Non-colliding dir under output/<name>/intervention/generations/: key, else key_2, key_3, …"""
    base = _generations_dir(name)
    dest = os.path.join(base, key)
    if not os.path.exists(dest):
        return dest
    i = 2
    while os.path.exists(os.path.join(base, f"{key}_{i}")):
        i += 1
    return os.path.join(base, f"{key}_{i}")


def _pull(name: str, key: str):
    """Fetch /<key>/ from the results volume into output/<name>/intervention/generations/<key>/.

    A run commits its output dir in a `finally`, so the remote dir can exist while still EMPTY --
    the job is mid-generation, or it died before producing anything (and a killed run leaves the
    empty dir behind as debris). `volume get` then downloads nothing and the expected subdirectory
    never appears, so report that plainly instead of dying on a bare FileNotFoundError.
    """
    dest = _local_dest(name, key)
    os.makedirs(_generations_dir(name), exist_ok=True)
    # Download into a scratch dir OUTSIDE the output tree so a failure never leaves debris.
    tmp = tempfile.mkdtemp(prefix=f"pull_{key}_")
    modal_cli = [sys.executable, "-m", "modal"]  # `modal` may not be on PATH
    try:
        subprocess.run(modal_cli + ["volume", "get", "intervention-results", f"/{key}", tmp,
                                    "--force"], check=True)
        src_dir = os.path.join(tmp, key)
        if not os.path.isdir(src_dir) or not os.listdir(src_dir):
            print(f"  nothing to pull yet: /{key} is empty on the volume "
                  f"(run still generating, or it failed before writing any layer)")
            return
        shutil.move(src_dir, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"pulled -> {dest} ({len(os.listdir(dest))} files)")


@app.local_entrypoint()
def entry(
    model: str = "qwen35",
    model_size: str = "9b",
    left: int = 0,
    right: int = 50,
    dataset: str = "alpaca_data_instruction",
    vector: str = "hf",
    reverse_intervention: int = 0,
    intervene_context_only: int = 0,
    intervene_all: int = 1,
    arg_key_prompt: str = "instruction",
    layer_s: int = 8,
    layer_e: int = 20,
    coeff_select: float = 1.0,
    # 2048: measured on real 9b genthink rows, tokens needed just to close </think> are median 1017 /
    # p95 2438; 512 leaves ~81% of rows truncated before any answer exists.
    max_token_generate: int = 2048,
    max_decode_step_while_intervene: int = 1,
    use_inversion: int = 0,
    inversion_prompt_idx: int = 1,
    thinking_mode: str = "genthink",
    batch_size: int = 64,
    gpu: str = "A100",
    timeout: int = 14400,
    wait: bool = True,
    collect_only: bool = False,
    judge: bool = False,
    judge_only: bool = False,
    judge_config: str = "configs/eval/judge_config_thinking.json",
    judge_batch_size: int = 64,
    judge_overwrite: bool = False,
):
    """Launch (and optionally collect) one intervention run.

    Defaults target Section 3.4 / Figure 4 on the Qwen3.5 thinking track: alpaca, harmfulness
    vector, no inversion, banded layers.

    Interactive (runs, waits, pulls results locally):
        modal run src/experiments_replication/modal_intervene.py

    Detached (safe to close the laptop), then collect later:
        modal run --detach src/experiments_replication/modal_intervene.py --no-wait
        modal run src/experiments_replication/modal_intervene.py --collect-only

    Collect keys are deterministic (model+size+dataset+vector+direction+left+right+inv), so the
    same flags retrieve exactly the run that was launched.
    """
    key = _key(model, model_size, dataset, vector, left, right, reverse_intervention, use_inversion,
               coeff_select, layer_s, layer_e, thinking_mode,
               max_decode_step_while_intervene)

    if collect_only:
        try:
            _pull(_name(model, model_size), key)
        except subprocess.CalledProcessError:
            print(f"  not ready on volume yet (skipped): {key}")
        return

    if judge_only:
        # Label an already-generated run in place on the volume. No generation, no 9B download.
        jfn = judge_key.with_options(gpu=gpu, timeout=timeout)
        h = jfn.spawn(key, judge_config, judge_batch_size, thinking_mode, judge_overwrite)
        if not wait:
            print(f"\nLaunched judge detached for {key}.")
            return
        h.get()
        _pull(_name(model, model_size), key)
        return

    fn = run_intervention.with_options(gpu=gpu, timeout=timeout)
    handle = fn.spawn(
        model, model_size, left, right, dataset, vector,
        reverse_intervention, intervene_context_only, intervene_all, arg_key_prompt,
        layer_s, layer_e, coeff_select,
        max_token_generate, max_decode_step_while_intervene,
        use_inversion, inversion_prompt_idx, thinking_mode, batch_size,
    )

    if not wait and judge:
        # --judge chains the judge AFTER handle.get(), which the --no-wait path never reaches. Fail
        # loudly rather than silently dropping the flag: the generations would land fine and the
        # labels would just never appear, which looks like a judge crash rather than a dropped flag.
        raise ValueError(
            "--judge cannot be combined with --no-wait: the judge is chained after the generation "
            "handle resolves, which a detached launch never waits for. Launch with --no-wait, then "
            "label the finished run with --judge-only (same flags), which is a separate GPU job."
        )

    if not wait:
        print("\nLaunched detached; results commit to the 'intervention-results' volume as the run "
              f"finishes.\nCollect later with:\n"
              f"  modal run {os.path.relpath(__file__, os.getcwd())} --collect-only "
              f"--model {model} --model-size {model_size} --dataset {dataset} --vector {vector} "
              f"--reverse-intervention {reverse_intervention} --left {left} --right {right} "
              f"--use-inversion {use_inversion} --coeff-select {coeff_select} "
              f"--layer-s {layer_s} --layer-e {layer_e}\n")
        return

    handle.get()  # blocks until the run finishes
    if judge:
        # Judge on the same GPU class right after generating: the generations are already on the
        # volume, so this only pays for the 4B judge download, not another 9B one.
        jfn = judge_key.with_options(gpu=gpu, timeout=timeout)
        jfn.remote(key, judge_config, judge_batch_size, thinking_mode, judge_overwrite)
    _pull(_name(model, model_size), key)
