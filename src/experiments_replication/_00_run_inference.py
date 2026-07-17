"""Run all inference passes and split each into accepted/refused generations.

Two stages, both writing under output/<model><size>/datasets_outputs/:
  run_all_inference -> generations/            one JSON per dataset x generation-config.
  evaluate          -> classified_generations/ each generation split by refusal label.

Bucket/cluster composition and train/test splits are intentionally NOT done here: they
are a pure-CPU step in dynamic_bucket_formation.py, on top of the activations that
_01_compute_activations.py extracts from classified_generations/.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")


def _read_rows(path):
    """Read a list of records from either an indented JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_pretty(path, rows):
    """Write records as an indented JSON array (readable; still loadable by read_row)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def run_inference_on_dataset(
    input_file,
    output_file,
    do_not_use_last_inst_tok,
    *,
    model="qwen",
    model_size="0.5b",
    left=0,
    right=10,
):
    """Invoke src/inference.py on one dataset, writing JSONL to output_file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd = [
        sys.executable,
        "inference.py",
        "--input", input_file,
        "--output_file_name", output_file,
        "--model", model,
        "--model_size", model_size,
        "--left", str(left),
        "--right", str(right),
        "--use_template", "1",
        "--use_jb", "0",
        "--use_inversion", "0",
        "--record_prob_max_pos", "0",
        "--max_len", "128",
        "--do_not_use_last_inst_tok", str(do_not_use_last_inst_tok),
    ]
    # cwd=SRC_DIR so inference.py's `from utils import ...` resolves.
    subprocess.run(cmd, check=True, cwd=SRC_DIR)

    # inference.py writes JSONL; rewrite as an indented array for readability.
    _write_pretty(output_file, _read_rows(output_file))


def _out_dirs(model, model_size):
    """The 3 stage dirs under output/<model><size>/datasets_outputs/.

    Returns (generations, classified_generations, activations). The last is written by
    _01_compute_activations.py but named here so all stages share one layout definition.
    """
    base = os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs")
    return (
        os.path.join(base, "generations"),
        os.path.join(base, "classified_generations"),
        os.path.join(base, "activations"),
    )


# (dataset file, output name, do_not_use_last_inst_tok)
# do_not_use_last_inst_tok: 1 = t_inst config, 0 = t_post config. The 'gen' in the output
# name marks the generation prompt config (whether post-instruction tokens were removed).
RUNS = [
    # harmful x gen-t_inst
    ("advbench.json", "advbench_gentinst.json", 1),
    ("jbb.json", "jbb_gentinst.json", 1),
    ("sorry-badq.json", "sorrybench_gentinst.json", 1),
    # harmful x gen-t_post
    ("advbench.json", "advbench_gentpost.json", 0),
    ("jbb.json", "jbb_gentpost.json", 0),
    ("sorry-badq.json", "sorrybench_gentpost.json", 0),
    # harmless x gen-t_post (single config)
    ("xstest-harmless.json", "xstest.json", 0),
    ("alpaca_data_instruction.json", "alpaca.json", 0),
]


def run_inference_on_dataset_thinking(
    input_file,
    output_file,
    thinking_mode,
    *,
    model="qwen35",
    model_size="0.8b",
    left=0,
    right=10,
    max_len=512,
    batch_size=8,
    sampling_config="",
):
    """Invoke src/inference.py in thinking mode (genthink/gennothink) on one dataset."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd = [
        sys.executable, "inference.py",
        "--input", input_file,
        "--output_file_name", output_file,
        "--model", model,
        "--model_size", model_size,
        "--left", str(left),
        "--right", str(right),
        "--use_template", "1",
        "--use_jb", "0",
        "--record_prob_max_pos", "0",
        "--max_len", str(max_len),
        "--batch_size", str(batch_size),
        "--thinking_mode", thinking_mode,
    ]
    if sampling_config:
        cmd += ["--sampling_config", sampling_config]  # absolute path (subprocess cwd is SRC_DIR)
    subprocess.run(cmd, check=True, cwd=SRC_DIR)
    _write_pretty(output_file, _read_rows(output_file))


# (dataset file, output name, thinking_mode). 5 datasets x {genthink, gennothink,
# gennothink_stripped, gennothink_stripped_v2} = 20 runs, all in the same generations/ dir. Unlike
# the Qwen2 pipeline there is no gentinst/gentpost split. The two stripped modes remove a different
# amount of the trailing template so the model must produce it itself (the thinking-family analogue
# of gentinst): gennothink_stripped cuts everything after <|im_start|>, gennothink_stripped_v2 keeps
# gennothink's prompt but drops its trailing "\n\n". Scope a run to one via only_modes.
RUNS_THINKING = [
    ("advbench.json", "advbench_genthink.json", "genthink"),
    ("advbench.json", "advbench_gennothink.json", "gennothink"),
    ("advbench.json", "advbench_gennothink_stripped.json", "gennothink_stripped"),
    ("advbench.json", "advbench_gennothink_stripped_v2.json", "gennothink_stripped_v2"),
    ("jbb.json", "jbb_genthink.json", "genthink"),
    ("jbb.json", "jbb_gennothink.json", "gennothink"),
    ("jbb.json", "jbb_gennothink_stripped.json", "gennothink_stripped"),
    ("jbb.json", "jbb_gennothink_stripped_v2.json", "gennothink_stripped_v2"),
    ("sorry-badq.json", "sorrybench_genthink.json", "genthink"),
    ("sorry-badq.json", "sorrybench_gennothink.json", "gennothink"),
    ("sorry-badq.json", "sorrybench_gennothink_stripped.json", "gennothink_stripped"),
    ("sorry-badq.json", "sorrybench_gennothink_stripped_v2.json", "gennothink_stripped_v2"),
    ("xstest-harmless.json", "xstest_genthink.json", "genthink"),
    ("xstest-harmless.json", "xstest_gennothink.json", "gennothink"),
    ("xstest-harmless.json", "xstest_gennothink_stripped.json", "gennothink_stripped"),
    ("xstest-harmless.json", "xstest_gennothink_stripped_v2.json", "gennothink_stripped_v2"),
    ("alpaca_data_instruction.json", "alpaca_genthink.json", "genthink"),
    ("alpaca_data_instruction.json", "alpaca_gennothink.json", "gennothink"),
    ("alpaca_data_instruction.json", "alpaca_gennothink_stripped.json", "gennothink_stripped"),
    ("alpaca_data_instruction.json", "alpaca_gennothink_stripped_v2.json", "gennothink_stripped_v2"),
]


def _dataset_key(out_name):
    """The dataset name a run belongs to, from its output filename: the part before '_gen'/'.json'.
    e.g. 'alpaca_genthink.json' -> 'alpaca', 'advbench_gentinst.json' -> 'advbench', 'xstest.json' ->
    'xstest'. Used to filter runs to a single dataset."""
    return out_name[: -len(".json")].split("_gen")[0]


def _filter_runs(runs, only_datasets):
    """Keep only the runs whose dataset (_dataset_key of out_name, at tuple index 1) is in
    only_datasets (a comma-separated string or a list). None/empty -> all runs unchanged."""
    if not only_datasets:
        return runs
    wanted = {d.strip() for d in (only_datasets.split(",") if isinstance(only_datasets, str)
                                  else only_datasets) if d and d.strip()}
    sel = [r for r in runs if _dataset_key(r[1]) in wanted]
    if not sel:
        avail = sorted({_dataset_key(r[1]) for r in runs})
        raise ValueError(f"no runs match datasets={sorted(wanted)}; available: {avail}")
    return sel


def _filter_modes(runs, only_modes):
    """Keep only the runs whose thinking_mode (tuple index 2) is in only_modes (a comma-separated
    string or a list). None/empty -> all runs unchanged. Used to scope a run to one generation type
    (e.g. only gennothink_stripped)."""
    if not only_modes:
        return runs
    wanted = {m.strip() for m in (only_modes.split(",") if isinstance(only_modes, str)
                                  else only_modes) if m and m.strip()}
    sel = [r for r in runs if r[2] in wanted]
    if not sel:
        avail = sorted({r[2] for r in runs})
        raise ValueError(f"no runs match modes={sorted(wanted)}; available: {avail}")
    return sel


def run_all_inference_thinking(model, model_size, left, right, max_len=512, batch_size=8,
                               sampling_config="", only_datasets=None, only_modes=None):
    """Generate genthink/gennothink/gennothink_stripped responses for all 5 datasets (Qwen3.5
    thinking family).

    sampling_config: path to a sampling json (relative paths resolve against this dir); "" uses the
    inference defaults.
    only_datasets: comma-separated dataset names (e.g. "alpaca") to restrict generation to; None runs
    all. A named dataset runs all of its thinking configs.
    only_modes: comma-separated thinking modes (e.g. "gennothink_stripped") to restrict generation to;
    None runs all. Applied on top of only_datasets.
    """
    generations_dir, _, _ = _out_dirs(model, model_size)
    scfg = sampling_config
    if scfg and not os.path.isabs(scfg):
        scfg = os.path.join(HERE, scfg)
    for dataset_file, out_name, thinking_mode in _filter_modes(
            _filter_runs(RUNS_THINKING, only_datasets), only_modes):
        run_inference_on_dataset_thinking(
            input_file=os.path.join(DATA_DIR, dataset_file),
            output_file=os.path.join(generations_dir, out_name),
            thinking_mode=thinking_mode,
            model=model,
            model_size=model_size,
            left=left,
            right=right,
            max_len=max_len,
            batch_size=batch_size,
            sampling_config=scfg,
        )


def evaluate_thinking(model, model_size, use_judge=False,
                      judge_config="configs/eval/judge_config_thinking.json", only_modes=None,
                      only_datasets=None):
    """Split each thinking generation into accepted/refused using easy_eval_thinking.

    For genthink the classification runs only on the post-</think> answer; gennothink has no trace.
    gennothink_stripped is evaluated leniently: the model autocompleted the template, so we classify
    the text after the last </think> when present, else the whole output (no </think> -> keep, don't
    drop). The stored ori_output is left intact. gennothink_stripped_v2 needs no such handling: its
    </think> is in the prompt (only the trailing "\n\n" was cut), so like gennothink its output is
    already the bare answer and is classified whole.

    only_modes: comma-separated thinking modes (e.g. "gennothink_stripped") to restrict evaluation to;
    None evaluates all. Scope this to match the modes generated by a partial infer run.
    only_datasets: comma-separated dataset names (e.g. "advbench,jbb") to restrict evaluation to;
    None evaluates all. Applied under only_modes, mirroring run_all_inference_thinking — scope BOTH to
    match a partial infer run, or this reads generations that were never produced. The judge step is
    unaffected: it independently skips stems whose split files are missing.

    use_judge: after the split, re-verify the two opposing buckets with the judge LLM (thinking judge
    config; strip_think=true so it judges the post-</think> answer). Writes a parallel
    classified_generations_judge/ dir and leaves classified_generations/ untouched.
    """
    sys.path.insert(0, SRC_DIR)
    from eval import easy_eval_thinking

    generations_dir, classified_dir, _ = _out_dirs(model, model_size)
    os.makedirs(classified_dir, exist_ok=True)

    for _, out_name, thinking_mode in _filter_modes(
            _filter_runs(RUNS_THINKING, only_datasets), only_modes):
        data = _read_rows(os.path.join(generations_dir, out_name))
        if thinking_mode == "gennothink_stripped":
            # lenient literal strip: drop the model's autocompleted template prefix up to the last
            # </think> if it emitted one, else classify the whole output. Classify a view so
            # ori_output stays intact (kept for future activation extraction).
            view = [{**d, "_ans": d["ori_output"].split("</think>")[-1]
                     if "</think>" in d["ori_output"] else d["ori_output"]} for d in data]
            labels = easy_eval_thinking(view, tag="_ans", mode="refusal", think=False)
        else:
            labels = easy_eval_thinking(data, tag="ori_output", mode="refusal",
                                        think=(thinking_mode == "genthink"))
        stem = out_name[: -len(".json")]
        for suffix, keep_label in (("accepted", "5"), ("refused", "0")):
            split = [item for item, label in zip(data, labels) if label == keep_label]
            _write_pretty(os.path.join(classified_dir, f"{stem}_{suffix}.json"), split)
            print(f"{stem}_{suffix}: {len(split)} items")

    if use_judge:
        sys.path.insert(0, HERE)
        from judge_llm import reclassify_opposing
        reclassify_opposing(model, model_size, judge_config, only_modes=only_modes)


def run_all_inference(model, model_size, left, right, only_datasets=None):
    """Generate responses for all 8 dataset/config combinations.

    only_datasets: comma-separated dataset names (e.g. "alpaca") to restrict generation to; None runs
    all. A named harmful dataset runs BOTH its gentinst and gentpost configs.
    """
    generations_dir, _, _ = _out_dirs(model, model_size)
    for dataset_file, out_name, no_last_tok in _filter_runs(RUNS, only_datasets):
        run_inference_on_dataset(
            input_file=os.path.join(DATA_DIR, dataset_file),
            output_file=os.path.join(generations_dir, out_name),
            do_not_use_last_inst_tok=no_last_tok,
            model=model,
            model_size=model_size,
            left=left,
            right=right,
        )


def evaluate(model, model_size, use_judge=False,
             judge_config="configs/eval/judge_config.json"):
    """Split each generation into accepted/refused classified generations.

    label: '5' = accepted, '0' = refused (easy_eval refusal mode). No cross-source
    aggregation here; that is the job of dynamic_bucket_formation.py.

    use_judge: after the easy_eval split, re-verify the two error-prone "opposing" buckets
    (harmful->accepted, harmless->refused) with a judge LLM and rebucket disagreements. Purely
    additive on top of easy_eval; see judge_llm.reclassify_opposing. It only rewrites the classified
    JSONs (activations are left stale — re-run `acts` if you need judged figures).
    """
    sys.path.insert(0, SRC_DIR)
    from eval import easy_eval

    generations_dir, classified_dir, _ = _out_dirs(model, model_size)
    os.makedirs(classified_dir, exist_ok=True)

    for _, out_name, _ in RUNS:
        data = _read_rows(os.path.join(generations_dir, out_name))
        labels = easy_eval(data, tag="ori_output", mode="refusal")
        stem = out_name[: -len(".json")]
        for suffix, keep_label in (("accepted", "5"), ("refused", "0")):
            split = [item for item, label in zip(data, labels) if label == keep_label]
            _write_pretty(os.path.join(classified_dir, f"{stem}_{suffix}.json"), split)
            print(f"{stem}_{suffix}: {len(split)} items")

    if use_judge:
        sys.path.insert(0, HERE)
        from judge_llm import reclassify_opposing
        reclassify_opposing(model, model_size, judge_config)
