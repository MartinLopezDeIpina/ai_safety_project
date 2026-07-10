"""CPU-only re-bucketing from an activation store (no GPU re-extraction).

Insight (verified): an example's extracted activation is identical whether it was processed in the
tinst-run or tpost-run bucket, because extract_hidden.py always re-tokenizes with the full template.
So the 6 existing buckets_activations/*.pt already contain activations for ~every example, each
mappable to its instruction text via the bucket JSON (same row order). We build a text->activation
store once, recompute clean accept/refuse labels (recovering true tinst responses from the qwen
prompt-leak), and reassemble corrected buckets by re-indexing — all on CPU.

Corrected bucket definitions (paper-faithful, Appendix B):
  tinst_refused_harmful   = advbench+jbb, refused at tinst
  tinst_accepted_harmful  = advbench+jbb, accepted at tinst        (NO sorrybench: it sits on the
                                                                    harmless side of the tinst axis)
  tpost_refused_harmful   = advbench+jbb, refused at tpost
  tpost_accepted_harmful  = advbench+jbb+sorrybench, accepted at tpost
  accepted_harmless       = xstest+alpaca, accepted at tpost
  refused_harmless        = xstest, refused at tpost

Usage: python rebucket_store.py qwen 7b [--tinst-accepted-sources advbench,jbb]
Writes output/<model><size>/buckets_activations_v2/*.pt
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/ for utils
from utils import REFUSAL_PHRASE  # noqa: E402

BUCKET_NAMES = ["tinst_accepted_harmful", "tinst_refused_harmful",
                "tpost_accepted_harmful", "tpost_refused_harmful",
                "accepted_harmless", "refused_harmless"]

# dataset_outputs files -> (source tag, is tinst-run?)
DATASET_FILES = {
    "advbench_tinst": ("advbench", True), "advbench_tpost": ("advbench", False),
    "jbb_tinst": ("jbb", True), "jbb_tpost": ("jbb", False),
    "sorrybench_tinst": ("sorrybench", True), "sorrybench_tpost": ("sorrybench", False),
    "xstest": ("xstest", False), "alpaca": ("alpaca", False),
}


def instr_key(rec):
    """Instruction text used as the store key (mirrors formatInp_llama_persuasion precedence)."""
    if "prompt" in rec:
        return rec["prompt"]
    for k in ("instruction", "question", "bad_q"):
        if k in rec:
            return rec[k]
    return None


def source_of(rec):
    if "question_id" in rec or "turns" in rec or "prompt_style" in rec:
        return "sorrybench"
    if "focus" in rec or "type" in rec:
        return "xstest"
    if "instruction" in rec:
        return "alpaca"
    if "source" in rec or "category" in rec:
        return "jbb"
    return "advbench"


def _strip_special(text):
    """Text content before the first end-of-turn marker, stripped."""
    text = (text or "").strip()
    for marker in ("<|im_end|>", "<|eot_id|>", "</s>"):
        text = text.split(marker)[0]
    return text.strip()


def classify(text):
    """Three-way label: 'empty' | 'refused' | 'accepted'.

    Faithful to eval.py::easy_eval refusal phrases, but splits out EMPTY responses (which easy_eval
    lumps into 'refused'). Empty responses are generation artifacts, not refusals — for over-refusal
    analysis on harmless prompts they are contamination, so we exclude them from all buckets.
    """
    content = _strip_special(text)
    if content == "":
        return "empty"
    if any(phrase in content for phrase in REFUSAL_PHRASE):
        return "refused"
    return "accepted"


def is_refused(text):
    """Binary compat (empty counts as refused, like easy_eval)."""
    return classify(text) != "accepted"


def clean_tinst_response(ori_output, bad_q, model):
    """Recover the true model response from a (possibly prompt-leaked) tinst ori_output.

    For qwen/llama3 the tinst template has no assistant turn, so extract_model_output failed to strip
    the echoed prompt and ori_output == '<PREFIX>user\\n<instruction><response>'. Strip up to and
    including the instruction text; if not found, return ori_output unchanged (already clean).
    """
    if bad_q and bad_q in ori_output:
        return ori_output.split(bad_q, 1)[-1]
    # fallbacks: strip a known user-turn prefix
    for marker in ("<|im_start|>user\n", "<|start_header_id|>user<|end_header_id|>\n"):
        if ori_output.startswith(marker):
            return ori_output[len(marker):]
    return ori_output


def build_store(model, size):
    """key -> {'act': (L,T,H), 'source': str} from the 6 existing bucket tensors."""
    base = os.path.join(HERE, "output", f"{model}{size}")
    acts_dir = os.path.join(base, "buckets_activations")
    buckets_dir = os.path.join(base, "buckets")
    store = {}
    for name in BUCKET_NAMES:
        pt = os.path.join(acts_dir, name + ".pt")
        js = os.path.join(buckets_dir, name + ".json")
        if not (os.path.exists(pt) and os.path.exists(js)):
            continue
        acts = torch.load(pt, map_location="cpu").float()  # (L,N,T,H)
        recs = json.load(open(js, encoding="utf-8"))
        assert acts.shape[1] == len(recs), f"{name}: {acts.shape[1]} != {len(recs)}"
        for i, rec in enumerate(recs):
            k = instr_key(rec)
            if k is None or k in store:
                continue
            store[k] = {"act": acts[:, i], "source": source_of(rec)}
    return store


def compute_labels(model, size):
    """key -> {'source','tinst_refused','tpost_refused'} from cleaned dataset_outputs."""
    do_dir = os.path.join(HERE, "output", f"{model}{size}", "dataset_outputs")
    labels = {}
    for fname, (src, is_tinst) in DATASET_FILES.items():
        path = os.path.join(do_dir, fname + ".json")
        if not os.path.exists(path):
            continue
        for rec in json.load(open(path, encoding="utf-8")):
            k = instr_key(rec)
            if k is None:
                continue
            ori = rec.get("ori_output", "")
            if is_tinst:
                resp = clean_tinst_response(ori, rec.get("bad_q", ""), model)
            else:
                resp = ori
            entry = labels.setdefault(k, {"source": src})
            entry["tinst" if is_tinst else "tpost"] = classify(resp)  # empty|refused|accepted
    return labels


def assemble(store, keys):
    acts = [store[k]["act"] for k in keys if k in store]
    if not acts:
        return None, 0
    return torch.stack(acts, dim=1), len(acts)  # (L,N,T,H)


def main(model, size, tinst_accepted_sources, tpost_accepted_sources, harmless_sources):
    store = build_store(model, size)
    labels = compute_labels(model, size)
    print(f"store size={len(store)}  labels={len(labels)}")

    def keys_where(sources, field, want):
        """want in {'refused','accepted'}; 'empty' rows are excluded from every bucket."""
        out = []
        for k, e in labels.items():
            if e["source"] not in sources or field not in e:
                continue
            if e[field] == want and k in store:
                out.append(k)
        return out

    defs = {
        "tinst_refused_harmful":  keys_where(("advbench", "jbb"), "tinst", "refused"),
        "tinst_accepted_harmful": keys_where(tinst_accepted_sources, "tinst", "accepted"),
        "tpost_refused_harmful":  keys_where(("advbench", "jbb"), "tpost", "refused"),
        # advbench/jbb are ~all refused at tpost; their few "accepted" are substring false-negatives
        # (they project to the refused side). Sorry-Bench provides genuine accepted-harmful@tpost.
        "tpost_accepted_harmful": keys_where(tpost_accepted_sources, "tpost", "accepted"),
        # harmless anchor: Alpaca is the canonical clean harmless reference (Arditi et al.). Including
        # xstest-accepted here would put refused_harmless (also xstest) in the same distribution as its
        # own harmless anchor, a shortcut that masks the tpost refusal flip. Default: alpaca only.
        "accepted_harmless":      keys_where(harmless_sources, "tpost", "accepted"),
        "refused_harmless":       keys_where(("xstest",), "tpost", "refused"),
    }

    out_dir = os.path.join(HERE, "output", f"{model}{size}", "buckets_activations_v2")
    os.makedirs(out_dir, exist_ok=True)
    for name, keys in defs.items():
        tensor, n = assemble(store, keys)
        if tensor is None:
            print(f"  {name:26s}: EMPTY")
            continue
        torch.save(tensor, os.path.join(out_dir, name + ".pt"))
        from collections import Counter
        srcs = dict(Counter(store[k]["source"] for k in keys if k in store))
        print(f"  {name:26s}: N={n:4d}  sources={srcs}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="qwen")
    ap.add_argument("size", nargs="?", default="7b")
    ap.add_argument("--tinst-accepted-sources", default="advbench,jbb",
                    help="comma-separated sources for tinst_accepted_harmful")
    ap.add_argument("--tpost-accepted-sources", default="sorrybench",
                    help="comma-separated sources for tpost_accepted_harmful")
    ap.add_argument("--harmless-sources", default="alpaca",
                    help="comma-separated sources for accepted_harmless anchor")
    a = ap.parse_args()
    main(a.model, a.size, tuple(a.tinst_accepted_sources.split(",")),
         tuple(a.tpost_accepted_sources.split(",")), tuple(a.harmless_sources.split(",")))
