"""Table 1 — compare refusal rates across generation configs.

For a given model directory (thinking or non-thinking), reads the classified generations and
computes, for each generation type, the refusal rate on harmful content, on harmless content, and
over all data points. This mirrors the paper's Table 1 (refusal rate w/ vs w/o post-instruction
tokens = gentpost vs gentinst), generalized to the thinking configs
(genthink / gennothink / gennothink_stripped).

Refusal rate = refused / (accepted + refused), aggregated over the datasets in each category:
  harmful  = advbench + jbb + sorrybench
  harmless = xstest + alpaca

The `eval` stage already splits each generation into <stem>_accepted.json / <stem>_refused.json, so
this is a pure count over those files — no model load and no re-evaluation. The generation type is
recovered from the filename: <dataset>_<gentype>_{accepted,refused}.json, where <gentype> is the part
after the first "_gen" (e.g. gentinst, gentpost, genthink, gennothink, gennothink_stripped). Harmless
non-thinking files carry no gen suffix (xstest.json, alpaca.json) and are grouped under "(default)".

Usage: there is no argparse — edit the config block in the `__main__` at the bottom (MODEL,
MODEL_SIZE, USE_JUDGE_CLASSIFICATIONS) and run:

    python _table1_compare_acceptance_rates.py

Set USE_JUDGE_CLASSIFICATIONS=True to read the judge-corrected splits (judge_classifications/)
instead of the default easy_eval splits (classified_generations/).
"""

import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

HARMFUL_DATASETS = ("advbench", "jbb", "sorrybench")
HARMLESS_DATASETS = ("xstest", "alpaca")

# preferred display order of generation types; anything else is appended alphabetically.
GEN_TYPE_ORDER = ("gentinst", "gentpost", "genthink", "gennothink", "gennothink_stripped",
                  "gennothink_stripped_v2", "(default)")


def _classified_dir(model, model_size, classified_subdir="classified_generations"):
    """output/<model><size>/datasets_outputs/<classified_subdir> (mirrors _00_run_inference._out_dirs)."""
    return os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs", classified_subdir)


def _parse_stem(stem):
    """(dataset, gen_type, label) from a classified filename stem, or None if it isn't a split file.

    'advbench_gennothink_stripped_accepted' -> ('advbench', 'gennothink_stripped', 'accepted')
    'xstest_refused'                        -> ('xstest', '(default)', 'refused')
    """
    if stem.endswith("_accepted"):
        base, label = stem[: -len("_accepted")], "accepted"
    elif stem.endswith("_refused"):
        base, label = stem[: -len("_refused")], "refused"
    else:
        return None
    dataset = base.split("_gen")[0]
    gen_type = base[len(dataset) + 1:] if len(base) > len(dataset) else "(default)"
    return dataset, gen_type, label


def compute_from_dir(cdir):
    """Count accepted/total per (gen_type, category) and per dataset from a classified dir.

    Returns (agg, per_ds):
      agg[gen_type][category]              = [accepted, total]
      per_ds[(gen_type, category, dataset)] = [accepted, total]
    category in {'harmful', 'harmless'}.
    """
    if not os.path.isdir(cdir):
        raise FileNotFoundError(f"no classified generations dir: {cdir}")

    agg = defaultdict(lambda: {"harmful": [0, 0], "harmless": [0, 0]})
    per_ds = defaultdict(lambda: [0, 0])

    for path in sorted(glob.glob(os.path.join(cdir, "*.json"))):
        parsed = _parse_stem(os.path.splitext(os.path.basename(path))[0])
        if parsed is None:
            continue
        dataset, gen_type, label = parsed
        if dataset in HARMFUL_DATASETS:
            category = "harmful"
        elif dataset in HARMLESS_DATASETS:
            category = "harmless"
        else:
            print(f"  (skipping unrecognized dataset '{dataset}')")
            continue
        with open(path, encoding="utf-8") as f:
            n = len(json.load(f))
        agg[gen_type][category][1] += n
        per_ds[(gen_type, category, dataset)][1] += n
        if label == "accepted":
            agg[gen_type][category][0] += n
            per_ds[(gen_type, category, dataset)][0] += n
    return agg, per_ds


def compare_acceptance_rates(model, model_size, classified_subdir="classified_generations"):
    """Convenience wrapper: resolve the model's classified dir, then compute_from_dir."""
    return compute_from_dir(_classified_dir(model, model_size, classified_subdir))


def _rate(pair):
    """Refusal rate from an [accepted, total] pair: refused = total - accepted."""
    acc, tot = pair
    ref = tot - acc
    return "—" if tot == 0 else f"{100.0 * ref / tot:5.1f}%  ({ref}/{tot})"


def _total(cats):
    """Sum the harmful and harmless [accepted, total] pairs into one overall pair."""
    return [cats["harmful"][0] + cats["harmless"][0], cats["harmful"][1] + cats["harmless"][1]]


def _sorted_gen_types(gen_types):
    order = {g: i for i, g in enumerate(GEN_TYPE_ORDER)}
    return sorted(gen_types, key=lambda g: (order.get(g, len(order)), g))


def print_table(label, agg, per_ds, detail=True):
    print(f"\nRefusal rates — {label}  (refusal% = refused / total)\n")
    header = f"{'gen type':<22}{'harmful refusal':<26}{'harmless refusal':<26}{'total refusal'}"
    print(header)
    print("-" * max(len(header), 60))
    for gt in _sorted_gen_types(agg.keys()):
        print(f"{gt:<22}{_rate(agg[gt]['harmful']):<26}{_rate(agg[gt]['harmless']):<26}"
              f"{_rate(_total(agg[gt]))}")
        if detail:
            for cat, datasets in (("harmful", HARMFUL_DATASETS), ("harmless", HARMLESS_DATASETS)):
                for ds in datasets:
                    pair = per_ds.get((gt, cat, ds))
                    if pair:
                        print(f"    {cat:<9}{ds:<12}{_rate(pair)}")
    print()


if __name__ == "__main__":
    # ---- configure here, then run: python _table1_compare_acceptance_rates.py ----
    MODEL = "qwen35"
    MODEL_SIZE = "9b"
    # True -> read output/<model><size>/datasets_outputs/judge_classifications/ (the judge-corrected
    # splits from an eval run with use_judge=True); False -> the default classified_generations/.
    USE_JUDGE_CLASSIFICATIONS = True
    # ------------------------------------------------------------------------------

    subdir = "judge_classifications" if USE_JUDGE_CLASSIFICATIONS else "classified_generations"
    agg, per_ds = compare_acceptance_rates(MODEL, MODEL_SIZE, subdir)
    print_table(f"{MODEL}{MODEL_SIZE}  [{subdir}]", agg, per_ds)
