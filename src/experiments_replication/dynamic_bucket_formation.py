"""CPU-only train/test bucket formation from the activation store (no GPU re-extraction).

_01_compute_activations.py writes one activation tensor per classified generation under
datasets_outputs/activations/<source>.pt, each of shape (L, N, T, H) holding both the
t_inst (token index 1) and t_post (index -1) positions. This module composes those source
tensors into single-token-position clusters used by the experiments, and splits every
source into a train/test partition, entirely on CPU.

Each cluster is tied to one token position, so a returned cluster tensor has shape
(L, N, H) — the token axis is already sliced away. Composition is configurable:
BUCKET_TRAIN / BUCKET_TEST map a cluster name to (its activation sources, token position).
Experiments import build_splits and may override those lists (and test_ratio) to try
different compositions without re-running any GPU work.

Split rule (per source, deterministic for a given seed):
  - source in BOTH train and test configs -> ratio split (train = first 1-test_ratio,
    test = remaining test_ratio, over a fixed per-source permutation).
  - source ONLY in train -> all its examples go to train.
  - source ONLY in test  -> all its examples go to test.

Usage: python dynamic_bucket_formation.py qwen 0.5b [test_ratio]
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# token position -> index into a source tensor's token axis (matches _02/_03).
POSITION_INDEX = {"tinst": 1, "tpost": -1}

# cluster name -> (activation source stems in datasets_outputs/activations/ without .pt,
#                  token position to slice out).
# Paper-faithful defaults (see EXPERIMENT_LOG.md / former rebucket_store.py):
#  - harmful_*_tinst read the gentinst-config sources at t_inst; harmful_*_tpost read the
#    gentpost-config sources at t_post.
#  - accepted_harmful @ tinst uses advbench+jbb (Sorry-Bench sits on the harmless side of the
#    advbench/jbb harmfulness axis at t_inst); accepted_harmful @ tpost uses Sorry-Bench
#    (advbench/jbb are ~all refused at t_post).
#  - harmless sources are generated in a single (gentpost) config but each tensor holds both
#    token positions, so the same source feeds its _tinst and _tpost clusters. Anchored on
#    Alpaca (accepted) / xstest (refused) to avoid the xstest self-similarity confound.
# "normal" config: accepted_harmful drawn from advbench+jbb+sorrybench gentinst_accepted at BOTH
# positions (so its _tinst/_tpost clusters share sources and pair per-instruction in Figure 3);
# harmless anchored on alpaca+xstest; refused_harmless is HELD OUT of train (empty) and appears only
# in test. These module constants are the fallback default when no bucket_config json is passed.
BUCKET_TRAIN = [
    ("accepted_harmful_tinst",  ["advbench_gentinst_accepted", "jbb_gentinst_accepted", "sorrybench_gentinst_accepted"], "tinst"),
    ("accepted_harmful_tpost",  ["advbench_gentinst_accepted", "jbb_gentinst_accepted", "sorrybench_gentinst_accepted"], "tpost"),
    ("refused_harmful_tinst",   ["advbench_gentinst_refused", "jbb_gentinst_refused"], "tinst"),
    ("refused_harmful_tpost",   ["advbench_gentpost_refused", "jbb_gentpost_refused"], "tpost"),
    ("accepted_harmless_tinst", ["alpaca_accepted", "xstest_accepted"], "tinst"),
    ("accepted_harmless_tpost", ["alpaca_accepted", "xstest_accepted"], "tpost"),
    # no refused harmless in train, held out
    ("refused_harmless_tinst", [], "tinst"),
    ("refused_harmless_tpost", [], "tpost"),
]

BUCKET_TEST = [
    ("accepted_harmful_tinst",  ["advbench_gentinst_accepted", "jbb_gentinst_accepted", "sorrybench_gentinst_accepted"], "tinst"),
    ("accepted_harmful_tpost",  ["advbench_gentinst_accepted", "jbb_gentinst_accepted", "sorrybench_gentinst_accepted"], "tpost"),
    ("refused_harmful_tinst",   ["advbench_gentinst_refused", "jbb_gentinst_refused"], "tinst"),
    ("refused_harmful_tpost",   ["advbench_gentpost_refused", "jbb_gentpost_refused"], "tpost"),
    ("accepted_harmless_tinst", ["alpaca_accepted", "xstest_accepted"], "tinst"),
    ("accepted_harmless_tpost", ["alpaca_accepted", "xstest_accepted"], "tpost"),
    ("refused_harmless_tinst",  ["xstest_refused"], "tinst"),
    ("refused_harmless_tpost",  ["xstest_refused"], "tpost"),
]

# "ALT" config (paper-faithful anchors): accepted_harmful from advbench+jbb @tinst / sorrybench
# @tpost; harmless anchored on alpaca (accepted) / xstest (refused) only; refused_harmless present in
# both splits. Same composition for train and test (ratio-split each source).
BUCKET_TRAIN_ALT = [
    ("accepted_harmful_tinst",  ["advbench_gentinst_accepted", "jbb_gentinst_accepted"], "tinst"),
    ("accepted_harmful_tpost",  ["sorrybench_gentpost_accepted"], "tpost"),
    ("refused_harmful_tinst",   ["advbench_gentinst_refused", "jbb_gentinst_refused"], "tinst"),
    ("refused_harmful_tpost",   ["advbench_gentpost_refused", "jbb_gentpost_refused"], "tpost"),
    ("accepted_harmless_tinst", ["alpaca_accepted"], "tinst"),
    ("accepted_harmless_tpost", ["alpaca_accepted"], "tpost"),
    ("refused_harmless_tinst",  ["xstest_refused"], "tinst"),
    ("refused_harmless_tpost",  ["xstest_refused"], "tpost"),
]
BUCKET_TEST_ALT = [(name, list(sources), pos) for name, sources, pos in BUCKET_TRAIN_ALT]


def _acts_dir(model, model_size):
    return os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs", "activations")


def _sources(config):
    """Flat set of every activation source referenced by a bucket config."""
    return {s for _, sources, _ in config for s in sources}


def _split_indices(n, test_ratio, seed, source):
    """Deterministic (train_idx, test_idx) for a source's N examples.

    Seeded per source so the same source splits identically on every call, independent of
    the order sources are processed.
    """
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()
    rng = np.random.default_rng([seed, int(digest, 16) % (2**32)])
    perm = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    return perm[n_test:], perm[:n_test]  # train, test


def build_splits(model, model_size, bucket_train=BUCKET_TRAIN, bucket_test=BUCKET_TEST,
                 test_ratio=0.5, seed=0):
    """Compose and split activations into train/test clusters.

    Returns (train_acts, test_acts), each a dict cluster_name -> (L, N, H) tensor at the
    cluster's token position. A cluster with no available source examples is omitted.
    """
    acts_dir = _acts_dir(model, model_size)
    train_sources = _sources(bucket_train)
    test_sources = _sources(bucket_test)

    cache = {}  # source -> (tensor|None, train_idx, test_idx)

    def load(source):
        if source not in cache:
            path = os.path.join(acts_dir, source + ".pt")
            if not os.path.exists(path):
                cache[source] = (None, None, None)
            else:
                tensor = torch.load(path, map_location="cpu").float()
                train_idx, test_idx = _split_indices(tensor.shape[1], test_ratio, seed, source)
                cache[source] = (tensor, train_idx, test_idx)
        return cache[source]

    def assemble(config, side):
        other_sources = test_sources if side == "train" else train_sources
        out = {}
        for cluster, sources, position in config:
            pidx = POSITION_INDEX[position]
            slices = []
            for source in sources:
                tensor, train_idx, test_idx = load(source)
                if tensor is None:
                    continue
                if source in other_sources:
                    idx = train_idx if side == "train" else test_idx
                else:
                    idx = np.arange(tensor.shape[1])  # source only on this side -> all rows
                if len(idx):
                    sl = tensor[:, idx, pidx, :]  # (L, n, H) at this token position
                    # drop corrupt rows whose vector is zero at this position in ANY layer (a zero
                    # norm makes cos_sim 0/0 -> NaN downstream). Seen when an extraction run
                    # offloads the model and zeroes some activations.
                    keep = (sl.norm(dim=-1) > 0).all(dim=0)  # (n,)
                    if not bool(keep.all()):
                        for did in np.asarray(idx)[~keep.numpy()]:
                            print(f"[warning] dropped activation id {int(did)}, null "
                                  f"(source={source}, pos={position})")
                    sl = sl[:, keep]
                    if sl.shape[1]:
                        slices.append(sl)
            if slices:
                out[cluster] = torch.cat(slices, dim=1)
        return out

    return assemble(bucket_train, "train"), assemble(bucket_test, "test")


def load_config(path):
    """Load a bucket config json -> dict. Missing keys fall back to the module defaults."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "bucket_train": cfg.get("bucket_train", BUCKET_TRAIN),
        "bucket_test": cfg.get("bucket_test", BUCKET_TEST),
        "test_ratio": cfg.get("test_ratio", 0.5),
        "seed": cfg.get("seed", 0),
    }


def gen_buckets(model, model_size, bucket_config=None):
    """Compose train/test buckets for one model -> {"train": {...}, "test": {...}}.

    bucket_config: path to a json config (resolved against HERE when relative). None -> module
    defaults. Each dict maps cluster_name -> (L, N, H) tensor at the cluster's token position.
    """
    if bucket_config is not None:
        path = bucket_config if os.path.isabs(bucket_config) else os.path.join(HERE, bucket_config)
        if not os.path.exists(path):
            print(f"[warning] {bucket_config} not found; using hard-coded default bucket config")
            bucket_config = None
    if bucket_config is None:
        cfg = {"bucket_train": BUCKET_TRAIN, "bucket_test": BUCKET_TEST,
               "test_ratio": 0.5, "seed": 0}
    else:
        cfg = load_config(path)

    train_acts, test_acts = build_splits(
        model, model_size,
        bucket_train=cfg["bucket_train"], bucket_test=cfg["bucket_test"],
        test_ratio=cfg["test_ratio"], seed=cfg["seed"])
    return {"train": train_acts, "test": test_acts}


# ---------------------------------------------------------------------------
# Qwen3.5 thinking buckets. The thinking activation tensors are (L, N, 22, H) — one fixed 22-slot
# token layout instead of two named positions. So here a "family" (refused_harmful / accepted_harmless
# / accepted_harmful / refused_harmless) is kept whole (L, N, 22, H); the figure slices positions
# itself. Config entries are position-agnostic 2-tuples [family_name, [source_stems]]. Split rule is
# the same md5-seeded per-source rule as build_splits. Zero-norm (null) slots are NOT dropped here —
# that is done per-position at plot time, since a row is legitimately null at some slots.
# ---------------------------------------------------------------------------
def build_splits_thinking(model, model_size, families_train, families_test, test_ratio=0.2, seed=0):
    """Compose thinking families into train/test dicts family_name -> (L, N, 22, H)."""
    acts_dir = _acts_dir(model, model_size)
    train_sources = {s for _, ss in families_train for s in ss}
    test_sources = {s for _, ss in families_test for s in ss}
    cache = {}

    def load(source):
        if source not in cache:
            path = os.path.join(acts_dir, source + ".pt")
            if not os.path.exists(path):
                cache[source] = (None, None, None)
            else:
                tensor = torch.load(path, map_location="cpu").float()
                train_idx, test_idx = _split_indices(tensor.shape[1], test_ratio, seed, source)
                cache[source] = (tensor, train_idx, test_idx)
        return cache[source]

    def assemble(config, side):
        other_sources = test_sources if side == "train" else train_sources
        out = {}
        for cluster, sources in config:
            slices = []
            for source in sources:
                tensor, train_idx, test_idx = load(source)
                if tensor is None:
                    continue
                if source in other_sources:
                    idx = train_idx if side == "train" else test_idx
                else:
                    idx = np.arange(tensor.shape[1])
                if len(idx):
                    slices.append(tensor[:, idx])  # (L, n, 22, H)
            if slices:
                out[cluster] = torch.cat(slices, dim=1)
        return out

    return assemble(families_train, "train"), assemble(families_test, "test")


def gen_buckets_thinking(model, model_size, bucket_config):
    """Compose thinking train/test buckets -> {"train": {...}, "test": {...}} of (L, N, 22, H)."""
    path = bucket_config if os.path.isabs(bucket_config) else os.path.join(HERE, bucket_config)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    train_acts, test_acts = build_splits_thinking(
        model, model_size,
        families_train=cfg["families_train"], families_test=cfg["families_test"],
        test_ratio=cfg.get("test_ratio", 0.2), seed=cfg.get("seed", 0))
    return {"train": train_acts, "test": test_acts}


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "0.5b"

    bucket_config = "bucket_config.json"
    buckets = gen_buckets(model, model_size, bucket_config)
    for side in ("train", "test"):
        print(f"[{side}]")
        for cluster, tensor in buckets[side].items():
            print(f"  {cluster:26s}: {tuple(tensor.shape)}")
