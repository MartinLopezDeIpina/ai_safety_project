#!/usr/bin/env python
"""Verify that a thinking activation `.pt` stores the right token at each of its 22 slots.

For one classified-generation file and one row index, this reconstructs the exact
`prompt + ori_output` token sequence (the same way extract_hidden.py does), and for each of the 22
activation slots prints:
  - the token that slot points to (read from the query's token sequence), and
  - whether the stored activation equals a fresh forward pass at that token index (they must MATCH).
Null slots (e.g. all CoT slots in gennothink) must store zeros.

There is no argparse (matching the rest of the pipeline): set the parameters in the __main__ block
at the bottom, then run `python verify_activation_tokens.py`.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)  # utils, extract_hidden live in src/

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import formatInp_thinking                     # exact prompt builder used by extraction
from extract_hidden import compute_positions_thinking    # exact 22-slot index computation

ROLES = {0: "t_inst", 1: "t_post/assist", 2: "\\n", 3: "<think>", 4: "\\n",
         5: "CoT_first1", 6: "CoT_first2", 7: "CoT_first3", 8: "CoT_first4", 9: "CoT_first5",
         10: "CoT_mid1", 11: "CoT_mid2", 12: "CoT_mid3", 13: "CoT_mid4", 14: "CoT_mid5",
         15: "CoT_last1", 16: "CoT_last2", 17: "CoT_last3", 18: "CoT_last4", 19: "CoT_last5",
         20: "gen1", 21: "gen2"}


def verify(classified, row_id, full=False, model="qwen35", size="0.8b"):
    """Verify one row's activation slots.

    classified : path to a *_gentype_accepted/refused.json under classified_generations/ (relative
                 paths resolve against this script's directory). The matching .pt is found under
                 activations/ automatically; mode (genthink/gennothink) is read from the filename.
    row_id     : row index into that file.
    full       : also print the entire indexed token sequence of the query.
    model/size : Qwen3.5 family/size (weights cached under src/models/<model>).
    """
    cls_path = classified if os.path.isabs(classified) else os.path.join(HERE, classified)
    cls_path = os.path.abspath(cls_path)
    stem = os.path.splitext(os.path.basename(cls_path))[0]
    mode = "genthink" if "genthink" in stem else "gennothink"
    pt_path = cls_path.replace(f"{os.sep}classified_generations{os.sep}",
                               f"{os.sep}activations{os.sep}")
    pt_path = os.path.splitext(pt_path)[0] + ".pt"
    for p in (cls_path, pt_path):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    rows = json.load(open(cls_path))
    if not (0 <= row_id < len(rows)):
        sys.exit(f"id {row_id} out of range (file has {len(rows)} rows)")
    row = rows[row_id]
    pt = torch.load(pt_path, map_location="cpu").float()  # (L, N, 22, H)
    assert pt.shape[1] == len(rows), f"row mismatch: pt {tuple(pt.shape)} vs {len(rows)} rows"

    mid = f"Qwen/Qwen3.5-{size.upper()}"
    cache = os.path.join(SRC, "models", model)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, cache_dir=cache)
    lm = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map="auto",
                                              trust_remote_code=True, cache_dir=cache).eval()

    # Reconstruct the exact sequence the extractor saw, and recompute hidden states.
    full_text = formatInp_thinking(row, thinking_mode=mode, model=model) + row.get("ori_output", "")
    enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    with torch.no_grad():
        out = lm(input_ids=enc.input_ids.to(lm.device), output_hidden_states=True)
    hs = torch.stack(out.hidden_states).squeeze(1).detach().cpu().half()  # (L, seq, H)
    idxs = compute_positions_thinking(ids.tolist(), tok, mode)
    stored = pt[:, row_id]  # (L, 22, H)

    query = row.get("instruction") or row.get("question") or row.get("bad_q") or ""
    print(f"file : {stem}")
    print(f"mode : {mode}   row id : {row_id}   seq_len : {hs.shape[1]}")
    print(f"query: {query!r}")

    if full:
        print("\n--- full query token sequence (idx : token) ---")
        for j, t in enumerate(tok.convert_ids_to_tokens(ids.tolist())):
            print(f"  {j:>5} {t!r}")

    print("\n--- 22 activation slots: the query token each slot points to + activation check ---")
    print(f"{'slot':>4} {'role':<13} {'seq_idx':>7}  {'query token':<22} activation")
    ok = True
    for s in range(22):
        j = idxs[s]
        if j is None:                      # null slot -> stored must be zeros
            n = stored[:, s, :].norm().item()
            ok &= (n == 0.0)
            print(f"{s:>4} {ROLES[s]:<13} {'-':>7}  {'(NULL)':<22} "
                  f"stored {'zero OK' if n == 0 else f'NONZERO {n:.2f}'}")
        else:                              # real slot -> token from the sequence + activation match
            token = repr(tok.decode([ids[j].item()]))
            diff = (stored[:, s, :] - hs[:, j, :]).abs().max().item()
            ok &= (diff < 1e-2)
            print(f"{s:>4} {ROLES[s]:<13} {j:>7}  {token:<22} "
                  f"{'MATCH' if diff < 1e-2 else 'MISMATCH'} (maxdiff {diff:.1e})")

    print("\nRESULT:", "ALL SLOTS MATCH" if ok else "DISCREPANCY")
    return ok


if __name__ == "__main__":
    # ---------------- configure here, then run `python verify_activation_tokens.py` ----------------
    CLASSIFIED = "output/qwen350.8b/datasets_outputs/classified_generations/advbench_genthink_refused.json"
    ROW_ID = 0
    FULL = False           # True also dumps the whole indexed token sequence
    MODEL = "qwen35"
    SIZE = "0.8b"
    # -----------------------------------------------------------------------------------------------
    verify(CLASSIFIED, ROW_ID, full=FULL, model=MODEL, size=SIZE)
