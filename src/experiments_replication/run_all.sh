#!/usr/bin/env bash
# Run the full replication pipeline in order.
# Each step saves outputs to results/ so subsequent steps can load them.
#
# Estimated total time on RTX 5080 (16GB VRAM, Qwen2.5-1.5B-Instruct):
#   00  ~30 min   (inference on ~600 prompts)
#   01  ~20 min   (forward passes, 8 category × position combos)
#   02   ~5 min   (inference on 100 prompts)
#   03  <1 min    (reads saved activations, no GPU needed)
#   04  <1 min    (reads saved activations, no GPU needed)
#   05  ~30 min   (29 layers × 2 dirs × 30 prompts)
#   06  ~60 min   (29 layers × 4 conditions × 30 prompts)
#   07  ~10 min   (extracts jailbreak activations + scatter)
#   08  <1 min    (reads saved activations)

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

echo "=================================================================="
echo " Step 0/8 — Collect behaviors"
echo "=================================================================="
python 00_collect_behaviors.py

echo "=================================================================="
echo " Step 1/8 — Extract hidden states and directions"
echo "=================================================================="
python 01_extract_directions.py

echo "=================================================================="
echo " Step 2/8 — Table 1 (post-inst token ablation)"
echo "=================================================================="
python 02_table1_post_inst_tokens.py

echo "=================================================================="
echo " Step 3/8 — Figure 2 (layer-wise sl clustering)"
echo "=================================================================="
python 03_figure2_clustering.py

echo "=================================================================="
echo " Step 4/8 — Figure 3 (Δharmful × Δrefuse scatter)"
echo "=================================================================="
python 04_figure3_scatter.py

echo "=================================================================="
echo " Step 5/8 — Figure 4 (per-layer steering on harmless)"
echo "=================================================================="
python 05_figure4_steering_layers.py

echo "=================================================================="
echo " Step 6/8 — Figure 5 (reply inversion, per-layer)"
echo "=================================================================="
python 06_figure5_reply_inversion.py

echo "=================================================================="
echo " Step 7/8 — Figure 6 (jailbreak scatter)"
echo "=================================================================="
python 07_figure6_jailbreak.py

echo "=================================================================="
echo " Step 8/8 — Table 3 (Latent Guard)"
echo "=================================================================="
python 08_latent_guard.py

echo ""
echo "=================================================================="
echo " Done! All outputs are in src/experiments_replication/results/"
echo "=================================================================="
ls -lh results/*.png results/*.json 2>/dev/null || true
