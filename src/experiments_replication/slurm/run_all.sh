#!/bin/bash
# Master script: submit all replication jobs with SLURM dependencies (KISSKI).
#
# Execution order:
#   00 → 01 → (02, 03+04, 05, 06, 07, 08) in parallel
#
# PREREQUISITES (run once on the LOGIN NODE first):
#   1. bash setup_venv.sh                 # build personal venv + pip install
#   2. pre-download models (login node)   # compute nodes have no internet
#   3. sbatch smoke_test.slurm            # verify env before burning hours
#
# Usage: cd src/experiments_replication/slurm && bash run_all.sh
# Monitor: squeue --me

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/user/m.lopezdeipinamuno/u27880/ai_safety_project
LOGDIR="$REPO/src/experiments_replication/logs"

# SLURM opens --output/--error before the job runs, so the log dir must exist
# at submit time or jobs fail instantly with no log. Create it once here.
mkdir -p "$LOGDIR"

echo "Submitting replication pipeline..."

J00=$(sbatch --parsable "$SLURM_DIR/00_collect_behaviors.slurm")
echo "  [J00=$J00] 00_collect_behaviors"

J01=$(sbatch --parsable --dependency=afterok:$J00 "$SLURM_DIR/01_extract_directions.slurm")
echo "  [J01=$J01] 01_extract_directions (after J00)"

J02=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/02_table1.slurm")
echo "  [J02=$J02] 02_table1 (after J01)"

J0304=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/03_04_figures23.slurm")
echo "  [J0304=$J0304] 03_04_figures23 (after J01)"

J05=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/05_figure4.slurm")
echo "  [J05=$J05] 05_figure4 (after J01)"

J06=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/06_figure5.slurm")
echo "  [J06=$J06] 06_figure5 (after J01)"

J07=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/07_figure6.slurm")
echo "  [J07=$J07] 07_figure6 (after J01)"

J08=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/08_latent_guard.slurm")
echo "  [J08=$J08] 08_latent_guard (after J01)"

echo ""
echo "All jobs submitted. Results in src/experiments_replication/results/"
echo "Logs in $LOGDIR"
echo ""
echo "Monitor:"
echo "  squeue --me"
echo "  tail -f $LOGDIR/00_collect_behaviors_*.out"
