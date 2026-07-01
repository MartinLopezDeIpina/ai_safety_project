#!/bin/bash
# Master script: submit all replication jobs with SLURM dependencies.
#
# Execution order:
#   00 → 01 → (02, 03+04, 05, 06, 07, 08) in parallel
#
# Usage: cd experiments_replication/slurm && bash run_all.sh
#
# After submission, monitor with: squeue -u $USER

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/mnt/beegfs/home/stud127/LLMs_Encode_Harmfulness_Refusal_Separately
LOGDIR="$BASE/src/experiments_replication/logs"

# SLURM opens the --output/--error files BEFORE the job script runs, so the
# log directory must exist at submit time or every job fails instantly with
# no log explaining why. Create it once here.
mkdir -p "$LOGDIR"

echo "Submitting replication pipeline..."

# Step 1: Collect behaviors (prerequisite for everything)
J00=$(sbatch --parsable "$SLURM_DIR/00_collect_behaviors.sh")
echo "  [J00=$J00] 00_collect_behaviors"

# Step 2: Extract directions (requires behaviors.json from step 1)
J01=$(sbatch --parsable --dependency=afterok:$J00 "$SLURM_DIR/01_extract_directions.sh")
echo "  [J01=$J01] 01_extract_directions (after J00)"

# Steps 3-8: All depend only on step 2 → run in parallel
J02=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/02_table1.sh")
echo "  [J02=$J02] 02_table1 (after J01)"

J0304=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/03_04_figures23.sh")
echo "  [J0304=$J0304] 03_04_figures23 (after J01)"

J05=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/05_figure4.sh")
echo "  [J05=$J05] 05_figure4 (after J01)"

J06=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/06_figure5.sh")
echo "  [J06=$J06] 06_figure5 (after J01)"

J07=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/07_figure6.sh")
echo "  [J07=$J07] 07_figure6 (after J01)"

J08=$(sbatch --parsable --dependency=afterok:$J01 "$SLURM_DIR/08_latent_guard.sh")
echo "  [J08=$J08] 08_latent_guard (after J01)"

echo ""
echo "All jobs submitted. Results will be in experiments_replication/results/"
echo "Logs will be in $LOGDIR"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f $LOGDIR/00_collect_behaviors_*.out"
