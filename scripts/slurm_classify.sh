#!/bin/bash
# Classify millions of ASVs as a SLURM array job: each task processes one
# deterministic shard (--chunk i/n); a dependent job merges the shards.
# Results are identical to a single run (per-sequence RNG seeding).
#
#   sbatch --array=1-50 scripts/slurm_classify.sh model.idtx asvs.fasta.gz outdir
#   sbatch --dependency=afterok:<jobid> --wrap "idtaxa merge --inputs outdir/part_*.tsv --output outdir/all.tsv"
#
#SBATCH --job-name=idtaxa
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=idtaxa_%A_%a.log
set -euo pipefail
MODEL=$1; INPUT=$2; OUT=$3
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-1}
# array indices must be 1..N
mkdir -p "$OUT"
idtaxa classify --model "$MODEL" --input "$INPUT" \
  --output "$OUT/part_${I}.tsv" --chunk "${I}/${N}" \
  --threads "${SLURM_CPUS_PER_TASK:-1}" \
  --cache "$OUT/cache_${I}.redb"   # optional: reuse results across reruns
