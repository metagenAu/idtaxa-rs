#!/bin/bash
# Recommended end-to-end workflow (see README, "Recommended configuration"):
#   extract-region -> clean-ref -> train -> calibrate -> classify
#   (calibrated, novelty flag, adaptive bootstraps) -> placement on the
#   flagged subset -> consensus
#
# Usage: workflow.sh reference.fasta[.gz] reference_taxonomy.tsv asvs.fasta[.gz] outdir [refpkg_dir]
# The reference taxonomy TSV may be omitted ("-") when FASTA headers hold the
# lineage (DADA2-formatted SILVA/PR2). Primers default to 18S V4
# (TAReuk454FWD1 / TAReukREV3); override with FWD=... REV=... env variables.
set -euo pipefail
REF=$1; TAX=$2; ASVS=$3; OUT=$4; PKG=${5:-}
FWD=${FWD:-CCAGCASCYGCGGTAATTCC}
REV=${REV:-ACTTTCGTTCTTGATYRA}
THREADS=${THREADS:-$(nproc)}
mkdir -p "$OUT"
taxarg=(); [ "$TAX" != "-" ] && taxarg=(--taxonomy "$TAX")

# 1. amplicon region (primers trimmed), with rescue of partial references
idtaxa extract-region --ref "$REF" "${taxarg[@]}" --fwd "$FWD" --rev "$REV" \
  --max-mismatch 2 --min-len 300 --max-len 550 --rescue \
  --output "$OUT/region.fasta" --report "$OUT/region_report.tsv"
regtax=(); [ "$TAX" != "-" ] && regtax=(--taxonomy "$TAX")

# 2. label cleaning: truncate placeholder ranks (problem sequences are only
#    reported for review: dropping them did not help in docs/benchmarks.md)
idtaxa clean-ref --ref "$OUT/region.fasta" "${regtax[@]}" --drop-placeholder-labels \
  --output-prefix "$OUT/cleaned"

# 3. held-out splits for calibration (random) and novelty (clade exclusion at genus)
idtaxa split --ref "$OUT/cleaned.fasta" --taxonomy "$OUT/cleaned_taxonomy.tsv" \
  --mode random --fraction 0.1 --output-prefix "$OUT/rand"
idtaxa split --ref "$OUT/rand_train.fasta" --taxonomy "$OUT/rand_train.tsv" \
  --mode clade --rank "${CLADE_RANK:-8}" --fraction 0.05 --output-prefix "$OUT/clade"

# 4. calibration model (trained without either held-out set) ...
idtaxa train --ref "$OUT/clade_train.fasta" --taxonomy "$OUT/clade_train.tsv" --output "$OUT/calib_model.idtx" --threads "$THREADS"
idtaxa validate --model "$OUT/calib_model.idtx" --holdout "$OUT/rand_holdout.fasta" --taxonomy "$OUT/rand_holdout.tsv" \
  --output-prefix "$OUT/validate"
# calibrate on random + clade-exclusion held-out sequences so thresholds
# account for taxa missing from the reference
cat "$OUT/rand_holdout.fasta" "$OUT/clade_holdout.fasta" > "$OUT/calib_holdout.fasta"
cat "$OUT/rand_holdout.tsv" "$OUT/clade_holdout.tsv" > "$OUT/calib_holdout.tsv"
idtaxa calibrate --model "$OUT/calib_model.idtx" --holdout "$OUT/calib_holdout.fasta" --taxonomy "$OUT/calib_holdout.tsv" \
  --novelty-holdout "$OUT/clade_holdout.fasta" --novelty-taxonomy "$OUT/clade_holdout.tsv" \
  --target-error "${TARGET_ERROR:-0.05}" --output "$OUT/calibration.json" --reliability "$OUT/reliability.tsv"

# 5. classify all ASVs with the calibrated model. Adaptive bootstraps are
#    available (ADAPTIVE=1000) but gave no measurable gain in the benchmarks.
adapt=(); [ -n "${ADAPTIVE:-}" ] && adapt=(--adaptive-bootstraps "$ADAPTIVE")
idtaxa classify --model "$OUT/calib_model.idtx" --input "$ASVS" --output "$OUT/asv_taxonomy.tsv" \
  --calibration "$OUT/calibration.json" --novelty-flag "${adapt[@]}" --full-confidences \
  --threads "$THREADS" --cache "$OUT/cache.redb"

# 6. placement on the novelty-flagged subset + consensus (needs a reference package)
if [ -n "$PKG" ]; then
  "$(dirname "$0")/run_placement.sh" "$OUT/asv_taxonomy.tsv" "$ASVS" "$PKG" "$OUT/placement" novel
  idtaxa consensus --idtaxa "$OUT/asv_taxonomy.tsv" --placement "$OUT/placement/per_query.tsv" \
    --jplace "$OUT/placement/epa_result.jplace" --taxonomy "$PKG/taxonomy.tsv" \
    --placement-min-lwr 0.9 --max-pendant 0.1 --output "$OUT/consensus.tsv"
fi
echo "workflow complete: $OUT/asv_taxonomy.tsv"
