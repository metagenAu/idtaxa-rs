#!/usr/bin/env bash
# Example: combine an in-house soil metabarcoding database with public
# datasets generated with different primer sets.
#
# Inputs (edit paths):
#   refs/silva_nr99.fa.gz      SILVA (or GTDB) 16S reference with lineages in headers
#   refs/unite.fa.gz           UNITE general release (ITS)
#   refs/inhouse_16S.fa        your curated sequences, same lineage format
#   refs/inhouse_ITS.fa
#   studies/<study>/asvs.fa    ASV sequences (primers removed), one directory per study
#   studies/<study>/table.tsv  ASV x sample count table
#   studies/<study>/R1.fastq.gz, R2.fastq.gz   a sample of raw reads (optional)
#   studies/<study>/primers    a file containing the primer pair name (written below if detected)
set -euo pipefail

THREADS=${THREADS:-0}
RANK=${RANK:-genus}
mkdir -p models results

# 1. One classifier per primer set, from the same reference release + in-house sequences.
cat refs/silva_nr99.fa.gz <(gzip -c refs/inhouse_16S.fa) > refs/combined_16S.fa.gz
cat refs/unite.fa.gz <(gzip -c refs/inhouse_ITS.fa) > refs/combined_ITS.fa.gz

for P in 515F-806R 341F-806R 27F-519R; do
  [ -s models/16S_$P.idtx ] && continue
  idtaxa -t "$THREADS" trim -i refs/combined_16S.fa.gz --primers "$P" -o /dev/null \
      --coverage results/coverage_$P.tsv --coverage-rank phylum
  idtaxa -t "$THREADS" train -r refs/combined_16S.fa.gz --primers "$P" \
      -o models/16S_$P.idtx --problems results/problems_16S_$P.tsv
done
for P in fITS7-ITS4 ITS3-ITS4; do
  [ -s models/ITS_$P.idtx ] && continue
  # UNITE lacks the LSU flank where ITS4 binds: keep forward-only amplicons
  idtaxa -t "$THREADS" train -r refs/combined_ITS.fa.gz --primers "$P" --partial \
      -o models/ITS_$P.idtx --problems results/problems_ITS_$P.tsv
done

# 2. Per study: detect primers if unknown, classify with the matching model, summarise.
out=results/all_studies_${RANK}_long.tsv
header=1
: > "$out"
for dir in studies/*/; do
  study=$(basename "$dir")
  if [ ! -s "$dir/primers" ] && [ -s "$dir/R1.fastq.gz" ]; then
    idtaxa detect-primers "$dir/R1.fastq.gz" "$dir/R2.fastq.gz" -n 5000 2> "$dir/primer_detection.log" >/dev/null || true
    grep -m1 "consistent with primer pair" "$dir/primer_detection.log" | awk '{print $5}' > "$dir/primers" || true
  fi
  P=$(cat "$dir/primers" 2>/dev/null || true)
  model=$(ls models/*_"$P".idtx 2>/dev/null | head -1 || true)
  if [ -z "$model" ]; then
    echo "skipping $study: no model for primer set '$P'" >&2
    continue
  fi
  idtaxa -t "$THREADS" classify -m "$model" -i "$dir/asvs.fa" -o "$dir/taxonomy.tsv" --strand top
  extra=()
  [ $header -eq 0 ] && extra=(--no-header)
  idtaxa summarize -c "$dir/taxonomy.tsv" --counts "$dir/table.tsv" --rank "$RANK" \
      --long --study "$study" --primer-set "$P" "${extra[@]}" >> "$out"
  header=0
done
echo "wrote $out"
