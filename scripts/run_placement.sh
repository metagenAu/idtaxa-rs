#!/bin/bash
# Phylogenetic placement of a subset of ASVs (e.g. those flagged novel by
# `idtaxa classify --novelty-flag`) with hmmalign -> EPA-ng -> gappa, for use
# with `idtaxa consensus`. Placement is the expensive step, so it is run on a
# subset only ("fast method on all, placement on a subset").
#
# Requirements on PATH: hmmalign (HMMER), epa-ng, gappa, and a reference
# package made from the SAME taxonomy as the idtaxa model:
#   ref_alignment.fasta  reference MSA
#   ref.hmm              HMM built from it (hmmbuild ref.hmm ref_alignment.fasta)
#   ref.tree             reference tree (e.g. RAxML-NG best tree)
#   ref.model            RAxML-NG model file (e.g. ref.raxml.bestModel)
#   taxonomy.tsv         id<TAB>lineage of the reference sequences
#
# Usage: run_placement.sh idtaxa_out.tsv asvs.fasta refpkg_dir outdir [select]
#   select = novel (default: novelty_rank != none) | unassigned (empty taxonomy) | all
set -euo pipefail
IDT=$1; ASVS=$2; PKG=$3; OUT=$4; SEL=${5:-novel}
mkdir -p "$OUT"
python3 - "$IDT" "$ASVS" "$SEL" "$OUT/subset.fasta" <<'PY'
import sys
idt, asvs, sel, out = sys.argv[1:]
want = set()
with open(idt) as f:
    hdr = None
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = {k: i for i, k in enumerate(p)}
            continue
        if sel == "all":
            want.add(p[0])
        elif sel == "unassigned" and p[hdr["taxonomy"]] == "":
            want.add(p[0])
        elif sel == "novel" and "novelty_rank" in hdr and p[hdr["novelty_rank"]] not in ("", "none"):
            want.add(p[0])
n = 0
with open(asvs) as f, open(out, "w") as o:
    keep = False
    for line in f:
        if line.startswith(">"):
            keep = line[1:].split()[0] in want
            n += keep
        if keep:
            o.write(line)
print(f"selected {n} sequences", file=sys.stderr)
PY
hmmalign --outformat afa --mapali "$PKG/ref_alignment.fasta" "$PKG/ref.hmm" "$OUT/subset.fasta" > "$OUT/combined.afa"
epa-ng --split "$PKG/ref_alignment.fasta" "$OUT/combined.afa" --out-dir "$OUT"
epa-ng --ref-msa "$OUT/reference.fasta" --tree "$PKG/ref.tree" --query "$OUT/query.fasta" \
       --model "$PKG/ref.model" --outdir "$OUT" --redo
gappa examine assign --jplace-path "$OUT/epa_result.jplace" --taxon-file "$PKG/taxonomy.tsv" \
      --per-query-results --out-dir "$OUT" --allow-file-overwriting
echo "placement done: $OUT/per_query.tsv and $OUT/epa_result.jplace"
