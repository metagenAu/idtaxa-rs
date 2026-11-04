# Phase 6 setup

Pool construction (Python, from the extracted PR2 V4 regions `pr2_v4.fasta` and from `pr2_version_5.0.0_SSU_dada2.fasta.gz` for 6.1):

```python
# natural-distribution pool: 15,000 random V4 regions (seed 5)
import random; rng = random.Random(5)
recs = read_fasta("pr2_v4.fasta")            # (lineage header, sequence)
rng.shuffle(recs); pool = recs[:15000]      # -> pool.fasta / pool.tsv (ids p0..)
# 6.1: 15,000 random full-length references (seed 9) -> full.fasta / full.tsv (ids f0..)
```

Experiment driver (`run_p6.sh`), followed by the mixed-calibration runs:

```bash
#!/bin/bash
# Phase 6 experiments on a natural-distribution PR2 V4 pool (15k sequences).
set -e
B=/home/user/idtaxa-rs/target/release/idtaxa
export RUST_BACKTRACE=0
TH=0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99
# splits: genus exclusion (10% of genera) then random 20% of the rest
$B -q split --ref pool.fasta --taxonomy pool.tsv --mode clade --rank 8 --fraction 0.1 --seed 3 --output-prefix c
$B -q split --ref c_train.fasta --taxonomy c_train.tsv --mode random --fraction 0.2 --seed 4 --output-prefix r
$B -q split --ref r_holdout.fasta --taxonomy r_holdout.tsv --mode random --fraction 0.5 --seed 5 --output-prefix rh   # rh_train=cal, rh_holdout=eval
$B -q split --ref c_holdout.fasta --taxonomy c_holdout.tsv --mode clade --rank 8 --fraction 0.5 --seed 6 --output-prefix ch # ch_train=cal, ch_holdout=eval
for f in r_train rh_train rh_holdout ch_train ch_holdout; do echo "$f $(grep -c '>' $f.fasta)"; done
# models
$B -q train --ref r_train.fasta --taxonomy r_train.tsv --output std.idtx
$B -q train --ref r_train.fasta --taxonomy r_train.tsv --output bal5.idtx --mode balanced --max-group-size 5
$B -q train --ref r_train.fasta --taxonomy r_train.tsv --output spaced.idtx --mode spaced
$B info --model std.idtx | grep -E "^K|training|problem"
# calibration (random cal set + clade-exclusion cal set for novelty)
$B -q calibrate --model std.idtx --holdout rh_train.fasta --taxonomy rh_train.tsv --novelty-holdout ch_train.fasta --novelty-taxonomy ch_train.tsv --target-error 0.05 --output cal.json --reliability reliability.tsv
for set in rh_holdout:random ch_holdout:clade; do
  f=${set%%:*}; n=${set##*:}
  V="--model std.idtx --holdout $f.fasta --taxonomy $f.tsv --thresholds $TH --ci-replicates 1000 -q"
  $B validate $V --output-prefix ${n}_balscore --compare-mode standard,balanced
  $B validate $V --output-prefix ${n}_balscore3 --compare-mode standard,balanced --balance-exponent 0.3
  $B validate $V --output-prefix ${n}_baltrain --compare-model bal5.idtx
  $B validate $V --output-prefix ${n}_spaced --compare-model spaced.idtx
  $B validate $V --output-prefix ${n}_adaptive --compare-adaptive 1000
  $B validate $V --output-prefix ${n}_calibrated --compare-calibration cal.json --compare-calibrated-thresholds
  $B validate $V --output-prefix ${n}_novelty --compare-calibration cal.json --compare-truncate-novel
done
echo P6DONE
# ---- 6.1 region extraction: full-length vs region-trained model on V4 queries
$B -q extract-region --ref full.fasta --taxonomy full.tsv --fwd CCAGCASCYGCGGTAATTCC --rev ACTTTCGTTCTTGATYRA --max-mismatch 2 --min-len 300 --max-len 550 --output full_region.fasta --report full_region_report.tsv
$B extract-region --ref full.fasta --taxonomy full.tsv --fwd CCAGCASCYGCGGTAATTCC --rev ACTTTCGTTCTTGATYRA --max-mismatch 2 --min-len 300 --max-len 550 --rescue --output full_region_rescue.fasta --report full_region_rescue_report.tsv 2> rescue_summary.txt
python3 - <<'PY'
ids=[l[1:].strip() for l in open("full_region.fasta") if l.startswith(">")]
tax=dict(l.rstrip("\n").split("\t") for l in open("full.tsv"))
import random; rng=random.Random(2); rng.shuffle(ids)
hold=set(ids[:len(ids)//5])
def sub(src,dst,keep):
    w=False
    with open(dst,"w") as o:
        for l in open(src):
            if l.startswith(">"): w=l[1:].strip() in keep
            if w: o.write(l)
allfull=set(tax)
train=allfull-hold
sub("full.fasta","e61_train_full.fasta",train)
sub("full_region_rescue.fasta","e61_train_region.fasta",train)
sub("full_region.fasta","e61_hold.fasta",hold)
for name,keep in (("e61_train",train),("e61_hold",hold)):
    with open(name+".tsv","w") as o:
        for i in sorted(keep): o.write(f"{i}\t{tax[i]}\n")
PY
$B -q train --ref e61_train_full.fasta --taxonomy e61_train.tsv --output e61_full.idtx
$B -q train --ref e61_train_region.fasta --taxonomy e61_train.tsv --output e61_region.idtx
$B validate --model e61_full.idtx --compare-model e61_region.idtx --holdout e61_hold.fasta --taxonomy e61_hold.tsv --thresholds $TH --output-prefix e61 -q
# ---- 6.2 label cleaning (truth = cleaned labels for both models)
$B -q clean-ref --ref r_train.fasta --taxonomy r_train.tsv --drop-placeholder-labels --output-prefix e62_clean
$B -q clean-ref --ref r_train.fasta --taxonomy r_train.tsv --drop-placeholder-labels --retrain-rounds 2 --output-prefix e62_cleanps
$B -q clean-ref --ref rh_holdout.fasta --taxonomy rh_holdout.tsv --drop-placeholder-labels --output-prefix e62_hold
$B -q train --ref e62_clean.fasta --taxonomy e62_clean_taxonomy.tsv --output e62_clean.idtx
$B -q train --ref e62_cleanps.fasta --taxonomy e62_cleanps_taxonomy.tsv --output e62_cleanps.idtx
$B validate --model std.idtx --compare-model e62_clean.idtx --holdout e62_hold.fasta --taxonomy e62_hold_taxonomy.tsv --thresholds $TH --output-prefix e62_labels -q
$B validate --model std.idtx --compare-model e62_cleanps.idtx --holdout e62_hold.fasta --taxonomy e62_hold_taxonomy.tsv --thresholds $TH --output-prefix e62_problems -q
wc -l e62_clean_changes.tsv e62_cleanps_problem_sequences.tsv e62_clean_conflicts.tsv
echo E6XDONE
# mixed calibration set (random + clade-exclusion halves)
cat rh_train.fasta ch_train.fasta > calmix.fasta; cat rh_train.tsv ch_train.tsv > calmix.tsv
$B calibrate --model std.idtx --holdout calmix.fasta --taxonomy calmix.tsv \
  --novelty-holdout ch_train.fasta --novelty-taxonomy ch_train.tsv --target-error 0.05 \
  --output calmix.json --reliability reliability_mix.tsv
for set in rh_holdout:random ch_holdout:clade; do f=${set%%:*}; n=${set##*:}
  $B validate --model std.idtx --holdout $f.fasta --taxonomy $f.tsv --thresholds $TH \
    --output-prefix ${n}_calmix --compare-calibration calmix.json --compare-calibrated-thresholds
done
# matched-error tables
python3 scripts/matched_error.py random_spaced.tsv --targets 0.02,0.05 --depths 6,7,8,9
```
