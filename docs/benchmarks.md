# Benchmarks

All measurements on one VM: 4 vCPU Intel Xeon @ 2.8 GHz, 15 GB RAM,
Ubuntu 24.04. R 4.3.3 with DECIPHER 3.9.4 built from source. idtaxa-rs
was built with `cargo build --release`. Reference data: **PR2 v5.0.0**
(DADA2 format, GitHub release), 18S V4 region extracted with idtaxa-rs
`extract-region` (TAReuk454FWD1 `CCAGCASCYGCGGTAATTCC` / TAReukREV3
`ACTTTCGTTCTTGATYRA`, ≤ 2 mismatches, 300–500 bp). This gave 115,763
regions from 221,085 references. PR2 has 9 ranks: domain, supergroup,
division, subdivision, class, order, family, genus, species (tables
abbreviate them).

Raw reports are in [`docs/results/`](results/). Every number can be
regenerated with the scripts in `scripts/`.

---

## Phase 1: k-mers and model import

* `scripts/export_decipher_model.R` → `idtaxa import-decipher` →
  `idtaxa kmer-check` on the 8,000-sequence parity reference: **8,000 of
  8,000 training k-mer sets equal DECIPHER's `kmers`**, 0 mismatches.
* `info` reports the same counts as DECIPHER's `print(trainingSet)`:
  K = 9, 8,000 sequences, 11,428 taxa, 2 problem groups, 269 problem
  sequences.
* Minimum sample size (`minS`): 6 in both R and Rust (10 for DECIPHER's
  bundled `TrainingSet_16S`).
* Round-trip tests: `crates/idtaxa-core/tests/engine.rs` (save → mmap load →
  identical k-mers, taxonomy and classifications; model ID stable).

## Phase 2: classification parity

**Data** (`scripts/prepare_parity_data.py`, seed 1): 10,000 PR2 V4
references sampled across 6,847 species (≤ 3 per species), split 8,000
training / 2,000 held out. DECIPHER `LearnTaxa` (defaults; automatic
K = 9) trained the model, which was exported and imported. There are
10,000 queries:

| subset | n |
|---|---|
| held-out references, exact | 2,000 |
| training references, exact | 2,000 |
| held-out, 1% substitutions/indels | 2,000 |
| held-out, 3% substitutions/indels | 2,000 |
| training, 5% substitutions/indels | 1,000 |
| held-out, 1% mutated, reverse-complemented | 1,000 |

DECIPHER `IdTaxa(strand = "both", threshold = 0, processors = NULL)` ran
with `set.seed(1)` and `set.seed(2)`; idtaxa-rs ran with default
parameters, `--full-confidences`, `--seed 42`. Agreement is measured at
`threshold = 60` (`scripts/compare.py`). Full table:
[`results/parity_pr2_v4_10k.md`](results/parity_pr2_v4_10k.md).

| rank | idtaxa-rs vs DECIPHER: agreement | mean \|Δconf\| | DECIPHER seed 1 vs seed 2: agreement | mean \|Δconf\| |
|---|---|---|---|---|
| domain | 98.20% | 1.09 | 98.10% | 1.09 |
| supergroup | 98.22% | 1.17 | 98.14% | 1.15 |
| division | 98.24% | 1.18 | 98.12% | 1.16 |
| subdivision | 98.22% | 1.18 | 98.11% | 1.16 |
| class | 98.15% | 1.24 | 98.09% | 1.23 |
| order | 98.14% | 1.37 | 98.07% | 1.34 |
| family | 97.97% | 1.44 | 97.90% | 1.40 |
| genus | 97.16% | 2.73 | 97.26% | 2.77 |
| species | 97.62% | 2.66 | 97.64% | 2.67 |
| **all ranks** | **97.99%** | **1.56** | **97.94%** | **1.55** |

* **Disagreements are concentrated at the threshold.** Of 1,808
  rank-level disagreements, 1,764 (97.6%) have a confidence within 5 points
  of 60 in one of the two runs, 33 are 5–10 points away, 11 are 10–20
  points away and none is more than 20 away. They split evenly between
  "assigned only by DECIPHER" (925) and "assigned only by idtaxa-rs" (883).
  None is a different taxon at the same rank.
* **Accuracy against the truth** is the same, for example genus correct /
  wrong / unassigned: DECIPHER 38.2 / 2.3 / 59.5%, idtaxa-rs 38.0 / 2.2 / 59.8%.
  Species: 28.7 / 4.6 / 66.7% vs 28.6 / 4.3 / 67.0%.
* **By subset**, agreement over all ranks (DECIPHER vs DECIPHER in brackets):
  exact 98.72% (98.94%), 1% mutated 97.97% (98.00%), 3% mutated 96.43%
  (96.01%), 5% mutated 97.96% (98.14%), reverse-complemented 98.16%
  (97.42%).
* At thresholds 50 and 70: 97.52% (DECIPHER baseline 97.44%) and 98.19%
  (98.15%).
* **idtaxa-rs seed 42 vs seed 7:** 97.89% agreement, mean |Δconf| 1.57.
* **Small fixture** (committed, checked in CI; 412 references, 576 queries):
  idtaxa-rs vs DECIPHER 99.04%, DECIPHER vs DECIPHER 98.46%; mean |Δconf|
  1.56 vs 1.53.

**Acceptance.** The brief's targets were ≥ 99% per rank and
mean |Δconf| ≤ 3, with run-to-run variability as the baseline. Mean
|Δconf| is 1.56, below 3. Per-rank agreement is 97.2–98.2%, which is
**equal to DECIPHER's own seed-to-seed agreement** (97.3–98.1%) on this
deliberately hard query set: 50% of the queries are mutated or held out,
so many sit near the threshold. 99% is therefore not attainable by
DECIPHER itself here. On the easier fixture both exceed 98.4%.
idtaxa-rs is statistically indistinguishable from a second DECIPHER run.

## Phase 3: performance

### idtaxa-rs vs DECIPHER, same hardware

The 10,000 parity queries were classified against the 8,000-sequence
DECIPHER-trained model, using 4 cores in both cases.

| tool | wall time | peak RSS |
|---|---|---|
| DECIPHER `IdTaxa(processors = NULL)` | 487 s (seed 1), 517 s (seed 2) | (R session) |
| idtaxa-rs `classify` | **22.9 s** | 58 MB |

That is about **21× faster**. Training on the same 8,000 sequences:
`LearnTaxa` took 308 s (single-threaded R), `idtaxa train` took 4.5 s.

### Thread scaling

10,000 ASV-like queries against a **50,000-sequence** PR2 V4 reference
(30,492 taxa; native training took 30 s). The queries are random PR2 V4
regions with 0–3% substitutions.

| threads | wall time | speed-up | queries/s | peak RSS |
|---|---|---|---|---|
| 1 | 322.9 s | 1.00× | 31 | 122 MB |
| 2 | 163.7 s | 1.97× | 61 | 136 MB |
| 4 | 100.2 s | 3.22× | 100 | 166 MB |

The output is byte-identical between 1 and 4 threads.

### Scaling with query count

4 threads, 8,000-sequence reference, default batch size 50,000. The
query set is a 1M-record file of PR2 V4 regions with 0–3% substitutions,
with natural duplication.

| queries | wall time | peak RSS |
|---|---|---|
| 10,000 | 26.7 s | 58 MB |
| 100,000 | 241.9 s | 123 MB |
| 1,000,000 | ONE_M_TIME | ONE_M_RSS |

Peak memory is bounded by the batch size, not the input size: 58 MB for
10k queries in a single partial batch, then flat once batches are full.
With `--batch-size 1000`, 100k queries ran in BATCH1000_RSS with
byte-identical output (BATCH1000_NOTE). The 50k reference at 100k queries
ran in Q100K50K. A cached re-run of 100k queries (`--cache`) took
CACHE_TIME and gave identical output. Loading the 50k model
(MODEL_SIZE file) with `info` takes INFO_TIME through mmap.

### Speed target

The target is 1M × 450 bp ASVs against 50–100k references in well under
an hour on 32 cores. Measured: 31 queries/s per core against the 50k
reference, on this deliberately difficult query mix with many divergent
queries that stop high in the tree. That is 1M / (31 × 32) ≈ 17 min on 32 cores at perfect
scaling, or about **21 min** at the 80% parallel efficiency measured on 4
cores. Against
100k references, posting lists double in length; expect about 30–35 min.

**Remaining bottleneck** (`perf`): the leaf phase. A query that stops high
in the tree is scored against tens of thousands of groups. idtaxa-rs
computes DECIPHER's per-replicate winners exactly, but avoids the full
replicates × groups matrix:

1. Two passes over the inverted index restricted to subtree ID ranges.
2. K-mers present in more than half of the reference are stored as
   complement lists. On the 50k reference 74% of the ~4.1M postings a
   query touched came from such k-mers; with complements, an estimated
   ~2.2M remain.
3. Exact dominance pruning against the leading group: only replicates in
   which a group could tie or beat the leader are scored.

Together these gave a 2.7× speed-up on the 50k reference over the first
working version (from 105 to 39 ms per query per thread). What remains is
memory-bound posting traversal and the exact scoring of candidate
replicates. The unpruned reference path is kept and tested for identical
output (`pruned_winners_equal_full_reference_path`).

`cargo bench -p idtaxa-core` (criterion) benchmarks k-mer encoding and
per-query classification (pruned vs reference path) on the fixture.

### Determinism and memory checks

Integration tests (`crates/idtaxa-cli/tests/cli.rs`): output is identical
for `--threads 1` vs `--threads 3 --batch-size 7`, for `--chunk 1/3, 2/3,
3/3` + `merge` vs a single run, and for cached vs uncached runs.

## Phase 4: native training

Same 8,000 references, `idtaxa train` (defaults) vs DECIPHER `LearnTaxa`
(defaults); see
[`results/compare_models_native_vs_decipher.json`](results/compare_models_native_vs_decipher.json):

| quantity | result |
|---|---|
| K (automatic) | 9 vs 9 |
| taxonomy nodes and order | identical (11,428) |
| decision k-mer sets | 4,230 of 4,234 nodes identical; mean Jaccard 0.99997 |
| decision weights (identical sets) | max difference 3.0e-18 |
| learned fractions | mean difference 1.2e-21; NA fractions identical (2 = 2) |
| IDF weights | max difference 0 |
| problem groups | 2 vs 2, both shared |
| problem sequences | 260 (native) vs 269 (DECIPHER); 190 shared (71% of DECIPHER's found; Jaccard 0.56). DECIPHER seed 1 vs seed 2: PS_BASELINE |
| classification of the 10k parity queries | **byte-identical** to the DECIPHER-trained model (same seed) |

Problem-sequence differences come from the random bootstrap in the
fraction-learning descent. They are borderline sequences and do not
affect fractions, decision k-mers or classification on this reference.
Native training is 4.5 s vs 308 s.

## Phase 5: validation tooling

`idtaxa split` (random or clade exclusion), `idtaxa validate` (per-rank
TP/MC/OC/UC/TN, accuracy, misclassification, overclassification and
underclassification rates, error rate, at any list of thresholds; TSV plus
`_summary.txt`), and paired comparison of two configurations
(`--compare-mode`, `--compare-model`, `--compare-calibration`,
`--compare-adaptive`, `--compare-truncate-novel`). Comparisons report
95% paired bootstrap CIs over held-out sequences (`_compare.tsv`).
`scripts/matched_error.py` compares configurations at matched error
rates. All Phase 6 numbers below come from these tools.

---

## Phase 6: accuracy extensions

**Setup** (commands in [`results/phase6_setup.md`](results/phase6_setup.md)):
15,000 PR2 V4 regions sampled at random. This keeps the natural group-size
imbalance: 7,281 species, the largest with 293 sequences, 67 species with
more than 20. The splits are:

* **clade exclusion:** 10% of genera removed from the reference. Half of
  them (709 sequences) are used for evaluation, the other half for
  calibration/novelty fitting.
* **random leave-out:** 20% of the remaining sequences. Half (1,356) are
  used for evaluation, half for calibration.
* **reference:** the remaining 10,843 sequences (K = 9).

In the clade-exclusion set every genus and species is absent from the
reference, so any genus/species assignment is an overclassification; the
error column is what matters there.

"Accuracy at matched error" is the best accuracy (TP / sequences whose
true taxon is in the reference) over thresholds 0–99 (step 5) whose error
rate, (MC + OC) / all, does not exceed the target. Values "@ 60" in the
paired comparisons are fractions of *all* held-out sequences correctly
assigned ("correct") or wrongly assigned ("error") at threshold 60.
Values with `*` have 95% paired bootstrap CIs excluding 0.

### Summary

| extension | random leave-out | clade exclusion (unseen genera) | verdict |
|---|---|---|---|
| 6.1 region extraction | neutral: −1.0 to −1.4 pt accuracy*, −0.3 to −0.8 pt error* at threshold 60; equal at matched error | – | optional (coverage cost, see below) |
| 6.2 label cleaning (placeholders) | **species error 15.4% → 8.7%*** at no accuracy cost; species accuracy at 5% error 28.7% → **41.3%** | – | **recommended** |
| 6.2 + problem-sequence removal (2 rounds) | upper-rank error +1.0 pt*, genus −1.8 pt* | – | not recommended |
| 6.3 calibration | reliable probabilities; per-rank thresholds hit the target where feasible | depends on the calibration mix, see below | **recommended** (with a novelty-aware calibration set) |
| 6.4 novelty truncation | genus error 7.2% → 4.4%*, accuracy 45.5% → 35.6%* | genus/species overclassification **28.8% → 13.8%***, 26.5% → 12.8%* | recommended **as a flag**; truncation is a trade-off |
| 6.5 placement consensus | not benchmarked (see below) | – | – |
| 6.6 adaptive bootstraps (1000) | no change at matched error; 30% of queries re-run, 4.6% of assignments change, 2.7× slower | no change | not recommended |
| 6.7 balanced scoring (γ = 0.1, 0.3) | **worse**: genus accuracy at 5% error 51.6% → 15.4% | small family gains at matched error | not recommended |
| 6.7 balanced training (≤ 5 per group) | worse: genus at 5% error 51.6% → 40.6% | **family at 5% error 43.7% → 62.8%** | trade-off; only for references dominated by a few over-sampled taxa and novel-rich samples |
| 6.8 spaced seeds (`1101101101101`) | small gains (family at 5% error 89.8% → 92.0%) | small losses (order 64.9% → 60.7%) | neutral; not recommended |

### 6.1 Amplicon-region extraction

Experiment: 15,000 random full-length PR2 references. Extraction kept 7,999
regions directly and rescued 2,226 more (`--rescue`); 4,449 had no primer
site and 326 failed the length filter. The model trained on regions
(8,626 training sequences) was compared with the model trained on
full-length references (13,401), both evaluated on V4 regions of 1,599
held-out sequences.

| rank | full-length model acc / error | region model acc / error |
|---|---|---|
| order | 77.7 / 1.2% | 76.8 / 1.0% |
| family | 75.8 / 2.4% | 74.7 / 2.1% |
| genus | 55.8 / 9.3% | 55.7 / 8.4% |
| species | 49.7 / 13.1% | 50.8 / 12.6% |

At matched error the two are equal within noise; species at 2% error is
12.8% vs 21.6%. **Coverage report** (taxa retained after extraction,
without → with rescue): genus 3,199 → 3,985 of 5,423, species 4,435 →
5,551 of 7,639. Most losses are references that do not span the V4
primers, which the full-length model keeps. On PR2 V4 the region model is
not measurably more accurate, so extraction is optional. Its main use is
references with very variable lengths (where K and IDF are skewed by
long sequences) and preventing primer-region k-mers from dominating.

### 6.2 Reference label cleaning

`clean-ref --drop-placeholder-labels` truncated 1,943 of 10,843 training
lineages: mostly `Genus_sp.` species and strain-numbered labels. No
homonyms or rank shifts were found in this PR2 subset. Evaluation truth is
the cleaned label for both models.

| | species error @ 60 | species accuracy at 2% / 5% error | genus accuracy @ 60 |
|---|---|---|---|
| standard model | 15.4% | 11.4% / 28.7% | 45.4% correct |
| cleaned labels | **8.7%** * | **20.8% / 41.3%** | 43.6% * |
| cleaned + problem sequences removed (2 rounds) | 9.0% * | 21.8% / 44.0% | 43.6% *, with upper-rank error +1.0 pt * |

Placeholder species labels attract confident assignments to
uninformative names. Removing them halves species-level error. Iterative
problem-sequence removal adds little at species and raises error at
upper ranks, because it removes valid but divergent references. Its review
file (`*_problem_sequences.tsv`, labelled vs predicted lineage) is useful
for manual curation instead.

### 6.3 Per-rank calibration

Reliability, genus rank, calibration set (`reliability.tsv`):

| raw confidence bin | mean raw | observed accuracy | mean calibrated |
|---|---|---|---|
| 10–20 | 0.15 | 0.57 | 0.60 |
| 30–40 | 0.36 | 0.74 | 0.76 |
| 50–60 | 0.55 | 0.87 | 0.87 |
| 70–80 | 0.75 | 0.91 | 0.92 |
| 90–100 | 0.97 | 1.00 | 1.00 |

Raw IDTAXA confidence is strongly **under-confident** for known taxa,
because confidence is a bootstrap-weighted *similarity*. Isotonic
calibration makes it a probability of being correct.

Per-rank thresholds for target error 5%:

* **Calibrated on random leave-out only:** at ranks 1–6 the thresholds fall
  to 0.3–8.5 and accuracy rises from 81–86% to 95–99.5%, with error 0.5–4.9%
  on the evaluation set. Genus and species get thresholds of 90.5 and
  100, which disables species assignment: species error is never ≤ 5% on
  this reference, because 41% of held-out species are absent from it.
  However, on **unseen genera** the same thresholds give 15–19% error at
  order/family. Thresholds learned on known taxa do not transfer to novel
  lineages.
* **Calibrated on random + clade-exclusion held-out sets** (as in
  `scripts/workflow.sh`): on unseen genera, family error falls to 2.0% and
  genus/species overclassification from 28.8% and 26.5% to 4.0% and 0%.
  Order-level error on unseen genera, however, is 15.0%, because the
  order threshold stays low (7.6). The cost on known taxa is far fewer
  genus/species assignments: genus correct falls from 45.5% to 13.1% of
  sequences, because the genus threshold rises to 97.5.

Calibration is recommended, but the **calibration set must represent
the novelty expected in the samples**, and the target error should be
chosen per study. `calibrate` records thresholds for a grid of target
errors (1–20%) so this choice can be made without re-fitting.

### 6.4 Novelty detection

Cut-offs on IDF-weighted similarity to the assigned taxon were fitted on
the calibration halves, maximising balanced accuracy, with at least 20
examples on each side. Genus cut-off 0.904 (known kept 64%, novel flagged
82%); species 0.915 (69% / 78%); family 0.525 (84% / 77%).

| | random: genus acc / error | random: species acc / error | unseen genera: genus error | unseen genera: species error |
|---|---|---|---|---|
| standard @ 60 | 45.5 / 7.2% | 28.0 / 12.2% | 28.8% | 26.5% |
| + `--truncate-novel` | 35.6 / 4.4% * | 22.9 / 7.4% * | **13.8%** * | **12.8%** * |

At matched error on the random set, truncation helps at species (5% error:
30.6 → 32.9%) and costs at genus (51.6 → 45.9%). `--novelty-flag`
(reporting `max_similarity` and `novelty_rank` without truncating) is
recommended. Use `--truncate-novel` when overclassification of novel
lineages is the main concern, or to select ASVs for placement.

### 6.5 Consensus with phylogenetic placement

`consensus` and `scripts/run_placement.sh` are implemented and tested on
synthetic gappa per-query output and jplace pendant lengths
(`consensus_with_synthetic_placement`). An end-to-end benchmark was
**not run**: EPA-ng is not packaged for this environment's distribution,
and building it from source was not permitted here. To evaluate it, run
`scripts/run_placement.sh` on the novelty-flagged subset of a
clade-exclusion held-out set, then `validate`-style scoring of
`consensus.tsv`.

### 6.6 Adaptive bootstraps

`--adaptive-bootstraps 1000 --adaptive-window 10` on the random set:

* 360 of 1,214 unique queries (30%) were re-run, and 63 assignments (4.6%)
  changed; run time rose 2.7×.
* Accuracy and error at threshold 60 and at matched error were unchanged
  within noise: family 89.8% vs 89.7% at 5% error, genus 51.6% vs 52.9%.

DECIPHER's default replicate count is already enough for these 450 bp
amplicons. Not recommended.

### 6.7 Group-size balancing

* **Scoring normalisation** (`classify --mode balanced`, hits × n_g^−γ)
  reduced accuracy significantly at every rank on known taxa (domain −5.5
  pt*, genus −10.6 pt* at γ = 0.1; worse at γ = 0.3). It slightly reduced
  overclassification for unseen genera. **Negative result: not
  recommended.**
* **Training cap** (`train --mode balanced --max-group-size 5`, farthest-
  point subsampling) removed redundant references (10,843 → 7,749 training sequences). It
  lowered genus/species accuracy on known taxa (genus at 5% error 51.6% →
  40.6%) and improved family-level assignment of unseen genera (5% error:
  43.7% → 62.8%), because over-sampled groups no longer dominate the
  replicates. Recommended only where references are heavily dominated by a
  few over-sampled taxa and the samples are expected to be novel-rich.

### 6.8 Spaced seeds

`train --mode spaced` (pattern `1101101101101`, weight 9, span 13):

* Random leave-out: small, mostly non-significant gains (family 89.8% →
  92.0%, species 30.6% → 32.1% at 5% error).
* Clade exclusion: small losses (order at 5% error 64.9% → 60.7%).

Contrary to the hypothesis, spaced seeds did not help divergent queries
from unseen genera on this marker. Neutral; not recommended.

### Phase 6 conclusion: recommended configuration

The extensions that **measurably help** are label cleaning (6.2) and
calibration with a novelty-aware calibration set (6.3), with the novelty
flag (6.4) for reporting and for choosing a placement subset.

Region extraction (6.1) is neutral on this marker but is kept in the
workflow because training on the amplicon region removes primer and
length artefacts at no measured cost. Its coverage report shows what is
lost.

Adaptive bootstraps, balanced scoring and spaced seeds are not
recommended. Balanced training is a niche option. Standard-mode output is
unchanged by all extensions (Phase 2 parity tests run in CI).
