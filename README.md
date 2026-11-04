# idtaxa-rs

A scalable Rust reimplementation of the **IDTAXA** taxonomic classifier
(Murali, Bhargava & Wright 2018, *Microbiome* 6:140), as implemented by
`LearnTaxa()` / `IdTaxa()` in the R/Bioconductor package
[DECIPHER](https://bioconductor.org/packages/DECIPHER). It is built for
metabarcoding: millions of ~450 bp ASVs against references of 10⁴–10⁵
sequences, on one node or a cluster.

* **Standard mode is DECIPHER-equivalent.** On a 10,000-query PR2 18S V4
  benchmark, idtaxa-rs agrees with DECIPHER 3.9.4 as closely as DECIPHER agrees
  with itself across random seeds (97.99% vs 97.94% of rank assignments, mean
  |Δconfidence| 1.56 vs 1.55 points). Natively trained models are
  equivalent to `LearnTaxa` models: they give the same classification output
  on all 10,000 queries. See [`docs/benchmarks.md`](docs/benchmarks.md).
* **Fast:** 21× faster than `IdTaxa(processors = NULL)` on the same 4
  cores, and `train` is about 70× faster than `LearnTaxa`. Memory stays
  flat as the number of queries grows. Results are identical regardless of
  thread count, batch size or sharding.
* **Opt-in accuracy extensions:** amplicon-region extraction, reference
  label cleaning, per-rank calibration, novelty detection, consensus with
  phylogenetic placement, adaptive bootstraps, group-size balancing and
  spaced seeds. Each is evaluated against standard mode in
  `docs/benchmarks.md`, including the ones that do not help.

## Licence

**Option A (GPL-3 port) was followed.** DECIPHER's R and C source
(`R/LearnTaxa.R`, `R/IdTaxa.R`, `src/EnumerateSequence.c`,
`src/VectorSums.c`, `src/Utils.c`, version 3.9.4) were read to match
formulas, defaults and edge cases exactly; every one is documented with its
source in [`docs/algorithm-notes.md`](docs/algorithm-notes.md).
Consequently idtaxa-rs is licensed under the **GNU GPL v3 or later**
(`LICENSE`).

## Install

Requires stable Rust ≥ 1.87.

```bash
git clone https://github.com/metagenAu/idtaxa-rs && cd idtaxa-rs
cargo build --release          # binary: target/release/idtaxa
cargo install --path crates/idtaxa-cli   # or install into ~/.cargo/bin
idtaxa --version               # tool version + native model format version
```

The R scripts need R with DECIPHER (≥ 2.x) and jsonlite. The comparison
script needs Python 3.

## Quick start

```bash
# train from a FASTA + taxonomy TSV (id<TAB>Root;Domain;...;Species)
idtaxa train --ref ref.fasta --taxonomy ref_taxonomy.tsv --output model.idtx
#   (lineages may instead be in the FASTA headers, e.g. DADA2-formatted PR2/SILVA)

idtaxa info --model model.idtx

# classify (FASTA/FASTQ, plain or gzip; '-' for stdin)
idtaxa classify --model model.idtx --input asvs.fasta.gz --output out.tsv \
  --threshold 60 --bootstraps 100 --strand both --threads 32
```

Output columns: `id`, `seq_hash` (xxh3-64 of the normalised sequence),
`strand` (`+`/`-`, `NA` = too few k-mers to classify), `taxonomy` (ranks with
confidence ≥ threshold, `;`-separated, Root omitted), `confidences`
(per rank), `bootstraps_used`; with `--full-confidences` also
`full_taxonomy`/`full_confidences` (whole lineage, any confidence).

`--format decipher` writes DECIPHER's `IdTaxa(type = "collapsed")`
strings (`Root [rootrank, 100.0%]; Eukaryota [...]; unclassified_X [...]`),
and `scripts/read_idtaxa_rs.R` turns native output into a DECIPHER
`Taxa`/`Test` object, so downstream R code works unchanged:

```r
source("scripts/read_idtaxa_rs.R")
ids <- read_idtaxa_rs("out.tsv")   # behaves like IdTaxa(..., type = "extended")
```

## Using an existing DECIPHER training set

```bash
Rscript scripts/export_decipher_model.R trainingSet.RData model_export/   # or .rds
idtaxa import-decipher --input model_export/ --output model.idtx
idtaxa kmer-check --model model.idtx --ref the_training_sequences.fasta  # optional
```

The neutral export format is specified in
`crates/idtaxa-core/src/import.rs`. Imported models carry DECIPHER's decision
k-mers, fractions, IDF weights and problem sequences unchanged; k-mer codes
are translated from DECIPHER's 1-based to 0-based.

## Scaling

* **Streaming and batching:** input is read in batches (`--batch-size`,
  default 50,000), so memory depends on the model and batch size, not on
  the number of queries.
* **Memory-mapped models:** the native model (`.idtx`) is a versioned
  header plus flat arrays. Loading is near-instant, and processes on one
  node share the model through the page cache.
* **Dereplication:** identical sequences in a batch are classified once.
* **Persistent cache** (`--cache cache.redb`): key = (model ID, parameters,
  sequence). Re-runs and overlapping datasets skip ASVs already classified.
* **Determinism:** every query has its own RNG stream, seeded from
  `hash(sequence) XOR --seed`. Output is byte-identical across thread
  counts, batch sizes and chunking.
* **Cluster sharding:** `--chunk i/n` processes every n-th record
  (1-based i); `idtaxa merge` restores input order.

SLURM array example (`scripts/slurm_classify.sh`):

```bash
sbatch --array=1-50 scripts/slurm_classify.sh model.idtx asvs.fasta.gz results/
# after all tasks finish:
idtaxa merge --inputs results/part_*.tsv --output results/all.tsv
```

## Validation and calibration

```bash
# held-out sets: random leave-out, or clade exclusion (whole genera/species)
idtaxa split --ref ref.fasta --taxonomy tax.tsv --mode random --fraction 0.1 --output-prefix rnd
idtaxa split --ref ref.fasta --taxonomy tax.tsv --mode clade --rank 8 --fraction 0.1 --output-prefix gen

idtaxa train --ref rnd_train.fasta --taxonomy rnd_train.tsv --output m.idtx
idtaxa validate --model m.idtx --holdout rnd_holdout.fasta --taxonomy rnd_holdout.tsv \
  --output-prefix val                      # val.tsv (all thresholds) + val_summary.txt
idtaxa validate ... --compare-mode standard,balanced      # paired bootstrap CIs
idtaxa validate ... --compare-model m_spaced.idtx         # compare two models
idtaxa calibrate --model m.idtx --holdout cal.fasta --taxonomy cal.tsv \
  --novelty-holdout gen_holdout.fasta --novelty-taxonomy gen_holdout.tsv \
  --target-error 0.05 --output calibration.json --reliability reliability.tsv
```

`validate` reports per rank, across thresholds: accuracy, misclassification,
overclassification and underclassification rates and error rate. Taxa
absent from the reference count as overclassification when assigned
(Murali et al. 2018).

## Accuracy extensions (all opt-in)

| command / flag | what it does |
|---|---|
| `extract-region --fwd P --rev P` | in-silico PCR (IUPAC, mismatches, 3′-exact bases, both orientations, length filter, `--rescue` of references lacking a primer site), with a per-rank report of retained taxa |
| `clean-ref` | truncates lineages at placeholder labels (`uncultured`, `metagenome`, `Genus sp.`, …), normalises taxonomy, reports homonyms and rank shifts, reviews or drops problem sequences, and can retrain iteratively; every change is logged |
| `calibrate` / `classify --calibration` | per-rank isotonic calibration and thresholds for a target error rate, tied to the model checksum |
| `classify --novelty-flag [--truncate-novel]` | IDF-weighted similarity to the reference and per-rank novelty cut-offs learned from clade exclusion |
| `classify --adaptive-bootstraps N` | re-classifies queries near a threshold with N replicates |
| `train --mode balanced` / `classify --mode balanced` | caps sequences per group (diversity-preserving) or normalises scores by group size |
| `train --mode spaced [--seed-pattern]` | spaced-seed k-mers, stored in the model |
| `consensus` | combines IDTAXA with EPA-ng/gappa placement (`scripts/run_placement.sh`), flags disagreements and long pendant branches |

### Recommended configuration

Based on the measurements in
[`docs/benchmarks.md`](docs/benchmarks.md#phase-6-accuracy-extensions)
(PR2 18S V4, random leave-out and genus-exclusion held-out sets):

| step | recommendation | evidence |
|---|---|---|
| `extract-region` | optional; train on the amplicon region and check the coverage report | neutral on PR2 V4 at matched error; `--rescue` raises species coverage from 58% to 73% |
| `clean-ref --drop-placeholder-labels` | **yes** | species error 11.4% → 7.8% at threshold 60; species accuracy at 5% error 34.0% → 41.6% |
| problem-sequence removal | no (review the file instead) | raises upper-rank error, no species gain over cleaning |
| `train` | standard mode | spaced seeds neutral; balanced training is a trade-off |
| `calibrate` | **yes**, on a held-out set that mixes random and clade-exclusion sequences | raw confidence is strongly under-confident; thresholds learned on known taxa only give 15–19% order/family error on unseen genera |
| `classify --calibration --novelty-flag` | **yes**; `--truncate-novel` when overclassification matters most | truncation halves genus/species overclassification of unseen genera (28.8% → 13.8%) at a cost of 10 points of genus accuracy on known taxa |
| `--adaptive-bootstraps` | no | 30% of queries re-run, no gain |
| `--mode balanced` (scoring) | no | genus accuracy at 5% error 51.6% → 15.4% |
| placement + `consensus` | on the novelty-flagged subset | implemented; not benchmarked end-to-end here (EPA-ng unavailable) |

End-to-end: `scripts/workflow.sh reference.fasta taxonomy.tsv asvs.fasta out/ [refpkg]`
runs extract-region → clean-ref → split → train → validate → calibrate →
classify (calibrated, novelty flag; `ADAPTIVE=N` to add adaptive
bootstraps) → placement on the flagged subset (`scripts/run_placement.sh`)
→ consensus.

## Known differences from DECIPHER

* **Random numbers:** idtaxa-rs uses per-query Xoshiro256++ streams, not R's
  RNG, so results are statistically, not bit-for-bit, identical.
  Run-to-run differences are of the same size as between two DECIPHER seeds.
* **Gaps:** `-` and `.` are removed from input sequences; DECIPHER stops
  with an error.
* **Lineages without `Root;`** are accepted (`Root;` is prepended);
  DECIPHER requires it.
* **Training:** the fraction-learning iterations process sequences in
  deterministic parallel blocks of 1,024 instead of strictly one by one (see
  `docs/algorithm-notes.md` §2.5). Fractions, problem groups and
  classification output are unchanged on the benchmark, and the
  problem-sequence sets overlap about as much as two DECIPHER runs.
* **Nucleotide models only:** amino-acid `LearnTaxa` models (reduced
  alphabets) are not supported.
* **No `rank` data.frame:** `train --rank-names` assigns rank names by
  level instead of DECIPHER's `rank` table.
* **Strand choice:** with `strand = "both"`, the reverse orientation is kept
  only if its *similarity* is higher, as in DECIPHER. This differs from
  "higher confidence" in some descriptions.

## Repository layout

```
crates/idtaxa-core/   library: kmer, taxonomy, model (mmap format + inverted index),
                      classify, train, import, eval, calibration, region, clean,
                      consensus, balance
crates/idtaxa-cli/    the `idtaxa` binary
scripts/              export_decipher_model.R, train_decipher.R, run_decipher_reference.R,
                      compare.py, prepare_parity_data.py, read_idtaxa_rs.R,
                      slurm_classify.sh, run_placement.sh, workflow.sh, matched_error.py
tests/data/           412-sequence PR2 V4 fixture, DECIPHER export and cached IdTaxa outputs
docs/                 algorithm-notes.md, benchmarks.md
```

## Citation

If you use idtaxa-rs, please cite the IDTAXA paper (Murali A, Bhargava A,
Wright ES. IDTAXA: a novel approach for accurate taxonomic classification
of microbiome sequences. *Microbiome* 2018;6:140) and DECIPHER (Wright ES.
Using DECIPHER v2.0 to analyze big biological sequence data in R. *R
Journal* 2016;8:352–359).
