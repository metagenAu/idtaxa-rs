# idtaxa-rs

A fast, primer-aware taxonomic classifier for amplicon metabarcoding (16S, ITS, 18S), written in Rust and
based on the IDTAXA algorithm (Murali, Bhargava & Wright 2018, *Microbiome* 6:140).

It was built for a common situation in soil and agricultural microbiome work. You have **your own
metabarcoding database** and want to **reuse public datasets**, but those datasets were generated with
**different primer sets**. idtaxa-rs:

* trains a **separate classifier for each primer set** by trimming references to each amplicon *in silico*.
  This is usually much more accurate than classifying short amplicons against full-length references.
* reports **primer coverage per taxon**, which shows the taxa a primer set is blind to.
* **detects which primers** were used in public reads where the metadata is missing or wrong.
* runs a **leave-one-out check of your reference database** that flags mislabelled or contaminated
  references.
* **cross-validates** a reference + primer combination: accuracy, under-classification and
  over-classification per rank.
* collapses classified ASV tables to a common rank in a **long/tidy format** that concatenates across
  studies. This feeds meta-analysis and the [management → microbiome knowledge
  graph](docs/knowledge-graph.md).

## Install

```bash
cargo install --git https://github.com/metagenAu/idtaxa-rs
```

Or build from a clone: `cargo build --release` gives `target/release/idtaxa`.

## Quick start

```bash
# 1. which primer pairs are built in?
idtaxa primers

# 2. how well does 515F/806R cover the reference, per phylum?
idtaxa trim -i silva_nr99.fa.gz --primers 515F-806R -o silva_V4.fa \
    --coverage V4_coverage.tsv --coverage-rank phylum

# 3. train a V4-specific classifier (in-silico PCR is applied automatically)
idtaxa train -r silva_nr99.fa.gz --primers 515F-806R -o silva_V4.idtx

# 4. classify ASVs (primers already removed, e.g. DADA2 output)
idtaxa classify -m silva_V4.idtx -i asvs.fa -o taxonomy.tsv

# 5. collapse an ASV table to genus, in long format tagged with study + primer set
idtaxa summarize -c taxonomy.tsv --counts asv_table.tsv --rank genus \
    --long --study MYTRIAL2024 --primer-set 515F-806R -o genus_long.tsv
```

`scripts/soil_workflow.sh` is a complete example for a lab that combines its own data with public
16S V4, V3–V4 and ITS2 datasets.

## Commands

| command | purpose |
|---|---|
| `train` | Build a classifier from reference FASTA. Optional in-silico PCR (`--primers` or `--forward/--reverse`), leave-one-out curation (`--problems`, `--remove-problems`). |
| `classify` | Assign taxonomy with bootstrap confidences. Output is TSV with one column per rank, or QIIME 2 format. |
| `trim` | In-silico PCR only: write amplicons and a per-taxon coverage table. |
| `primers` | List the built-in primer catalog. |
| `detect-primers` | Report which catalog primers occur at read starts (FASTQ/FASTA, gzip ok). |
| `summarize` | Collapse a feature table to a rank: wide (taxon × sample) or long (`study, primer_set, sample, level, taxon, lineage, count, rel_abundance`). |
| `evaluate` | Hold-out cross-validation per rank. |
| `info` | Model metadata (reference, primers, k, ranks, notes) as JSON. |

Run `idtaxa <command> --help` for every option.

### Reference formats

Lineages are read from FASTA headers or from a QIIME-style taxonomy TSV (`--taxonomy`):

* SILVA / DECIPHER: `>AB000001.1.1500 Bacteria;Firmicutes;Bacilli;...`
* DADA2 training sets (header is the lineage): `>Bacteria;Firmicutes;Bacilli;...;`
* GTDB / QIIME / UNITE prefixes: `d__Bacteria; p__Firmicutes; ...`, UNITE `>name|acc|SH|reps|k__Fungi;p__...`

Ranks come from `d__/k__/p__/c__/o__/f__/g__/s__` prefixes, or from `--ranks domain,phylum,...` for
unprefixed lineages. Lineages are cut off at empty or uninformative names (`g__`, `uncultured`,
`unidentified`, `metagenome`, …), so these never become taxa.

### Built-in primers

| name | marker / region | notes |
|---|---|---|
| `515F-806R` | 16S V4 | EMP (Parada/Apprill modified) |
| `515F-806R-orig` | 16S V4 | original EMP; under-detects Thaumarchaeota (soil ammonia oxidisers) |
| `515F-926R` | 16S V4–V5 | also amplifies 18S |
| `341F-785R`, `341F-806R` | 16S V3–V4 | Illumina protocol / common commercial set |
| `27F-519R` | 16S V1–V3 | Australian Microbiome |
| `799F-1193R` | 16S V5–V7 | avoids chloroplast (rhizosphere, endophytes) |
| `27F-1492R` | 16S full length | PacBio / Nanopore |
| `ITS1F-ITS2`, `ITS3-ITS4`, `fITS7-ITS4`, `gITS7-ITS4`, `ITS86F-ITS4`, `ITS1F-ITS4` | fungal ITS1 / ITS2 / full ITS | |
| `AMV4.5NF-AMDGR`, `WANDA-AML2` | 18S, arbuscular mycorrhizal fungi | better than ITS for AMF |
| `TAReuk454FWD1-TAReukREV3` | 18S V4 | protists / eukaryotes |

For other primers use `--forward/--reverse` (5'→3', IUPAC codes allowed).

> **UNITE and other pre-trimmed references.** UNITE sequences usually lack the LSU flank where ITS4
> binds. Use `--partial` so that references where only one primer site is found are kept and run
> to the end of the sequence.

## Using public data with different primer sets

1. **Build one classifier per primer set, from the same reference release.** For example, SILVA 138.2
   trimmed to V4, V3–V4 and V1–V3, plus UNITE for ITS2. Add your in-house curated sequences to each
   reference before training. Keep the `.idtx` files with your data, since `idtaxa info` records the
   reference and primers used.
2. **Find out what each public dataset actually used.** Run `idtaxa detect-primers R1.fastq.gz R2.fastq.gz`
   on a sample of reads. SRA metadata is often wrong or missing. If no primer is detected, the reads
   were already trimmed.
3. **Classify each dataset with its primer-matched classifier**, using the same threshold everywhere.
4. **Compare studies at taxon level, never at ASV level.** `idtaxa summarize --long --study ... --primer-set ...`
   produces tables you can simply concatenate. Treat primer set as a batch effect in your models (a
   random effect, or a covariate in ANCOM-BC2 / MaAsLin2), or estimate effects within each study and
   then meta-analyse them.
5. **Know the blind spots.** `trim --coverage` gives the percentage of each taxon amplified *in silico*.
   A taxon absent from one study may simply not be amplified by its primers. Record these coverage
   values with your data, as the knowledge graph does (`UNDER_DETECTS` edges).
6. **Benchmark before you trust.** `idtaxa evaluate --primers 341F-806R ...` shows how accurate genus-level
   calls are for that region, which tells you which rank is safe for cross-study comparison.

## How it works

Training (`train`):

1. Lineages → taxonomic tree. Identical sequences with identical lineages are collapsed, and
   `--max-per-group N` can cap the number of sequences per taxon.
2. Sorted unique k-mers per sequence. By default k = ⌊log₄(100 × 99th-percentile length)⌋, which gives
   k = 8 for full-length 16S and k = 7 for V4 or ITS2. k-mer weights are inverse frequency, with each
   group weighted equally.
3. Sequences are stored in depth-first taxonomic order, so every taxon is one contiguous range, and an
   inverted k-mer index is built when the model loads.
4. Decision k-mers and naive-Bayes profiles are built per node, then sampling fractions are learned.
   These are used only by `--tree-descent` (below).

Classification (`classify`):

1. The query's weighted k-mer overlap with every reference is computed through the inverted index, and
   the `--max-candidates` (500) most similar references are kept.
2. **IDTAXA bootstrap.** 100 replicates each sample ⌈L^0.47⌉ of the query's L k-mers. The most similar
   group wins each replicate and receives a vote equal to its similarity, so distant matches earn low
   confidence. This is what keeps IDTAXA from over-classifying novel organisms.
3. **Upper ranks.** While one taxon wins ≥98% of replicates (each replicate sampling 20% of the k-mers),
   that rank is reported with this nearest-neighbour support. This plays the role of IDTAXA's
   tree-descent confidence. Below that point, confidence is the similarity-weighted vote.
4. Ranks are reported down to the last one at or above `--threshold` (default 60%). The `best_path`
   column always gives the full best guess with every confidence, so you can apply other thresholds
   afterwards.

Differences from DECIPHER's `IdTaxa`:

* **Default mode.** Upper ranks are fixed by nearest-neighbour bootstrap support, not by
  decision-k-mer tree descent. Group-level k-mer profiles describe large, diverse taxa (such as
  "Eukaryota") poorly next to small homogeneous ones. That made descent stall or be misdirected on
  real references. `--tree-descent` restores descent-first behaviour.
* **Curation.** Problem sequences are found by a **leave-one-out** classification of every reference
  (`train --problems`) instead of fraction-learning failures.

### Benchmark: PR2 v5.0.0, 18S V4 (TAReuk454FWD1/TAReukREV3)

221,085 PR2 references were trimmed *in silico* and 132,058 V4 amplicons were obtained. 5% were held
out and the classifier was trained on the rest, using 4 CPU cores.

| rank | correct | misclassified | over-classified (novel taxa) |
|---|---|---|---|
| domain | 99.7% | 0.1% | – |
| division | 99.0% | 0.2% | – |
| class | 97.3% | 0.5% | – |
| order | 94.0% | 1.0% | – |
| family | 89.1% | 1.6% | – |
| genus | 53.9% | 3.3% | 14.0% of 541 |
| species | 34.9% | 2.0% | 14.6% of 1,400 |

The remainder at each rank is under-classified (left unassigned rather than called wrongly). Low
genus and species rates are expected for 18S V4, which often cannot separate congeners. Original
tree-descent mode on the same split: 95.5% correct at domain, 85.4% at family, 51.8% at genus.

Throughput: trimming 221k references takes 10 s. Training on 82k unique amplicons takes 25 s.
Classification runs at about 4 ms per query per strand per core. The leave-one-out check of all 82k
references takes about 9 minutes. That check flagged, for example, 59 PR2 "Bacteria" records whose V4
sequence is k-mer-identical to plant or eukaryotic 18S.

## Development

```bash
cargo test --release      # unit tests + synthetic end-to-end tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Citation

If you use this software, please cite the original IDTAXA paper:

> Murali A, Bhargava A, Wright ES (2018). IDTAXA: a novel approach for accurate taxonomic
> classification of microbiome sequences. *Microbiome* 6:140. doi:10.1186/s40168-018-0521-5

## License

MIT
