# Algorithm notes

This document records every formula, default and edge case implemented in
`idtaxa-rs`, with its source. `idtaxa-rs` follows **Option A (GPL-3 port)**:
DECIPHER's R and C source were read to match formulas exactly.

Reference version: **DECIPHER 3.9.4** (Bioconductor devel, `DESCRIPTION`
dated 2026-08-30), files `R/LearnTaxa.R`, `R/IdTaxa.R`,
`src/EnumerateSequence.c` (`enumerateSequence`, `alphabetSize`),
`src/VectorSums.c` (`vectorSum`, `parallelMatch`), `src/Utils.c`
(`intMatch`, `groupMax`). Paper: Murali, Bhargava & Wright (2018)
*IDTAXA: a novel approach for accurate taxonomic classification of
microbiome sequences*, Microbiome 6:140.

Where the implementation brief and DECIPHER disagree, DECIPHER wins; such
cases are marked **Discrepancy with brief**.

---

## 1. K-mer encoding

Source: `src/EnumerateSequence.c`, `enumerateSequence()` and
`alphabetFrequency()`; called from `LearnTaxa`/`IdTaxa` with
`fastMovingSide = 1L`.

* Base codes: A = 0, C = 1, G = 2, T = 3 (Biostrings' 1/2/4/8 bit codes).
  Every other letter (IUPAC ambiguity codes, N) gives `-1`, and any window
  containing it is `NA`.
* With `fastMovingSide = 1` the position weight vector is `pwv[j] = 4^j`,
  so the window starting at `p` has code `Σ_{j=0}^{K-1} base[p+j]·4^j` — the
  **first base is the least significant digit**.
* `LearnTaxa`/`IdTaxa` then apply `sort(unique(x + 1L), na.last = NA)`:
  sorted, unique, 1-based codes, NAs removed. `idtaxa-rs` stores 0-based
  codes; `import-decipher` subtracts 1.
* Lower case and U: Biostrings `DNAStringSet`/`RNAStringSet` store U as T
  and are case-insensitive at construction; `idtaxa-rs` maps `acgtu` the same
  way (`kmer::BASE_CODE`).
* Gaps: DECIPHER stops with an error for `-` or `.` in `train`/`test`.
  `idtaxa-rs` **removes** gap characters when reading FASTA (documented
  deviation, needed for aligned references).
* Reverse complement: `IdTaxa` enumerates `reverseComplement(test)` for
  `strand = "both"`. `idtaxa-rs` computes the reverse-complement k-mer codes
  in the same rolling pass: `rc = ((rc << 2) | (3 − b)) & mask`. Exhaustive
  tests compare with encoding the reverse-complemented string
  (`kmer::tests::matches_naive_random`).
* Sequences shorter than K yield no k-mers.
* K range: DECIPHER accepts `K ≤ 15` for nucleotides; `idtaxa-rs` supports
  1..=16 in the encoder (`u32` codes), and `train --k` accepts 1..=15 like
  `LearnTaxa`.

Verified: for the 8,000-sequence PR2 V4 parity reference, the k-mer sets of
all training sequences imported from DECIPHER equal those recomputed by
`idtaxa kmer-check` (0 mismatches).

## 2. Training (`LearnTaxa`)

### 2.1 Defaults (R/LearnTaxa.R signature)

| parameter | DECIPHER default | note |
|---|---|---|
| `K` | `NULL` (automatic) | |
| `N` | 500 | expected matches by chance, for automatic K |
| `minFraction` | 0.01 | |
| `maxFraction` | **0.06** | **Discrepancy with brief** (brief says 0.4; 0.4 was an older default) |
| `maxIterations` | 10 | |
| `multiplier` | 100 | |
| `maxChildren` | 200 | nodes with more children get no decision k-mers |
| bootstrap replicates `B` | 100 (fixed) | |

### 2.2 Automatic K

`K = floor(log(N · quantile(width(train), 0.99), alphabetSize(train)))`,
where `alphabetSize` (`src/EnumerateSequence.c`) is `exp(H)` with `H` the
Shannon entropy of the A/C/G/T frequencies over all training sequences
(the effective alphabet size). `quantile` is R's type 7. K is clamped to
`[1, 13]` (`maxK = 13`, with warning "K may be smaller than ideal.").
**Discrepancy with brief**: the brief approximates this as
`floor(log4(100 × high-quantile length))`; DECIPHER uses N = 500, the 0.99
quantile and the entropy-based base. Implemented in `train::auto_k`.

### 2.3 Taxonomy

* Cleaning (`classes`): `gsub(" *; *", ";", taxonomy)`, then the part after
  the first `"Root;"` (`strsplit(x, "Root;")[[1]][2]`), trailing spaces
  removed, and `";"` appended if missing. DECIPHER requires `"Root;"` in
  every lineage; `idtaxa-rs` prepends it when absent (convenience).
* Nodes (`taxonomy`): every prefix of every distinct class, in order of
  first appearance; node 1 is `"Root;"`. `taxa` is the last name,
  `levels` the number of names (Root = 1), `children[[i]]` the later nodes
  one level deeper that start with node `i`'s path, `parents` the inverse.
  `idtaxa-rs` builds the identical order (0-based) — the `compare-models`
  check reports `same_taxonomy: true` against a DECIPHER model.
* `crossIndex` = node of each sequence's full class; `sequences[[k]]` =
  sequences whose class starts with node `k`'s path (its whole subtree,
  including sequences labelled at internal nodes).

### 2.4 Decision k-mers (`.createTree`, recursive)

For a node `k` with `l` children, `0 < l ≤ maxChildren`:

1. Recurse into each child, obtaining a k-mer frequency profile `p_j` and
   a descendant count `d_j`. A child with no children (or more than
   `maxChildren`) returns the normalised tabulation of all k-mers of all
   sequences in its subtree (`tabulate(unlist(kmers[w]))/sum`) with `d = 1`.
2. `q = Σ_j p_j·d_j / Σ_j d_j` (weighted mean profile); the node returns
   `(q, Σ d_j)` to its parent.
3. `H[x, j] = −p_j[x]·log(q[x])` (cross entropy). Where `q = 0` this is
   `−0·log 0 = NaN`.
4. `o = apply(H, 2, order, decreasing = TRUE)`: per child, k-mers by
   decreasing `H`. R's `order` is stable (ties keep ascending k-mer order,
   and `0` ties `−0`) and puts NaN last (verified in R 4.3:
   `order(c(1,2,2,NaN,0,-0,2), decreasing=TRUE)` = `2 3 7 1 5 6 4`).
5. `recordKmers = ceiling(max(colSums(profile > 0))) * 0.10`; k-mers are
   taken round-robin — rank 1 of child 1, rank 1 of child 2, …, rank 2 of
   child 1, … — skipping k-mers already chosen, until `count ≥ recordKmers`
   (so `ceil(0.1 · max non-zero count)` k-mers, evaluated in floating
   point as in R).
6. `decisionKmers[[k]] = list(sorted chosen k-mers, t(profile[keep, ]))`:
   the weights matrix is children × k-mers.

Nodes with no children or with more than `maxChildren` children get
`NULL`; children of a node with more than `maxChildren` children are not
visited (they never get decision k-mers).

`idtaxa-rs` (`train::TreeBuilder`) uses sparse profiles and a lazy per-child
iterator reproducing the R ordering (positive `H` descending with ascending
k-mer tie-break, then zero-`H` k-mers of `q`'s support ascending, then NaN
k-mers ascending). Against DECIPHER on the parity reference: 4,230 of 4,234
nodes have identical decision k-mer sets, mean Jaccard 0.99997 (the rest
differ through floating-point ties: R's `colSums` accumulates in long
double), max weight difference 3e-18.

Edge case: when exactly one k-mer is kept at a node with several children,
R's `t(profile[keep, ])` drops to a 1 × l matrix (mis-oriented).
`export_decipher_model.R` re-orients it; native training always stores
children × k-mers.

### 2.5 Fraction learning

* `fraction` starts at `maxFraction` for every node; `delta =
  (maxFraction − minFraction) · multiplier`.
* Each iteration, every sequence still `incorrect == TRUE` is classified by
  tree descent from Root:
  * stop (counted correct) if the node has no decision k-mers;
  * with more than one child: `s = ceiling(n · fraction[k])` (or
    `n · minFraction` if `fraction[k]` is NA), draw `B = 100` replicates of
    `s` decision-k-mer indices with replacement; for child `j`,
    `vectorSum` = Σ weights of sampled k-mers present in the sequence /
    Σ weights of all sampled k-mers (0 if the denominator is 0); a child
    scores a hit in a replicate if it attains the replicate's maximum and
    that maximum is > 0; `w = which.max(hits)` (first maximum); stop
    (correct) if `hits[w] < 0.8·B`;
  * with one child, `w = 1`;
  * if the chosen child is not an ancestor of the sequence's class:
    incorrect; else descend (stop when the child is a leaf).
* On an incorrect decision at node `k`: `predicted = taxonomy[child]`; if
  `fraction[k]` is NA the sequence becomes NA (given up); otherwise
  `fraction[k] −= delta / nSeqs[k]` (`nSeqs` = sequences in `k`'s subtree)
  and if `fraction[k] ≤ minFraction` the sequence becomes NA and
  `fraction[k] = NA`.
* Stops early when no sequence remains `TRUE`.
* `problemSequences` = sequences still `TRUE` or NA, with `Expected =
  "Root;" + class` and the last `Predicted`; `problemGroups` = taxonomy of
  nodes with NA fraction.

**Deviation (parallelism)**: DECIPHER processes sequences strictly one after
another, so a fraction reduction affects later sequences of the same
iteration. `idtaxa-rs` classifies blocks of 1,024 sequences in parallel
with the fractions fixed at the start of the block, then applies the
updates in sequence order. The block size is fixed, so results do not
depend on the thread count. RNG streams are per (sequence, iteration,
seed). Effect on the parity reference: identical problem groups, problem
sequence counts 260 (native) vs 269 (DECIPHER); see `docs/benchmarks.md`
for the overlap and DECIPHER's own seed-to-seed overlap.

### 2.6 IDF weights

`weight_i = 1 / |class(i)|` (each class weighted equally);
`counts[x] = Σ_{i: x ∈ kmers_i} weight_i`;
`IDF[x] = log(nClasses / (1 + counts[x]))`, `nClasses` = number of distinct
classes. K-mers absent from training have `IDF = log(nClasses)`.
`idtaxa-rs` stores IDF only for k-mers present in the index plus this
default; `import-decipher` verifies that every absent k-mer carries exactly
`log(nClasses)` and fails otherwise. Max IDF difference native vs DECIPHER:
0 on the parity reference.

## 3. Classification (`IdTaxa`)

### 3.1 Defaults

| parameter | default | note |
|---|---|---|
| `type` | `"extended"` | `--format decipher` reproduces `"collapsed"` strings |
| `strand` | `"both"` | |
| `threshold` | 60 | |
| `bootstraps` | 100 | an upper bound, see `B` below |
| `samples` | `L^0.47` | `--sample-exponent` |
| `minDescend` | 0.98 | **Discrepancy with brief**: not mentioned there |
| `fullLength` | 0 (off) | `--full-length F` gives `(1/F, F)` |

### 3.2 Per-query sample size and replicate count

* `L` = number of **non-NA k-mer positions** of the query (`notNAs`,
  counting duplicates), `S = ceiling(L^0.47)`.
* Minimum sample size (`minS`): with `L90 = quantile(lengths(kmers), 0.9)`
  over training sequences and `nKmers = 4^K`, `minS` is the first even
  value in 2..100 for which
  `1 − pbinom(0, nTrain, 1 − pbinom(minS/2 − 1, minS, L90/nKmers)) < 0.01`
  (probability that half the sampled k-mers match some training sequence by
  chance); `S = max(S, minS)`. Verified: `minS = 6` in both R and Rust for
  the parity reference, 10 for DECIPHER's `TrainingSet_16S`.
* `B = as.integer(min(5 · |unique k-mers| / S, bootstraps))` ("set B such
  that > 99% of k-mers are sampled"). **Discrepancy with brief**: the number
  of replicates is per query and usually below 100 for short amplicons
  (≈ 122 cap for 450 bp, so 100; fewer for shorter queries). `idtaxa-rs`
  reports it as `bootstraps_used`.
* If `|unique k-mers| ≤ S` the query is not classified:
  `Root [0%]; unclassified_Root [0%]` (strand `NA` in native output).

### 3.3 Tree descent

From Root (node `k`), repeat:

* `n = |decisionKmers[[k]]|`; if `n == 0` or `fraction[k]` is NA, use
  **all** children of `k` and stop.
* With more than one child: `s = ceiling(n · fraction[k])`; draw `B`
  replicates of `s` decision-k-mer indices; per replicate and child compute
  `vectorSum` (as in training); count per child the replicates where it
  attains a positive maximum (ties count for every tied child).
  * If exactly one child has `hits ≥ minDescend · B`, descend into it.
  * Otherwise, if no child has `hits ≥ 0.5 · B`, use **all** children and
    stop; else use every child with `hits > 0` (all if none) and stop.
* With one child: descend.
* If the chosen child has no children, use only it and stop.

**Discrepancy with brief**: the brief's descent threshold and rule are
unspecified; DECIPHER uses `minDescend = 0.98` to continue and the 50% rule
above to choose the groups for the leaf phase.

The training sequences scored are `unlist(sequences[children[[k]][w]])`, the
subtrees of the selected children (sequences labelled at `k` itself are
excluded). In `idtaxa-rs` sequences are ordered by a depth-first traversal
(each node's own sequences first, ascending original index, then its
children's subtrees), so each subtree is a contiguous ID range and the
selection is a union of ranges.

### 3.4 Leaf phase

* `sampling`: `B × S` query k-mers drawn with replacement from the unique
  query k-mers (the same sample for all training sequences).
* `hits[b, t]` = Σ IDF of the k-mers sampled in replicate `b` (with
  multiplicity) that occur in training sequence `t` (`parallelMatch`);
  `sumHits[t] = Σ_b hits[b, t]`.
* Per group (`crossIndex`), the top sequence is the first maximum of
  `sumHits` in ascending original index order (`groupMax` on
  `order(lookup)`, stable).
* `davg = mean_b(Σ IDF of the k-mers sampled in replicate b)`.
* For each replicate, the group whose top sequence has the most hits
  (`max.col`, **ties broken at random**) receives `hits[b, top]/davg`:
  `totHits[g] += hits[b, top_g] / davg`.
* `selected` = the group with maximum `totHits` (random among ties).
* Similarity `sims = Σ_b hits[b, top_selected] / davg` (reported by
  `idtaxa-rs` divided by `B` as `max_similarity` basis).

`idtaxa-rs` computes the same quantities with the inverted index: total
sample counts per query k-mer give `sumHits` in one pass over the postings
restricted to the selected ranges; a second pass adds the IDF of each
sampled k-mer to the replicates in which it was drawn, only for each
group's top sequence. Groups with no hit at all cannot win a replicate and
are not materialised; if no training sequence shares any sampled k-mer,
DECIPHER's random choice among all candidate groups is reproduced (all
confidences 0).

### 3.5 Confidence

* `confidence[selected lineage] = totHits[selected]/B · 100` for every node
  on the path Root → selected group.
* For every other group with `totHits > 0`, its `totHits/B · 100` is added
  to each of **its ancestors** (starting at its parent) that lies on the
  selected path. Root therefore gets `Σ totHits / B · 100`.
* Confidences are non-increasing down the lineage, so thresholding gives a
  prefix: ranks with `confidence ≥ threshold` are reported; if not all
  pass, `unclassified_<last assigned taxon>` is appended with the last
  assigned confidence (rank label = the next lineage rank). If none pass,
  Root is reported (`w <- 1`).

### 3.6 Strand handling

**Discrepancy with brief** (brief: "keep the orientation with higher
confidence"): for `strand = "both"`, DECIPHER classifies the forward
orientation first and the reverse complement second, and keeps the reverse
result only if its **similarity** `sims` is strictly larger than the
forward one. `"bottom"` classifies only the reverse complement. Sample size
`S` and `B` are taken from the forward orientation (identical for both
since k-mer counts are equal).

### 3.7 Dereplication

`IdTaxa` classifies each distinct sequence once (`!duplicated(test)`) and
re-expands the results. `idtaxa-rs` dereplicates within each batch on the
normalised sequence (upper case, U→T) and, with `--cache`, across runs.

### 3.8 Random numbers

DECIPHER uses R's RNG (`sample`, `max.col` with `ties.method = "random"`).
`idtaxa-rs` uses Xoshiro256++ seeded with `xxh3_64(normalised sequence) XOR
--seed` per query (descent, leaf phase and tie breaks draw from this one
stream, forward orientation first). Results are therefore identical across
thread counts, batch sizes and chunking, but not bit-identical to R; the
agreement is measured in `docs/benchmarks.md`.

## 4. Output conventions

* Native TSV: `id`, `seq_hash` (xxh3-64 of the normalised sequence),
  `strand` (`+`, `-`, `NA` = not classified), `taxonomy` (assigned names
  below Root, `;`-separated, no `unclassified_` filler), `confidences` (one
  decimal), optionally `full_taxonomy`/`full_confidences`,
  `calibrated_confidences`, `max_similarity`, `novelty_rank`, and
  `bootstraps_used`.
* `--format decipher`: `id<TAB>` + the exact `type = "collapsed"` string,
  e.g. `Root [rootrank, 100.0%]; Bacteria [domain, 99.0%]; unclassified_Bacteria [phylum, 99.0%]`
  (`formatC(..., digits = 1, format = "f")`; `[0%]` literal for
  unclassifiable queries). `scripts/read_idtaxa_rs.R` turns native TSV into
  a DECIPHER `Taxa`/`Test` object.

## 5. Extensions (not in DECIPHER; all opt-in)

* **Adaptive bootstraps** (`--adaptive-bootstraps N --adaptive-window W`):
  queries with any rank's confidence within `W` points of its threshold are
  re-classified with exactly `N` replicates (bypassing the `5·L/S` cap) using
  an independent deterministic stream (`seed XOR 0x9e3779b97f4a7c15`).
* **Balanced scoring** (`classify --mode balanced`): per-replicate hits of
  each group's top sequence are multiplied by `n_g^(−γ)` (`n_g` = training
  sequences in the group, `γ = --balance-exponent`, default 0.1) before
  choosing the replicate winner, to offset the extra chances large groups
  have of containing a high-scoring sequence.
* **Balanced training** (`train --mode balanced --max-group-size M`): groups
  with more than `M` sequences are reduced by greedy farthest-point sampling
  on k-mer Jaccard distance (starting from the member with most k-mers), so
  exact duplicates go first and divergent representatives are kept.
* **Spaced seeds** (`train --mode spaced [--seed-pattern P]`): k-mers are
  read through a 0/1 mask whose weight is K (default pattern repeats `110`,
  e.g. `11011011011011` for K = 9 … ). The pattern is stored in the model and
  used automatically by `classify`.
* **Calibration** (`calibrate`): per rank, isotonic regression (PAV) of
  "assignment correct" on raw confidence over held-out data; the per-rank
  threshold is the smallest raw confidence whose error rate among
  assignments with confidence ≥ it is ≤ `--target-error`. Stored with the
  model ID; `classify --calibration` refuses other models. The interplay of
  per-rank thresholds (truncation at the first failing rank) is not modelled
  when choosing them.
* **Novelty** (`--novelty-flag`): IDF-weighted containment
  `Σ_{x ∈ q ∩ t} IDF(x) / Σ_{x ∈ q} IDF(x)` of the query's k-mers (all, not
  sampled) in each training sequence; maxima over the whole reference and
  within each lineage node's subtree. Cut-offs per rank maximise balanced
  accuracy between "taxon present" and "taxon absent" held-out queries from
  a clade-exclusion split (`calibrate --novelty-holdout`). `novelty_rank`
  is the deepest rank whose cut-off is met at every rank above.
* **Region extraction**: IUPAC set-intersection matching, no indels,
  mismatches ≤ `--max-mismatch`, with the `--three-prime-exact` 3'-terminal
  bases exact; the reverse primer is searched as its reverse complement
  (3' end = start of the site); both template orientations. Rescue: the 5
  extracted regions (from a deterministic sample of ≤ 2,000) with highest
  8-mer Jaccard similarity are tried; their first/last 20 bases are located
  with ≤ 3 mismatches.
* **Label cleaning**: case-insensitive placeholder substrings (`uncultured`,
  `unidentified`, `unclassified`, `unassigned`, `unknown`, `environmental
  sample`, `metagenome`, `incertae sedis`, `clone`), and species labels of
  the form `Genus sp.`/`Genus_sp`/`Genus sp. XYZ` (strain-numbered); the
  lineage is truncated before the first placeholder. PR2's `_X` filler ranks
  are *not* placeholders by default (informative ranks follow them); add
  them with `--extra-placeholder _x` if wanted.
