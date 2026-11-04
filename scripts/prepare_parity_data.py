#!/usr/bin/env python3
"""Build the Phase 2 parity dataset from a DADA2-formatted reference whose
FASTA headers are lineages (e.g. PR2 v5 after `idtaxa extract-region`).

Outputs in OUTDIR:
  train.fasta / train_taxonomy.tsv     training references (ids ref_NNNNN)
  holdout.fasta / holdout_taxonomy.tsv held-out references (random leave-out)
  queries.fasta / queries_truth.tsv    10k queries: exact held-out/training,
                                       mutated variants, reverse complements
Usage: prepare_parity_data.py region.fasta OUTDIR [n_pool=10000] [seed=1]
"""
import random
import sys
from collections import defaultdict
from pathlib import Path


def read_fasta(p):
    name, seq = None, []
    with open(p) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:], []
            else:
                seq.append(line)
    if name is not None:
        yield name, "".join(seq)


COMP = str.maketrans("ACGTRYKMBVDHN", "TGCAYRMKVBHDN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def mutate(s, rate, rng, indels=True):
    out = []
    for c in s:
        r = rng.random()
        if r < rate:
            kind = rng.random() if indels else 0.0
            if kind < 0.8:
                out.append(rng.choice([b for b in "ACGT" if b != c]))
            elif kind < 0.9:
                continue  # deletion
            else:
                out.append(c)
                out.append(rng.choice("ACGT"))  # insertion
        else:
            out.append(c)
    return "".join(out)


def main():
    src, outdir = sys.argv[1], Path(sys.argv[2])
    n_pool = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rng = random.Random(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    by_species = defaultdict(list)
    for hdr, seq in read_fasta(src):
        lin = hdr.split()[0].rstrip(";")
        if len(lin.split(";")) < 2 or any(c not in "ACGT" for c in seq.upper()[:0]):
            continue
        by_species[lin].append(seq.upper())
    species = sorted(by_species)
    rng.shuffle(species)
    pool = []
    for sp in species:
        seqs = by_species[sp]
        rng.shuffle(seqs)
        for s in seqs[:3]:
            pool.append((sp, s))
        if len(pool) >= n_pool:
            break
    pool = pool[:n_pool]
    rng.shuffle(pool)
    n_hold = len(pool) // 5
    hold, train = pool[:n_hold], pool[n_hold:]

    def write(name, recs):
        with open(outdir / f"{name}.fasta", "w") as fa, open(outdir / f"{name}_taxonomy.tsv", "w") as tx:
            for i, (lin, s) in recs:
                fa.write(f">{i}\n{s}\n")
                tx.write(f"{i}\tRoot;{lin}\n")

    train_r = [(f"ref_{k:05d}", (lin, s)) for k, (lin, s) in enumerate(train)]
    hold_r = [(f"hold_{k:05d}", (lin, s)) for k, (lin, s) in enumerate(hold)]
    write("train", [(i, x) for i, x in train_r])
    write("holdout", [(i, x) for i, x in hold_r])

    queries = []
    def pick(recs, n):
        return [recs[rng.randrange(len(recs))] for _ in range(n)]
    for i, (lin, s) in pick(hold_r, 2000):
        queries.append((f"q_hold_exact_{len(queries)}", lin, s))
    for i, (lin, s) in pick(train_r, 2000):
        queries.append((f"q_train_exact_{len(queries)}", lin, s))
    for i, (lin, s) in pick(hold_r, 2000):
        queries.append((f"q_hold_mut1_{len(queries)}", lin, mutate(s, 0.01, rng)))
    for i, (lin, s) in pick(hold_r, 2000):
        queries.append((f"q_hold_mut3_{len(queries)}", lin, mutate(s, 0.03, rng)))
    for i, (lin, s) in pick(train_r, 1000):
        queries.append((f"q_train_mut5_{len(queries)}", lin, mutate(s, 0.05, rng)))
    for i, (lin, s) in pick(hold_r, 1000):
        queries.append((f"q_hold_rc_{len(queries)}", lin, revcomp(mutate(s, 0.01, rng))))
    with open(outdir / "queries.fasta", "w") as fa, open(outdir / "queries_truth.tsv", "w") as tx:
        for qid, lin, s in queries:
            fa.write(f">{qid}\n{s}\n")
            tx.write(f"{qid}\tRoot;{lin}\n")
    print(f"train={len(train_r)} holdout={len(hold_r)} queries={len(queries)} species={len(set(l for l, _ in pool))}")


if __name__ == "__main__":
    main()
