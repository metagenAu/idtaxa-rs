#!/usr/bin/env python3
"""Agreement and confidence comparison between two classification runs.

Inputs are TSVs with columns `id` plus either
  * `full_taxonomy` / `full_confidences` (idtaxa classify --full-confidences), or
  * `taxonomy` / `confidences` produced with threshold 0
    (scripts/run_decipher_reference.R).
Both list the full predicted lineage below Root with per-rank confidences,
so any threshold can be applied here.

Usage:
  compare.py A.tsv B.tsv [--threshold 60] [--truth truth.tsv] [--label-a X --label-b Y]
             [--baseline A2.tsv]   # second run of A (e.g. DECIPHER, other seed)
             [--json out.json]

Reports per-rank assignment agreement at the threshold, mean absolute
confidence differences, the distribution of disagreements relative to the
threshold, a confusion summary of disagreements and (optionally) accuracy
against the truth. With --baseline, the same statistics are computed for
A vs A2 (run-to-run variability of A) for reference.
"""
import argparse
import json
import statistics
from collections import Counter


def load(path):
    rows = {}
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                if "full_taxonomy" in header:
                    ti, ci = header.index("full_taxonomy"), header.index("full_confidences")
                else:
                    ti, ci = header.index("taxonomy"), header.index("confidences")
                ii = header.index("id")
                continue
            tax = [x for x in parts[ti].split(";") if x] if ti < len(parts) else []
            conf = [float(x) for x in parts[ci].split(";") if x] if ci < len(parts) else []
            # DECIPHER marks unclassifiable queries as unclassified_Root
            if tax and tax[0].startswith("unclassified_"):
                tax, conf = [], []
            rows[parts[ii]] = (tax, conf)
    return rows


def assigned(tax, conf, t, d):
    """Path assigned at depth d (1-based) under threshold t, or None."""
    if len(tax) < d or len(conf) < d:
        return None
    if all(c >= t for c in conf[:d]):
        return ";".join(tax[:d])
    return None


def compare(a, b, t, max_depth):
    ids = sorted(set(a) & set(b))
    out = {"n": len(ids), "ranks": []}
    for d in range(1, max_depth + 1):
        agree = 0
        n = 0
        diffs_all, diffs_same = [], []
        near = []
        confusion = Counter()
        for i in ids:
            ta, ca = a[i]
            tb, cb = b[i]
            if len(ta) < d and len(tb) < d:
                continue
            n += 1
            pa, pb = assigned(ta, ca, t, d), assigned(tb, cb, t, d)
            if pa == pb:
                agree += 1
            else:
                cva = ca[d - 1] if len(ca) >= d else 0.0
                cvb = cb[d - 1] if len(cb) >= d else 0.0
                # distance of the disagreement from the threshold
                near.append(min(abs(cva - t), abs(cvb - t)))
                kind = (
                    "A_only" if pb is None else "B_only" if pa is None else "different_taxon"
                )
                confusion[kind] += 1
            if len(ca) >= d and len(cb) >= d:
                diffs_all.append(abs(ca[d - 1] - cb[d - 1]))
                if ta[:d] == tb[:d]:
                    diffs_same.append(abs(ca[d - 1] - cb[d - 1]))
        hist = Counter()
        for x in near:
            hist["<=5" if x <= 5 else "5-10" if x <= 10 else "10-20" if x <= 20 else ">20"] += 1
        out["ranks"].append(
            {
                "depth": d,
                "n": n,
                "agreement": agree / n if n else float("nan"),
                "mean_abs_conf_diff": statistics.fmean(diffs_all) if diffs_all else float("nan"),
                "mean_abs_conf_diff_same_taxon": statistics.fmean(diffs_same) if diffs_same else float("nan"),
                "disagreements": n - agree,
                "disagreement_distance_from_threshold": dict(hist),
                "disagreement_types": dict(confusion),
            }
        )
    flat_all = []
    for i in ids:
        ta, ca = a[i]
        tb, cb = b[i]
        for d in range(min(len(ca), len(cb))):
            flat_all.append(abs(ca[d] - cb[d]))
    out["mean_abs_conf_diff_all_ranks"] = statistics.fmean(flat_all) if flat_all else float("nan")
    tot = sum(r["n"] for r in out["ranks"])
    out["agreement_all_ranks"] = sum(r["n"] * r["agreement"] for r in out["ranks"] if r["n"]) / tot if tot else float("nan")
    return out


def accuracy(rows, truth, t, max_depth):
    res = []
    for d in range(1, max_depth + 1):
        tp = wrong = un = 0
        for i, tr in truth.items():
            if i not in rows or len(tr) < d:
                continue
            p = assigned(*rows[i], t, d)
            if p is None:
                un += 1
            elif p == ";".join(tr[:d]):
                tp += 1
            else:
                wrong += 1
        n = tp + wrong + un
        res.append({"depth": d, "n": n, "correct": tp / n if n else float("nan"),
                    "wrong": wrong / n if n else float("nan"), "unassigned": un / n if n else float("nan")})
    return res


def load_truth(p):
    t = {}
    with open(p) as f:
        for line in f:
            i, lin = line.rstrip("\n").split("\t", 1)
            names = [x for x in lin.strip().rstrip(";").split(";") if x]
            if names and names[0] == "Root":
                names = names[1:]
            t[i] = names
    return t


def fmt(cmp, la, lb):
    lines = [f"### {la} vs {lb} (n = {cmp['n']})", "",
             "| depth | n | agreement | mean |Δconf| | mean |Δconf| (same taxon) | disagreements | distance from threshold (≤5/5-10/10-20/>20) | types |",
             "|---|---|---|---|---|---|---|---|"]
    for r in cmp["ranks"]:
        h = r["disagreement_distance_from_threshold"]
        lines.append(
            f"| {r['depth']} | {r['n']} | {100*r['agreement']:.2f}% | {r['mean_abs_conf_diff']:.2f} | "
            f"{r['mean_abs_conf_diff_same_taxon']:.2f} | {r['disagreements']} | "
            f"{h.get('<=5',0)}/{h.get('5-10',0)}/{h.get('10-20',0)}/{h.get('>20',0)} | "
            + ", ".join(f"{k}={v}" for k, v in sorted(r["disagreement_types"].items())) + " |")
    lines.append("")
    lines.append(f"Overall agreement (all ranks): {100*cmp['agreement_all_ranks']:.2f}%; "
                 f"mean |Δconfidence| over all ranks: {cmp['mean_abs_conf_diff_all_ranks']:.2f} points")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--truth")
    ap.add_argument("--baseline")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json")
    ap.add_argument("--id-regex", help="only compare ids matching this regular expression")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    if args.id_regex:
        import re
        rx = re.compile(args.id_regex)
        a = {k: v for k, v in a.items() if rx.search(k)}
        b = {k: v for k, v in b.items() if rx.search(k)}
    md = max(max((len(x[0]) for x in a.values()), default=0), max((len(x[0]) for x in b.values()), default=0))
    report = {"threshold": args.threshold, "a_vs_b": compare(a, b, args.threshold, md)}
    print(fmt(report["a_vs_b"], args.label_a, args.label_b))
    if args.baseline:
        a2 = load(args.baseline)
        if args.id_regex:
            a2 = {k: v for k, v in a2.items() if rx.search(k)}
        report["a_vs_a2"] = compare(a, a2, args.threshold, md)
        print()
        print(fmt(report["a_vs_a2"], args.label_a, args.label_a + " (second seed)"))
    if args.truth:
        truth = load_truth(args.truth)
        report["accuracy"] = {args.label_a: accuracy(a, truth, args.threshold, md),
                              args.label_b: accuracy(b, truth, args.threshold, md)}
        print()
        print("| depth | " + " | ".join(f"{l} correct / wrong / unassigned" for l in (args.label_a, args.label_b)) + " |")
        print("|---|---|---|")
        for ra, rb in zip(*report["accuracy"].values()):
            print(f"| {ra['depth']} | {100*ra['correct']:.1f} / {100*ra['wrong']:.1f} / {100*ra['unassigned']:.1f} | "
                  f"{100*rb['correct']:.1f} / {100*rb['wrong']:.1f} / {100*rb['unassigned']:.1f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
