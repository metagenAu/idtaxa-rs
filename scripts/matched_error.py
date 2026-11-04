#!/usr/bin/env python3
"""Compare configurations at matched error rates from `idtaxa validate` TSVs.

For every configuration and rank, the threshold grid in the validate TSV is
searched for the highest accuracy (TP / present) whose error rate
((MC + OC) / all held-out sequences) does not exceed each target. This
removes the effect of a configuration merely shifting the confidence scale.

Usage: matched_error.py validate.tsv [more.tsv ...] [--targets 0.01,0.02,0.05] [--depths 6,7,8,9]
"""
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--targets", default="0.01,0.02,0.05")
    ap.add_argument("--depths", default="")
    ap.add_argument("--label", action="append", default=[], help="rename configs: old=new")
    args = ap.parse_args()
    ren = dict(x.split("=", 1) for x in args.label)
    rows = defaultdict(list)  # (config, depth) -> [(thr, acc, err, oc)]
    ranks = {}
    order = []
    for p in args.tsv:
        with open(p) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {k: i for i, k in enumerate(hdr)}
            for line in f:
                v = line.rstrip("\n").split("\t")
                if v[ix["threshold"]] == "calibrated":
                    continue
                cfg = ren.get(v[ix["config"]], v[ix["config"]])
                if p != args.tsv[0] or True:
                    pass
                key = (cfg, int(v[ix["depth"]]))
                if cfg not in order:
                    order.append(cfg)
                ranks[int(v[ix["depth"]])] = v[ix["rank"]]
                acc = float(v[ix["accuracy"]]) if v[ix["accuracy"]] != "NaN" else 0.0
                rows[key].append((float(v[ix["threshold"]]), acc, float(v[ix["error_rate"]]),
                                  float(v[ix["overclassification"]]) if v[ix["overclassification"]] != "NaN" else float("nan")))
    targets = [float(x) for x in args.targets.split(",")]
    depths = [int(x) for x in args.depths.split(",")] if args.depths else sorted(ranks)
    print("| rank | target error | " + " | ".join(order) + " |")
    print("|---|---|" + "---|" * len(order))
    for d in depths:
        for t in targets:
            cells = []
            for cfg in order:
                best = None
                for thr, acc, err, oc in rows.get((cfg, d), []):
                    if err <= t and (best is None or acc > best[1]):
                        best = (thr, acc)
                cells.append("–" if best is None else f"{100*best[1]:.1f}% (t={best[0]:g})")
            print(f"| {ranks[d]} | {100*t:g}% | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
