"""Emit every scored YOLO26x cross-validation metric as markdown."""
import json
import math
import statistics as st
import sys
from pathlib import Path

M = Path("metrics")
POPS = [("visible", "Visible"), ("occluded", "Hidden"), ("all", "All")]
METRICS = [("recall", "recall"), ("precision", "precision"), ("f1", "F1"),
           ("ap50", "AP50"), ("ap75", "AP75"), ("ap50_95", "AP50-95")]


def val(e, pop, key):
    v = e.get("populations", {}).get(pop, {}).get(key)
    return v.get("mean") if isinstance(v, dict) else v


def seedsum(e, pop, key):
    p = e.get("populations", {}).get(pop, {})
    return p.get("n_gt", 0) if key == "n_gt" else \
        sum(s.get(key, 0) for s in p.get("per_seed", []))


def plab(t):
    a = abs(t)
    return "**<.01**" if a > 4.604 else "**<.02**" if a > 3.747 else \
           "**<.05**" if a > 2.776 else "<.1" if a > 2.132 else "ns"


out = []
for mod in ("thermal", "rgb"):
    folds = [json.loads((M / f"yolocv_{mod}_f{k}.json").read_text())["arms"]
             for k in range(5) if (M / f"yolocv_{mod}_f{k}.json").exists()]
    if not folds:
        continue
    out.append(f"\n## {mod.upper()}\n")
    for pop, plabel in POPS:
        out.append(f"\n### {plabel} population\n")
        out.append("| metric | ortho (single) | ALFS (multi) | ratio | t(4) | p | folds won |")
        out.append("|---|---|---|---|---|---|---|")
        for key, name in METRICS:
            a = [val(f[f"ortho_{mod}_f{i}"], pop, key) for i, f in enumerate(folds)]
            b = [val(f[f"alfs_{mod}_f{i}"], pop, key) for i, f in enumerate(folds)]
            d = [y - x for x, y in zip(a, b)]
            t = st.mean(d) / (st.stdev(d) / math.sqrt(len(d))) if st.stdev(d) else float("inf")
            r = st.mean(b) / st.mean(a)
            bold = "**" if r >= 1.5 or r <= 0.67 else ""
            out.append(f"| {name} | {st.mean(a):.4f} +- {st.stdev(a):.4f} "
                       f"| {st.mean(b):.4f} +- {st.stdev(b):.4f} "
                       f"| {bold}{r:.3f}{bold} | {t:.2f} | {plab(t)} "
                       f"| {sum(1 for v in d if v > 0)}/5 |")
        tp_o = sum(seedsum(f[f"ortho_{mod}_f{i}"], pop, "tp") for i, f in enumerate(folds))
        tp_a = sum(seedsum(f[f"alfs_{mod}_f{i}"], pop, "tp") for i, f in enumerate(folds))
        fn_o = sum(seedsum(f[f"ortho_{mod}_f{i}"], pop, "fn") for i, f in enumerate(folds))
        fn_a = sum(seedsum(f[f"alfs_{mod}_f{i}"], pop, "fn") for i, f in enumerate(folds))
        n = sum(seedsum(f[f"ortho_{mod}_f{i}"], pop, "n_gt") for i, f in enumerate(folds))
        out.append(f"| TP / FN | {tp_o} / {fn_o} | {tp_a} / {fn_a} | | | | |")
        out.append(f"\n{n:,} ground-truth boxes summed over five folds "
                   f"(each counted once per fold, three seeds each).")
print("\n".join(out))
