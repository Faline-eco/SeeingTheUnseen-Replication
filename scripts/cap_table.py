"""Head-capacity ablation: 0.36 M vs 3.90 M CenterNet head, paired by fold.

Same statistics as the debias study -- per-fold differences over the five
scene-level folds, a paired t-test, and how many folds the wider head wins.
The metrics store calls the hidden population `occluded`; the hidden mask in
`meta` is what restricts it, so it is the same population the paper reports.
"""
import json
import math
import statistics as st
from pathlib import Path

M = Path("/scratch/bambi/datasets/alfs_embed/metrics")
BASE = {"single": "cell_thermal_embed_single_embed_single",
        "multi":  "cell_thermal_embed_multi_embed_multi"}
WIDE = {"single": "cell_thermal_embed_single_embed_single_w512",
        "multi":  "cell_thermal_embed_multi_embed_multi_w512"}
POPS = [("visible", "recall"), ("visible", "precision"),
        ("visible", "ap50"), ("visible", "ap50_95"),
        ("occluded", "recall"), ("occluded", "precision"),
        ("occluded", "ap50"), ("occluded", "ap50_95")]
LABEL = {"visible": "visible", "occluded": "hidden", "ap50_95": "AP50-95",
         "ap50": "AP50", "recall": "recall", "precision": "precision"}


def arms(fold):
    out = {}
    for tag in ("thermal", "w512_thermal"):
        p = M / f"{tag}_f{fold}.json"
        if p.exists():
            out.update(json.loads(p.read_text())["arms"])
    return out


def val(a, name, fold, pop, metric):
    e = a.get(f"{name}_f{fold}")
    if e is None:
        return None
    v = e.get("populations", {}).get(pop, {}).get(metric)
    return v.get("mean") if isinstance(v, dict) else v


def tstat(d):
    if len(d) < 2:
        return float("nan")
    s = st.stdev(d)
    return st.mean(d) / (s / math.sqrt(len(d))) if s else float("inf")


def plabel(t):
    a = abs(t)
    return "< .01" if a > 4.604 else "< .02" if a > 3.747 else \
           "< .05" if a > 2.776 else "< .1" if a > 2.132 else "ns"


print(f"{'cell':<7} {'pop':<8} {'metric':<10} {'0.36M':>8} {'3.90M':>8} "
      f"{'ratio':>7} {'t(4)':>7} {'p':>6} {'won':>5}")
sig = 0
for cell in ("single", "multi"):
    for pop, metric in POPS:
        b, w, d = [], [], []
        for f in range(5):
            A = arms(f)
            x, y = val(A, BASE[cell], f, pop, metric), val(A, WIDE[cell], f, pop, metric)
            if x is None or y is None:
                continue
            b.append(x); w.append(y); d.append(y - x)
        if not d:
            print(f"{cell:<7} {LABEL[pop]:<8} {LABEL[metric]:<10}   (missing)")
            continue
        mb, mw, t = st.mean(b), st.mean(w), tstat(d)
        p = plabel(t)
        sig += p in ("< .05", "< .02", "< .01")
        print(f"{cell:<7} {LABEL[pop]:<8} {LABEL[metric]:<10} {mb:8.4f} {mw:8.4f} "
              f"{mw/mb if mb else float('nan'):7.3f} {t:7.2f} {p:>6} "
              f"{sum(1 for v in d if v > 0):>3}/5")
print(f"\n{sig} of {2 * len(POPS)} comparisons reach p < 0.05")
