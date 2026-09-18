"""Learned-aggregator arms against the mean control, pooled over the folds
each arm has scored. Reads metrics/[viewagg/]viewagg_{mean,<arm>}_<mod>_f<k>.json
and reports pooled recall / precision / AP50 per population, the arm/control
ratio, folds won, and a fold-paired t-test (df = folds-1).

    py scripts/viewagg_arms.py [attn gru ...]
"""
import glob, json, math, os, sys
from statistics import mean, stdev
try:
    from scipy import stats
except ImportError:
    stats = None
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics")
# the study keeps these under metrics/viewagg/; the replication package flattens them
d = os.path.join(_root, "viewagg") if os.path.isdir(os.path.join(_root, "viewagg")) else _root
arms = sys.argv[1:] or ["attn", "gru"]

def pop(j, key, p):
    return j["arms"][[k for k in j["arms"] if key in k][0]]["populations"][p]

for mod in ("thermal", "rgb"):
    for arm in arms:
        folds = sorted(int(f[-6]) for f in glob.glob(os.path.join(d, f"viewagg_{arm}_{mod}_f*.json")))
        if not folds:
            continue
        print(f"\n{mod} {arm} vs mean control, folds {folds}, 3 seeds each")
        for p in ("occluded", "visible"):
            for met in ("recall", "precision", "ap50"):
                tp = {"a": 0, "c": 0}; fp = {"a": 0, "c": 0}; n = 0; da = []; dc = []
                for k in folds:
                    ja = json.load(open(os.path.join(d, f"viewagg_{arm}_{mod}_f{k}.json")))
                    jm = json.load(open(os.path.join(d, f"viewagg_mean_{mod}_f{k}.json")))
                    pa, pc = pop(ja, f"_{arm}_", p), pop(jm, "viewgrid", p)
                    for key, q in (("a", pa), ("c", pc)):
                        tp[key] += sum(s["tp"] for s in q["per_seed"])
                        fp[key] += sum(s["fp"] for s in q["per_seed"])
                    n += pa["n_gt"] * 3; da.append(pa[met]["mean"]); dc.append(pc[met]["mean"])
                if met == "recall":
                    va, vc = tp["a"] / n, tp["c"] / n
                elif met == "precision":
                    va, vc = tp["a"] / (tp["a"] + fp["a"]), tp["c"] / (tp["c"] + fp["c"])
                else:
                    va, vc = mean(da), mean(dc)
                diff = [x - y for x, y in zip(da, dc)]
                t = mean(diff) / (stdev(diff) / math.sqrt(len(diff))) if len(diff) > 1 and stdev(diff) > 0 else float("nan")
                pv = 2 * stats.t.sf(abs(t), len(diff) - 1) if stats and len(diff) > 1 else float("nan")
                print(f"  {p:9}{met:10} ctrl {vc:.4f} {arm:4} {va:.4f} ratio {va/vc:.3f} "
                      f"folds {sum(x > 0 for x in diff)}/{len(diff)} t={t:.2f} p={pv:.3f}")
