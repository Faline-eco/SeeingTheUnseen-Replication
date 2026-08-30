"""Dump every embedding-detector run as one machine-readable results table.

Written for the backup: the documentation should quote numbers that came out of
the run directories, not numbers retyped from a chat log.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("/scratch/bambi/datasets/alfs_embed")
OUT = Path("/scratch/bambi/alfs_embed/backups/results_table.json")

rows = []
for d in ("embedding_runs", "embedding_runs_multiseed"):
    for f in sorted((ROOT / d).glob("*/summary.json")):
        m = json.loads(f.read_text())
        m["_run_dir"] = f.parent.name
        m["_group"] = d
        rows.append(m)

arms: dict[str, list[dict]] = {}
for m in rows:
    key = m["name"].rsplit("_s", 1)[0] if "_s" in m["name"] else m["name"]
    arms.setdefault((m["_group"], key), []).append(m)

table = []
for (group, arm), runs in sorted(arms.items()):
    def agg(field):
        v = [r[field] for r in runs if field in r]
        if not v:
            return None
        return {"mean": round(st.mean(v), 4),
                "stdev": round(st.stdev(v), 4) if len(v) > 1 else None,
                "n": len(v), "values": [round(x, 4) for x in v]}
    r0 = runs[0]
    table.append({
        "group": group,
        "arm": arm,
        "seeds": sorted(r.get("name", "").rsplit("_s", 1)[-1] for r in runs),
        "labels_dataset": r0.get("labels_dataset"),
        "eval_labels": r0.get("eval_labels"),
        "use_dims": r0.get("use_dims"),
        "stored_dims": r0.get("stored_dims"),
        "params_M": r0.get("params_M"),
        "epochs_run": [r.get("epochs_run") for r in runs],
        "best_epoch": [r.get("best_epoch") for r in runs],
        "mAP50": agg("mAP50"),
        "mAP50-95": agg("mAP50-95"),
        "precision": agg("precision"),
        "recall": agg("recall"),
    })

OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")
print(f"wrote {OUT}  ({len(table)} arms from {len(rows)} runs)\n")
hdr = f"{'arm':40s} {'n':>2} {'mAP50-95':>18} {'mAP50':>18}"
print(hdr); print("-" * len(hdr))
for t in table:
    def f(a):
        if not a:
            return "-"
        return f"{a['mean']:.4f}+/-{a['stdev']:.4f}" if a["stdev"] else f"{a['mean']:.4f}"
    print(f"{t['arm']:40s} {t['mAP50-95']['n']:2d} "
          f"{f(t['mAP50-95']):>18} {f(t['mAP50']):>18}")
