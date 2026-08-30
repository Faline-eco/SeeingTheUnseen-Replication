"""Precision and AP on hidden animals for the primary 2x2, both modalities.

The paper's tab:precision reports thermal only. This regenerates it and the RGB
counterpart from the same scored metrics the recall tables come from.
"""
import glob
import json
import os
import statistics as st

M = "/scratch/bambi/datasets/alfs_embed/metrics"
CELLS = [("image / single", "cell_{m}_realortho_single_realortho_single"),
         ("image / multi",  "cell_{m}_realalfs_multi_realalfs_multi"),
         ("emb. / single",  "cell_{m}_embed_single_embed_single"),
         ("emb. / multi",   "cell_{m}_embed_multi_embed_multi")]
WANT = ["recall", "precision", "ap50", "ap50_95"]


def arms(mod, fold):
    out = {}
    for p in glob.glob(f"{M}/*_f{fold}.json"):
        if f"_{mod}_" not in os.path.basename(p) and not os.path.basename(p).startswith(mod):
            continue
        try:
            j = json.loads(open(p).read())
        except Exception:
            continue
        if j.get("meta", {}).get("modality") != mod:
            continue
        out.update(j["arms"])
    return out


for mod in ("thermal", "rgb"):
    print(f"\n=== {mod}: hidden population")
    print(f"{'cell':<16} " + " ".join(f"{w:>10}" for w in WANT))
    for label, tpl in CELLS:
        name = tpl.format(m=mod)
        cols = {w: [] for w in WANT}
        for f in range(5):
            e = arms(mod, f).get(f"{name}_f{f}")
            if e is None:
                continue
            occ = e.get("populations", {}).get("occluded", {})
            for w in WANT:
                v = occ.get(w)
                v = v.get("mean") if isinstance(v, dict) else v
                if v is not None:
                    cols[w].append(v)
        if not cols["recall"]:
            print(f"{label:<16}  (missing: {name})")
            continue
        row = f"{label:<16} "
        for w in WANT:
            row += f" {st.mean(cols[w]):10.4f}" if cols[w] else f" {'--':>10}"
        n = len(cols['recall'])
        print(row + f"   [{n} folds]")
