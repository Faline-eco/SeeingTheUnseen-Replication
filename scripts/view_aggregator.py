"""Learned aggregation over the registered view stack, in embedding space.

The published `embed_multi` cell warps each source view's DINOv3 features onto
the focus-plane grid and takes the **mean** over the aperture. The mean is a
fixed aggregator: a fawn seen through a canopy gap in five of thirty-one views
contributes 5/31 of its feature vector, and a view in which it is fully
occluded is weighted exactly like one in which it is visible.

This trains the aggregator instead. Per target cell, a small network scores
every view against the aperture context and the views are combined by a masked
softmax over those scores. The final layer starts at zero, so at initialisation
the attention is uniform and the model **is** the published mean; anything it
learns is a departure from the baseline, not a re-implementation of it. The
same 0.36 M CenterNet head reads the aggregated grid, so the comparison with
`embed_multi` moves one thing: how the views are combined.

Inputs are not stored as stacks (31 x 128 x 128 x 128 per frame would be 1.4 TB)
but warped on the GPU from the per-source-frame embeddings, using the sampling
geometry `render_embedding_field.py --emit grid` writes (~2 MB per frame).

    py scripts\\view_aggregator.py --dataset viewgrid_thermal --agg attn

`--agg mean` is the control: it must reproduce `embed_multi` up to seed noise,
which is also the end-to-end check that the online warp equals the stored field.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_embedding_detector import (                       # noqa: E402
    EmbeddingDetector, build_targets, decode, evaluate, focal_loss,
)


# ----------------------------------------------------------------- data ----

class ViewBatch:
    """The tensors one batch of view stacks needs, movable as a unit.

    `evaluate` and the scorer call `x.to(device)` and `model(x)` on whatever
    the loader yields; giving them an object with those two entry points lets
    the existing training/scoring code run unchanged.
    """

    def __init__(self, feats, grid, valid):
        self.feats = feats      # (B, S, C, h, w)  float16, source-frame grids
        self.grid = grid        # (B, S, P, 2)     float32, grid_sample coords
        self.valid = valid      # (B, S, P)        bool

    def to(self, device, non_blocking: bool = False):
        return ViewBatch(self.feats.to(device, non_blocking=non_blocking),
                         self.grid.to(device, non_blocking=non_blocking),
                         self.valid.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        # DataLoader(pin_memory=True) calls this on custom batch types.
        return ViewBatch(self.feats.pin_memory(), self.grid.pin_memory(),
                         self.valid.pin_memory())


@functools.lru_cache(maxsize=512)
def _load_src(path: str) -> np.ndarray:
    # Consecutive centrals share ~28 of 31 sources, so even a shuffled loader
    # gets a useful hit rate per worker; 512 x 1 MB is the per-worker cost.
    return np.load(path)


class ViewStackDet(Dataset):
    """(view stack geometry + sources, boxes); boxes are YOLO cx cy w h."""

    def __init__(self, grid_dir: Path, label_dir: Path, out_hw: int = 0,
                 stems: set[str] | None = None, use_dims: int = 0,
                 srcframes: Path | None = None):
        self.grid_dir = Path(grid_dir)
        self.use_dims = use_dims
        # The grid store is named viewgrid_<mod>; the matching source-frame
        # store is srcframes_<mod> beside it, unless BAMBI_SRCFRAMES_ROOT
        # points at a faster mirror of the same files.
        if srcframes is None:
            name = self.grid_dir.name.replace("viewgrid", "srcframes")
            root = Path(os.environ.get("BAMBI_SRCFRAMES_ROOT", self.grid_dir.parent))
            srcframes = root / name
        self.srcframes = Path(srcframes)
        if not self.srcframes.is_dir():
            raise SystemExit(f"no source-frame store at {self.srcframes}")
        self.items: list[tuple[Path, Path]] = []
        for lp in sorted(Path(label_dir).glob("*.txt")):
            if stems is not None and lp.stem not in stems:
                continue
            gp = self.grid_dir / f"{lp.stem}.npz"
            if gp.exists():
                self.items.append((gp, lp))
        if not self.items:
            raise SystemExit(f"no (grid, label) pairs for {grid_dir} / {label_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        gp, lp = self.items[i]
        fid = lp.stem.split("_")[0]
        z = np.load(gp)
        grid = torch.from_numpy(z["grid"].astype(np.float32))          # (S, P, 2)
        valid = torch.from_numpy(
            np.unpackbits(z["valid"], axis=1, count=grid.shape[1]).astype(bool))
        feats = []
        for s in z["src"].tolist():
            a = _load_src(str(self.srcframes / f"{fid}_{s:06d}.npy"))    # (h, w, C)
            if self.use_dims:
                a = a[..., :self.use_dims]
            feats.append(torch.from_numpy(np.ascontiguousarray(a)))
        feats = torch.stack(feats).permute(0, 3, 1, 2).contiguous()      # (S, C, h, w)
        boxes = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append([float(v) for v in p[1:]])
        return (feats, grid, valid), torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), i


def collate(batch):
    """Pad the view axis: apertures at a flight's ends have fewer sources."""
    xs, bs, idx = zip(*batch)
    S = max(f.shape[0] for f, _, _ in xs)
    feats, grids, valids = [], [], []
    for f, g, v in xs:
        pad = S - f.shape[0]
        if pad:
            f = torch.cat([f, f.new_zeros(pad, *f.shape[1:])])
            g = torch.cat([g, g.new_full((pad, *g.shape[1:]), -2.0)])
            v = torch.cat([v, v.new_zeros(pad, *v.shape[1:])])
        feats.append(f); grids.append(g); valids.append(v)
    return (ViewBatch(torch.stack(feats), torch.stack(grids), torch.stack(valids)),
            list(bs), list(idx))


# ---------------------------------------------------------------- model ----


class SwinViewBlock(nn.Module):
    """One pre-norm transformer block over the tokens of a window.

    tok (Nw, T, d) with T = views * win * win, key_ok (Nw, T) bool. Invalid
    tokens are masked as keys (each token may always attend to itself, so a
    window with no valid view cannot produce NaN). Attention output projection
    and MLP output are zero-initialised, so the block is the identity at init.
    """

    def __init__(self, d: int, heads: int, win: int):
        super().__init__()
        self.heads = heads
        self.pos = nn.Parameter(torch.zeros(win * win, d))
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tok, key_ok):
        with torch.autocast("cuda", dtype=torch.float16, enabled=tok.is_cuda):
            n, t, d = tok.shape
            s = t // self.pos.shape[0]
            tok = tok + self.pos.repeat(s, 1).to(tok.dtype)
            h = self.norm1(tok)
            qkv = self.qkv(h).reshape(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            mask = key_ok[:, None, None, :] | torch.eye(t, dtype=torch.bool, device=tok.device)
            a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
            tok = tok + self.proj(a.transpose(1, 2).reshape(n, t, d))
            tok = tok + self.mlp(self.norm2(tok))
            return tok


class ViewAggregatorDetector(nn.Module):
    """warp -> aggregate over views -> the published CenterNet head.

    agg='mean' reproduces `integrate()` from render_embedding_field.py (sum of
    valid samples over their count, blanked below alpha_threshold). agg='attn'
    replaces the uniform weights with a masked softmax over per-(view, cell)
    scores computed from the view's features and the aperture mean.
    """

    def __init__(self, in_dim: int, width: int = 128, up: int = 4,
                 agg: str = "attn", hid: int = 32, alpha_threshold: float = 2.0,
                 win: int = 4):
        super().__init__()
        self.agg, self.hid, self.alpha_threshold, self.win = agg, hid, alpha_threshold, win
        self.head = EmbeddingDetector(in_dim, width, up)
        if agg == "attn":
            self.proj_v = nn.Conv2d(in_dim, hid, 1)
            self.proj_m = nn.Conv2d(in_dim, hid, 1)
            self.score = nn.Sequential(
                nn.Conv2d(2 * hid, hid, 3, padding=1), nn.SiLU(inplace=True),
                nn.Conv2d(hid, 1, 1))
            # Uniform attention at init: the model starts as the mean.
            nn.init.zeros_(self.score[-1].weight)
            nn.init.zeros_(self.score[-1].bias)
        elif agg == "gru":
            # Sequence model over the views of each cell, in flight order (the
            # grid store lists sources by frame index). Both directions run
            # with masked updates, so an invalid view leaves the state alone and
            # the final state is that of the last valid view. The aggregate is
            # the mean plus a zero-initialised readout of the two final states,
            # so here too the model starts as the published cell.
            self.proj_v = nn.Conv2d(in_dim, hid, 1)
            self.cell_f = nn.GRUCell(hid, hid)
            self.cell_b = nn.GRUCell(hid, hid)
            self.out = nn.Linear(2 * hid, in_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif agg == "swin":
            # Windowed self-attention over all (view, cell) tokens of a win x win
            # window, two pre-norm blocks, the second on a partition shifted by
            # win/2 (via padding, so windows never wrap). Learned view-index and
            # intra-window position embeddings. Afterwards a masked mean over
            # views per cell and a zero-initialised readout on top of the plain
            # mean, so the model starts as the published cell here too.
            self.proj_v = nn.Conv2d(in_dim, hid, 1)
            self.view_emb = nn.Parameter(torch.zeros(64, hid))
            nn.init.trunc_normal_(self.view_emb, std=0.02)
            self.blocks = nn.ModuleList([SwinViewBlock(hid, 4, win) for _ in range(2)])
            self.out = nn.Linear(hid, in_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif agg != "mean":
            raise ValueError(agg)
        # Cells per recurrence chunk: the 31-step scan keeps ~100 kB of
        # activations per cell for backward, so 32 k cells is ~3 GB, and each
        # chunk is checkpointed so only one is alive at a time.
        self.gru_chunk = 32768
        # Windows per attention chunk (each chunk checkpointed). At win=4 and
        # 31 views a window has 496 tokens; 1024 windows x 4 heads x 496^2 fp16
        # is 2 GB if the math kernel materialises the scores.
        self.swin_chunk = 1024

    @staticmethod
    def warp(feats, grid, valid, chunk: int = 128):
        """Source grids -> registered stack (B, S, C, H, W) in fp16.

        No gradient flows here (sources and geometry are fixed), and the fp32
        stack would be 4 GB at batch 16, so the warp runs in view chunks and
        writes straight into a preallocated fp16 tensor.
        """
        b, s, c, h, w = feats.shape
        p = grid.shape[2]
        hw = int(round(p ** 0.5))
        src = feats.reshape(b * s, c, h, w)
        g = grid.reshape(b * s, 1, p, 2)
        x = torch.empty(b * s, c, p, device=feats.device, dtype=torch.float16)
        for i in range(0, b * s, chunk):
            x[i:i + chunk] = F.grid_sample(
                src[i:i + chunk].float(), g[i:i + chunk], mode="bilinear",
                padding_mode="zeros", align_corners=False).squeeze(2).half()
        x = x.reshape(b, s, c, hw, hw)
        count = valid.reshape(b, s, 1, hw, hw).sum(1, dtype=torch.float32)
        return x, count

    def _scan(self, xs, m):
        """Masked bidirectional GRU over the view axis of one cell chunk.

        xs (N, S, hid), m (N, S) bool -> (N, 2*hid) final states. Runs under
        its own autocast so the checkpoint recompute matches the forward.
        """
        with torch.autocast("cuda", dtype=torch.float16, enabled=xs.is_cuda):
            n, s, _ = xs.shape
            outs = []
            for cell, order in ((self.cell_f, range(s)),
                                (self.cell_b, range(s - 1, -1, -1))):
                h = xs.new_zeros(n, self.hid)
                for t in order:
                    h = torch.where(m[:, t, None], cell(xs[:, t], h), h)
                outs.append(h)
            return torch.cat(outs, 1)

    def _gru_states(self, xs, m):
        from torch.utils.checkpoint import checkpoint
        outs = []
        for i in range(0, xs.shape[0], self.gru_chunk):
            xi, mi = xs[i:i + self.gru_chunk], m[i:i + self.gru_chunk]
            if torch.is_grad_enabled():
                outs.append(checkpoint(self._scan, xi, mi, use_reentrant=False))
            else:
                outs.append(self._scan(xi, mi))
        return torch.cat(outs)

    def _swin_partition(self, tok, ok, pad):
        """(B, S, H, W, d), (B, S, H, W) -> windows (Nw, S*win*win, d), (Nw, S*win*win).

        Pads H and W by `pad` on every side (invalid tokens), so the second
        block's shifted partition is just a partition of the padded grid.
        """
        w = self.win
        if pad:
            tok = F.pad(tok, (0, 0, pad, pad, pad, pad))
            ok = F.pad(ok, (pad, pad, pad, pad))
        b, s, h, ww, d = tok.shape
        nh, nw = h // w, ww // w
        tok = (tok.reshape(b, s, nh, w, nw, w, d).permute(0, 2, 4, 1, 3, 5, 6)
               .reshape(b * nh * nw, s * w * w, d))
        ok = (ok.reshape(b, s, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5)
              .reshape(b * nh * nw, s * w * w))
        return tok, ok, (b, s, nh, nw, d)

    def _swin_merge(self, tok, shape, pad):
        b, s, nh, nw, d = shape
        w = self.win
        tok = (tok.reshape(b, nh, nw, s, w, w, d).permute(0, 3, 1, 4, 2, 5, 6)
               .reshape(b, s, nh * w, nw * w, d))
        if pad:
            tok = tok[:, :, pad:-pad, pad:-pad]
        return tok

    def _swin_block(self, blk, tok, ok):
        """Run `blk` over the windows that hold at least one valid token; the
        others (outside the aperture's coverage) are left as they are, they
        are masked out at the pooling anyway."""
        from torch.utils.checkpoint import checkpoint
        live = ok.any(1).nonzero().squeeze(1)
        tl, ol, outs = tok[live], ok[live], []
        for i in range(0, tl.shape[0], self.swin_chunk):
            ti, oi = tl[i:i + self.swin_chunk], ol[i:i + self.swin_chunk]
            if torch.is_grad_enabled():
                outs.append(checkpoint(blk, ti, oi, use_reentrant=False))
            else:
                outs.append(blk(ti, oi))
        if live.numel() == tok.shape[0]:
            return torch.cat(outs)
        return tok.index_put((live,), torch.cat(outs))

    def aggregate(self, x, valid, count):
        b, s, c, hh, ww = x.shape
        # Reductions with dtype= accumulate in fp32 without materialising an
        # fp32 copy of the stack.
        mean = x.sum(1, dtype=torch.float32) / count.clamp(min=1)      # (B, C, H, W)
        if self.agg == "mean":
            out, attn = mean, None
        elif self.agg == "gru":
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                pv = self.proj_v(x.reshape(b * s, c, hh, ww))          # (B*S, hid, H, W)
            n = b * hh * ww
            xs = (pv.reshape(b, s, self.hid, hh * ww).permute(0, 3, 1, 2)
                  .reshape(n, s, self.hid))                             # one sequence per cell
            m = valid.reshape(b, s, hh * ww).permute(0, 2, 1).reshape(n, s)
            h = self._gru_states(xs, m)                                 # (N, 2*hid)
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                delta = self.out(h)                                     # (N, C)
            delta = delta.float().reshape(b, hh, ww, c).permute(0, 3, 1, 2)
            out, attn = mean + delta, None
        elif self.agg == "swin":
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                pv = self.proj_v(x.reshape(b * s, c, hh, ww))          # (B*S, hid, H, W)
            tok = pv.reshape(b, s, self.hid, hh, ww).permute(0, 1, 3, 4, 2)   # (B, S, H, W, hid)
            tok = (tok + self.view_emb[:s, None, None, :].to(tok.dtype)).contiguous()
            for i, blk in enumerate(self.blocks):
                pad = 0 if i % 2 == 0 else self.win // 2
                wt, wk, shape = self._swin_partition(tok, valid.reshape(b, s, hh, ww), pad)
                tok = self._swin_merge(self._swin_block(blk, wt, wk), shape, pad)
            vm = valid.reshape(b, s, hh, ww, 1).to(tok.dtype)
            pooled = (tok * vm).sum(1, dtype=torch.float32) / count.reshape(b, hh, ww, 1).clamp(min=1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                delta = self.out(pooled)                                # (B, H, W, C)
            delta = delta.float().permute(0, 3, 1, 2)
            out, attn = mean + delta, None
        else:
            # The stack is fp16 whether or not the caller runs under autocast
            # (evaluate() does not), so the scorer always casts for itself.
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                pv = self.proj_v(x.reshape(b * s, c, hh, ww))
                pm = self.proj_m(mean).unsqueeze(1).expand(b, s, self.hid, hh, ww)
                z = torch.cat([pv.reshape(b, s, self.hid, hh, ww), pm], 2)
                sc = self.score(z.reshape(b * s, 2 * self.hid, hh, ww))
            sc = sc.reshape(b, s, 1, hh, ww)
            sc = sc.float().masked_fill(~valid.reshape(b, s, 1, hh, ww), float("-inf"))
            attn = torch.softmax(sc, dim=1).nan_to_num(0.0)            # (B, S, 1, H, W)
            out = (attn.to(x.dtype) * x).sum(1, dtype=torch.float32)
        out = out * (count >= self.alpha_threshold)
        return out, attn

    def forward(self, batch: ViewBatch):
        x, count = self.warp(batch.feats, batch.grid, batch.valid)
        out, _ = self.aggregate(x, batch.valid, count)
        return self.head(out)

    @torch.no_grad()
    def attention(self, batch: ViewBatch):
        """(aggregated grid, attention weights) for inspection."""
        x, count = self.warp(batch.feats, batch.grid, batch.valid)
        return self.aggregate(x, batch.valid, count)

    def config(self) -> dict:
        return {"kind": "view_aggregator", "in_dim": self.head.stem[0].in_channels,
                "width": self.head.stem[0].out_channels,
                "up": 2 ** len([m for m in self.head.up if isinstance(m, nn.ConvTranspose2d)]),
                "agg": self.agg, "hid": self.hid, "win": self.win,
                "alpha_threshold": self.alpha_threshold}

    @classmethod
    def from_checkpoint(cls, ck: dict) -> "ViewAggregatorDetector":
        m = cls(ck["in_dim"], ck["width"], ck["up"], ck["agg"], ck["hid"],
                ck["alpha_threshold"], ck.get("win", 4))
        m.load_state_dict(ck["model"])
        return m


# ------------------------------------------------------------------ main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="viewgrid_thermal",
                    help="Grid store under --embeddings (viewgrid_<modality>).")
    ap.add_argument("--labels-dataset", required=True,
                    help="YOLO dataset whose labels/{train,val} define the fold.")
    ap.add_argument("--embeddings", type=Path, default=Path("D:/embeddings"))
    ap.add_argument("--srcframes", type=Path, default=None,
                    help="Source-frame embedding store; default derives it from "
                         "--dataset (see ViewStackDet).")
    ap.add_argument("--yolo-datasets", type=Path, default=Path("D:/yolo_datasets"))
    ap.add_argument("--eval-labels", default=None)
    ap.add_argument("--out", type=Path, default=Path("D:/embedding_runs"))
    ap.add_argument("--agg", default="attn", choices=("attn", "mean", "gru", "swin"))
    ap.add_argument("--hid", type=int, default=32,
                    help="scorer width (attn), GRU state size per direction (gru), "
                         "token dim (swin)")
    ap.add_argument("--win", type=int, default=4, help="swin window side in grid cells")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--up", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--use-dims", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: keep only the first N train / N val frames")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_dir = args.embeddings / args.dataset
    lab = args.yolo_datasets / args.labels_dataset / "labels"

    tr = ViewStackDet(grid_dir, lab / "train", 0, None, args.use_dims, args.srcframes)
    va = ViewStackDet(grid_dir, lab / "val", 0, None, args.use_dims, args.srcframes)
    if args.limit:
        tr.items, va.items = tr.items[:args.limit], va.items[:args.limit]
    (f0, g0, _), _, _ = tr[0]
    z0, fid0 = np.load(tr.items[0][0]), tr.items[0][1].stem.split("_")[0]
    stored_dim = np.load(tr.srcframes / f"{fid0}_{int(z0['src'][0]):06d}.npy").shape[-1]
    in_dim = f0.shape[1]
    grid = int(round(g0.shape[1] ** 0.5))
    out_hw = grid * args.up
    print(f"{args.dataset}: {len(tr)} train / {len(va)} val | sources {tr.srcframes} | "
          f"{f0.shape[0]} views x {f0.shape[2]}x{f0.shape[3]}x{in_dim} -> grid "
          f"{grid}x{grid} -> head {out_hw}x{out_hw} | agg={args.agg}", flush=True)

    gt_labels = None
    if args.eval_labels:
        gt_labels = {}
        for lp in (args.yolo_datasets / args.eval_labels / "labels" / "val").glob("*.txt"):
            rows = [[float(v) for v in l.split()[1:5]]
                    for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            gt_labels[lp.stem] = torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)
        print(f"  scoring against {args.eval_labels} labels "
              f"({sum(len(v) for v in gt_labels.values())} boxes)", flush=True)

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)

    model = ViewAggregatorDetector(in_dim, args.width, args.up, args.agg, args.hid,
                                   win=args.win).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device)
    print(f"  head: {n_head/1e6:.2f} M params, aggregator: {(n_par-n_head)/1e3:.1f} k",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    run = args.out / (f"{args.dataset}_{args.tag}" if args.tag else args.dataset)
    run.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = -1.0, -1, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        te = time.time()
        for x, boxes_list, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device):
                hm, wh, off = model(x)
                thm, twh, toff, m = build_targets(boxes_list, hm.shape[-1], device)
                l_hm = focal_loss(hm.float(), thm)
                n = m.sum().clamp(min=1)
                l_wh = (F.l1_loss(wh.float() * m, twh * m, reduction="sum") / n)
                l_off = (F.l1_loss(off.float() * m, toff * m, reduction="sum") / n)
                loss = l_hm + 5.0 * l_wh + l_off
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
        sched.step()

        met = evaluate(model, dl_va, device, gt_labels)
        fit = 0.1 * met["mAP50"] + 0.9 * met["mAP50-95"]
        hist.append({"epoch": ep, "loss": tot / max(len(dl_tr), 1), **met,
                     "minutes": round((time.time() - te) / 60, 1)})
        print(f"  ep {ep:3d}  loss {tot/max(len(dl_tr),1):7.4f}  "
              f"mAP50 {met['mAP50']:.4f}  mAP50-95 {met['mAP50-95']:.4f}  "
              f"P {met['precision']:.4f}  R {met['recall']:.4f}  "
              f"[{(time.time()-te)/60:.1f} min, "
              f"{torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else 0:.1f} GB]",
              flush=True)
        if fit > best:
            best, best_ep = fit, ep
            torch.save({"model": model.state_dict(), **model.config(), "metrics": met},
                       run / "best.pt")
        (run / "results.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        if ep - best_ep >= args.patience:
            print(f"  early stop (best epoch {best_ep})", flush=True)
            break

    best_met = next(h for h in hist if h["epoch"] == best_ep)
    summary = {"name": f"emb_{args.dataset}" + (f"_{args.tag}" if args.tag else ""),
               "mAP50": round(best_met["mAP50"], 4),
               "mAP50-95": round(best_met["mAP50-95"], 4),
               "precision": round(best_met["precision"], 4),
               "recall": round(best_met["recall"], 4),
               "best_epoch": best_ep, "epochs_run": len(hist),
               "minutes": round((time.time() - t0) / 60, 1),
               "params_M": round(n_par / 1e6, 2), "head_params_M": round(n_head / 1e6, 2),
               "agg": args.agg, "hid": args.hid, "win": args.win,
               "eval_labels": args.eval_labels or args.labels_dataset,
               "labels_dataset": args.labels_dataset,
               "use_dims": args.use_dims or stored_dim, "stored_dims": stored_dim}
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBEST epoch {best_ep}: mAP50 {best_met['mAP50']:.4f} "
          f"mAP50-95 {best_met['mAP50-95']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
