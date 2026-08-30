import sys, torch
from pathlib import Path
sys.path.insert(0, "scripts")
from train_embedding_detector import EmbeddingDetector

R = Path("/scratch/bambi/datasets/alfs_embed/embedding_runs_multiseed")
TAGS = ["cell_thermal_embed_single_embed_single_f0_s1337",
        "cell_thermal_embed_multi_embed_multi_f0_s1337",
        "cell_thermal_embed_multi_embed_multi_w512_f0_s1337",
        "cell_thermal_vjepa_single_vjepa_single_f0_s1337",
        "cell_thermal_vjepa_multi_vjepa_multi_f0_s1337",
        "cell_rgb_vjepa_multi_vjepa_multi_f0_s1337",
        "cell_thermal_realortho_single_realortho_single_f0_s1337",
        "cell_thermal_realalfs_multi_realalfs_multi_f0_s1337"]
print(f"{'run':<56} {'in_dim':>7} {'width':>6} {'up':>3} {'params':>10}")
for t in TAGS:
    p = R / t / "best.pt"
    if not p.is_file():
        print(f"{t:<56}   (missing)")
        continue
    ck = torch.load(p, map_location="cpu")
    m = EmbeddingDetector(ck["in_dim"], ck["width"], ck["up"])
    n = sum(x.numel() for x in m.parameters())
    print(f"{t:<56} {ck['in_dim']:>7} {ck['width']:>6} {ck['up']:>3} {n/1e6:9.2f}M")
