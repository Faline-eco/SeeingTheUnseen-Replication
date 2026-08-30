# Embedding encoding, detector training and the GL-free light-field integral
# for the BAMBI ALFS comparison.
#
# What must NOT run here: ALFS/orthographic *rendering*. That needs
# ModernGL/OpenGL, which in a container falls back to software rasterisation and
# has previously produced renderings with artifacts. Rendering stays on a
# workstation with a real GL driver; only its PNG output comes here.
#
# The *embedded* light field is different, and does run here:
# render_embedding_field.py reimplements the integral with embree ray-casts and
# torch grid_sample instead of GL, so it is unaffected by that failure mode. It
# agrees with the GL renderer to r=0.9996 (MAE 0.012) on the RGB self-test, and
# its virtual camera is bit-identical (max|diff| 0.0 on view/proj/position).
#
# Build (on the DGX, from this directory):
#   docker build -t bambi-embed:1.5 .
#
# Run (pin to free GPUs — the machine is shared):
#   docker run --rm --gpus '"device=2,3,4,5"' \
#       -v /scratch/bambi:/scratch/bambi -w /scratch/bambi/alfs_embed/code \
#       bambi-embed:1.5 python scripts/encode_embeddings.py --help
# or use dgx_run.sh, which refuses to start on a GPU another user is occupying.
# NOT the 2.3.1-cuda11.8 base: transformers >= 4.56 (required for DINOv3)
# needs PyTorch >= 2.4 and silently disables the PyTorch backend below that
# ("Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.1"),
# so the extractor would fail at load time. 2.6.0/cu124 also matches the
# workstation environment exactly, so both machines compute on the same torch.
# Host driver 595.71.05 supports up to CUDA 13.2, so cu124 is safe here.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# libgl1/libglib2.0 are for opencv's runtime deps; we use the headless wheel but
# a few cv2 codecs still link against them.
# Even the *headless* opencv wheel links against libxcb/libX11 in this base
# image (ImportError: libxcb.so.1). These are the shared objects only, no X
# server and no GL driver - rendering deliberately does not happen here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 git \
        libxcb1 libx11-6 libxext6 libxrender1 libsm6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ultralytics depends on opencv-python (not the headless wheel) and pulls it in
# over ours, which then needs libGL at import. Swap it back afterwards so the
# image stays GUI-free; libgl1 above covers anything that still links it.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless==5.0.0.93"
# --no-deps matters: without it this force-reinstall re-resolves opencv's
# dependency chain and quietly bumps numpy off the pin above (2.4.4 -> 2.4.6),
# defeating the parity the pins exist for.

# Standalone DINOv3 extractor: byte-identical preprocessing to the local runs,
# without importing alfspy.embedding (which pulls in the ModernGL render path).
COPY dino_extractor.py /opt/bambi/dino_extractor.py
ENV PYTHONPATH=/opt/bambi:/scratch/bambi/alfs_embed/code:/scratch/bambi/alfs_embed/code/src:/scratch/bambi/alfs_embed/code/alfspy_src

WORKDIR /scratch/bambi/alfs_embed/code
# has_embree must be True: without embreex trimesh silently falls back to a
# pure-numpy ray caster, which is correct but far too slow for full renders.
CMD ["python", "-c", "import torch, transformers, ultralytics, cv2, sklearn, trimesh; \
from trimesh.ray import has_embree; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus'); \
print('transformers', transformers.__version__); print('ultralytics', ultralytics.__version__); \
print('cv2', cv2.__version__); print('trimesh', trimesh.__version__, 'embree:', has_embree)"]
