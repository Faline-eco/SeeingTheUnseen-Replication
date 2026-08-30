from typing import Final

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoImageProcessor, AutoModel


_PATCH_SIZE: Final[int] = 16
# DINOv3 sequence layout: [CLS, reg0, reg1, reg2, reg3, patch0, ..., patchN]
# Skip 1 CLS token + 4 register tokens = 5 prefix tokens
_PREFIX_TOKENS: Final[int] = 5


class DinoV3Extractor:
    """
    Wraps the locally stored DINOv3 ViT-H+/16 model and extracts dense patch embeddings
    from BGR images.

    Patch embeddings are returned as a (H//16, W//16, 1280) float32 numpy array,
    where H and W are the (possibly resized) input image dimensions.
    """

    def __init__(self, model_dir: str):
        """
        :param model_dir: Path to the local HuggingFace model directory containing
            model.safetensors, config.json, and preprocessor_config.json.
        """
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._processor = AutoImageProcessor.from_pretrained(model_dir)
        self._model = AutoModel.from_pretrained(model_dir)
        self._model.eval()
        self._model.to(self._device)

    @property
    def embed_dim(self) -> int:
        return self._model.config.hidden_size  # 1280 for ViT-H+/16

    @property
    def patch_size(self) -> int:
        return _PATCH_SIZE

    def extract(self, img_bgr: NDArray) -> NDArray:
        """
        Extracts DINOv3 patch embeddings from a single BGR image (as loaded by OpenCV).

        The image is resized to the nearest dimensions that are multiples of the patch
        size (16) before being passed to DINOv3. The output spatial resolution therefore
        reflects the resized dimensions, not the original ones.

        :param img_bgr: Input image in BGR format as a (H, W, 3) or (H, W, 4) uint8 or
            float32 numpy array.
        :return: Patch embedding array of shape (H'//16, W'//16, embed_dim) float32,
            where H' and W' are the resized image dimensions.
        """
        img_rgb = self._prepare_image(img_bgr)
        h_r, w_r = img_rgb.shape[:2]

        inputs = self._processor(images=img_rgb, return_tensors='pt', do_resize=False)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # last_hidden_state: (1, 5 + H//16*W//16, embed_dim)
        patch_tokens = outputs.last_hidden_state[0, _PREFIX_TOKENS:, :]  # (N_patches, D)

        h_patches = h_r // _PATCH_SIZE
        w_patches = w_r // _PATCH_SIZE

        embedding = patch_tokens.reshape(h_patches, w_patches, self.embed_dim)
        return embedding.cpu().float().numpy()

    def extract_batch(self, imgs_bgr: list[NDArray]) -> list[NDArray]:
        """
        Extracts patch embeddings for a list of BGR images.

        Images are processed individually to handle varying sizes. For uniform-size
        batches, increase batch_size in the settings for better GPU utilisation.

        :param imgs_bgr: List of BGR images.
        :return: List of (H'//16, W'//16, embed_dim) float32 arrays, one per image.
        """
        return [self.extract(img) for img in imgs_bgr]

    @staticmethod
    def _prepare_image(img_bgr: NDArray) -> NDArray:
        """
        Converts BGR to RGB and resizes to the nearest dimensions that are multiples
        of the patch size.
        """
        # drop alpha if present
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[..., :3]

        # convert to uint8 if float
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        h_new = max(_PATCH_SIZE, round(h / _PATCH_SIZE) * _PATCH_SIZE)
        w_new = max(_PATCH_SIZE, round(w / _PATCH_SIZE) * _PATCH_SIZE)

        if h_new != h or w_new != w:
            img_rgb = cv2.resize(img_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        return img_rgb
