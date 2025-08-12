"""
padim_inference.py
------------------

Inference‑only implementation of PaDiM (Patch Distribution Modeling) for image
anomaly detection. It can read the model saved by the training script (new
``.npz`` format **or** the legacy ``.npz``/``.pkl`` format) and returns a
Mahalanobis‑based anomaly score.

Typical usage
~~~~~~~~~~~~~

>>> from padim_inference import detect_anomaly
>>> result = detect_anomaly(
...     image_path="samples/img.png",
...     model_path="models/padim_trained.npz"
... )
>>> print(result)
{'image_path': 'samples/img.png',
 'anomaly_score': 12.34,
 'classification': 'anomaly',
 'is_anomaly': True}
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
from typing import Tuple, Optional, Dict

import joblib
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# --------------------------------------------------------------------------- #
# Helper – register forward hooks to capture intermediate feature maps
# --------------------------------------------------------------------------- #
def _register_feature_hooks(
    model: torch.nn.Module, layer_names: list, features_dict: dict
) -> None:
    """Register forward hooks that store the output of *layer_names* in *features_dict*."""
    def _make_hook(name: str):
        def _hook(module, inp, out):
            features_dict[name] = out.detach().cpu()
        return _hook

    for name, module in model.named_children():
        if name in layer_names:
            module.register_forward_hook(_make_hook(name))

# -----------------------------------------------------------------
# Bottom‑percentage crop
# -----------------------------------------------------------------
def crop_bottom_percent_pil(image: Image.Image, percent: float = 0.35) -> Image.Image:
    if not (0.0 < percent <= 1.0):
        raise ValueError("percent must be in (0, 1]")
    w, h = image.size
    top = int(round(h * (1.0 - percent)))
    return image.crop((0, top, w, h))

def ensure_rgb(img: Image.Image) -> Image.Image:
    # If the image is mode "L" after edges, expand to RGB for 3-channel normalization
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

# -----------------------------------------------------------------
# Pre‑processing helpers
# -----------------------------------------------------------------
def canny_edge_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Canny → dilation → closing → convert back to 3‑channel BGR."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# PaDiM Detector – inference only
# --------------------------------------------------------------------------- #
class PaDiMDetector:
    """
    PaDiM anomaly detector (inference‑only).

    The detector can load:
      * the *new* `.npz` format produced by the training script, **or**
      * the *legacy* PaDiM format (``mus``, ``cov_invs`` …).

    After loading, ``predict()`` returns the maximum Mahalanobis distance over
    all image patches and a label (``"normal"`` / ``"anomaly"``) based on the
    stored threshold.
    """

    # ------------------------------------------------------------------- #
    # Construction
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        layers: list = ["layer1", "layer2", "layer3"],
        pca_components: int = 64,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        layers : list
            Names of ResNet‑18 layers to be used as feature sources.
        pca_components : int
            Number of PCA components (must match the value used during training).
        device : torch.device | None
            Execution device – defaults to CUDA if available.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.layers = layers
        self.pca_components = pca_components

        # ----------------------------------------------------------------
        # 1️⃣ Load the ResNet‑18 backbone (pre‑trained ImageNet weights)
        # ----------------------------------------------------------------
        try:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:          # fallback for older torchvision versions
            self.model = models.resnet18(weights=None)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ----------------------------------------------------------------
        # 2️⃣ Hook the requested layers
        # ----------------------------------------------------------------
        self.features: dict = {}
        _register_feature_hooks(self.model, self.layers, self.features)

        # ----------------------------------------------------------------
        # 3️⃣ Image preprocessing (same as training)
        # ----------------------------------------------------------------
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # ----------------------------------------------------------------
        # 4️⃣ Place‑holders for learned parameters (filled by ``load``)
        # ----------------------------------------------------------------
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None
        self.covariance_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

        # legacy flag – set automatically inside ``load`` if needed
        self.use_legacy_format: bool = False

    # ------------------------------------------------------------------- #
    # Private helpers
    # ------------------------------------------------------------------- #
    def _extract_patch_features(self, image_path: str) -> np.ndarray:
        """
        Extract multi‑scale patch features for *image_path*.

        Returns
        -------
        np.ndarray
            Shape ``[num_patches, feature_dim]`` – one row per spatial patch.
        """
        # 1️⃣ Load & preprocess
        img_bgr = cv2.imread(image_path)  # shape [H,W,3], BGR, uint8
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        img_bgr = crop_bottom_percent_pil(img_bgr, percent=0.35)
        img_bgr = canny_edge_preprocess(img_bgr)  # ensure it returns uint8

        # If edges are 1-channel, convert to 3-channel RGB
        if img_bgr.ndim == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)

        # 2) To tensor
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 3) Forward (hooks capture features)
        self.features.clear()  # optional but safer between calls
        with torch.no_grad():
            _ = self.model(tensor)

        # 4) Align spatial sizes
        ref_h, ref_w = self.features[self.layers[-1]].shape[2:]
        collected = []
        for name in self.layers:
            fm = self.features[name]
            if fm.dim() != 4:
                raise ValueError(f"Expected 4D feature map for {name}, got {fm.shape}")
            fm = fm.squeeze(0)  # [C,H,W]
            if (fm.shape[1], fm.shape[2]) != (ref_h, ref_w):
                fm = F.interpolate(
                    fm.unsqueeze(0), size=(ref_h, ref_w),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            collected.append(fm)

        # 5) Concatenate and flatten to patch vectors [N, C_total]
        cat = torch.cat(collected, dim=0)               # [C_total, H, W]
        patches = cat.reshape(cat.shape[0], -1).T.cpu().numpy()  # [H*W, C_total]
        return patches

    # ------------------------------------------------------------------- #
    # Model loading
    # ------------------------------------------------------------------- #
    def load(self, filepath: str) -> None:
        """
        Load a model exported by the training script.

        Supported formats:
          * ``*.pkl`` – joblib dump containing ``pca``, ``mean_vector``,
            ``covariance_inv`` and ``threshold``.
          * ``*.npz`` – either the *new* PaDiM format (contains PCA) or the
            *legacy* format (``mus``, ``cov_invs`` …).

        Raises
        ------
        FileNotFoundError
            If ``filepath`` does not exist.
        ValueError
            If the file format is not recognised or required keys are missing.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        # ----------------------------------------------------------------
        # .pkl – straightforward joblib load
        # ----------------------------------------------------------------
        if ext == ".pkl":
            data = joblib.load(filepath)
            self.pca = data["pca"]
            self.mean_vector = data["mean_vector"]
            self.covariance_inv = data["covariance_inv"]
            self.threshold = data["threshold"]
            self.use_legacy_format = False
            return

        # ----------------------------------------------------------------
        # .npz – could be *new* format (with PCA) or *legacy* format
        # ----------------------------------------------------------------
        if ext != ".npz":
            raise ValueError(
                "Unsupported model format. Expected *.pkl or *.npz, got "
                f"{ext}"
            )
        data = np.load(filepath, allow_pickle=True)

        # ----- New (refactored) format -----
        if "pca_components" in data:
            n_comp = data["pca_components"].shape[0]
            self.pca = PCA(n_components=n_comp)
            self.pca.components_ = data["pca_components"]
            self.pca.mean_ = data["pca_mean"]
            self.mean_vector = data["mean_vector"]
            self.covariance_inv = data["covariance_inv"]
            self.threshold = float(data["threshold"])
            self.use_legacy_format = False
            return

        # ----- Legacy format -----
        required_keys = {"mus", "cov_invs", "thresh", "H", "W", "D"}
        if not required_keys.issubset(set(data.keys())):
            missing = required_keys - set(data.keys())
            raise ValueError(
                f"Legacy model missing required keys: {missing}. "
                "Check that you are loading a proper PaDiM legacy file."
            )
        self.legacy_mus = data["mus"]                # (H*W, D)
        self.legacy_cov_invs = data["cov_invs"]      # (H*W, D, D)
        self.threshold = float(data["thresh"])
        self.legacy_H = int(data["H"])
        self.legacy_W = int(data["W"])
        self.legacy_D = int(data["D"])

        # optional channel‑selection vector
        self.legacy_ch_idx = data["ch_idx"] if "ch_idx" in data else np.arange(self.legacy_D)

        self.use_legacy_format = True

    # ------------------------------------------------------------------- #
    # Legacy‑format inference (kept for backward compatibility)
    # ------------------------------------------------------------------- #
    def _predict_legacy(self, image_path: str) -> Tuple[float, str]:
        """
        Predict an anomaly score using the legacy model format.
        """
        # 1) Load & resize exactly as original training
        img = Image.open(image_path).convert("RGB")
        img = crop_bottom_percent_pil(img, percent=0.35)

        # canny_edge_preprocess must accept/return PIL or adapt here
        img = canny_edge_preprocess(img)  # ensure it returns PIL.Image
        img = ensure_rgb(img)             # keep 3 channels for ImageNet stats

        img = img.resize((224, 224), Image.LANCZOS)

        # 2) Transform → tensor (ImageNet normalization)
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = trans(img).unsqueeze(0).to(self.device)

        # 3) Forward pass (hooks fill self.features)
        if hasattr(self, "features"):
            self.features.clear()
        with torch.no_grad():
            _ = self.model(tensor)

        # 4) Gather feature maps resized to the legacy spatial size
        target_hw = (self.legacy_H, self.legacy_W)  # (H, W)
        feature_maps = []
        for name in self.layers:
            if name not in self.features:
                continue
            fm = self.features[name]
            if fm.dim() != 4:
                raise ValueError(f"Feature {name} must be [N,C,H,W], got {fm.shape}")
            fm = fm.squeeze(0)  # [C, H, W]
            h, w = fm.shape[1], fm.shape[2]
            if (h, w) != target_hw:
                fm = F.interpolate(
                    fm.unsqueeze(0),  # [1,C,H,W]
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)        # [C, Ht, Wt]
            feature_maps.append(fm)

        if not feature_maps:
            return 0.0, "normal"

        # 5) Concatenate → flatten to patches
        cat = torch.cat(feature_maps, dim=0)                  # [C_total, Ht, Wt]
        patches = cat.reshape(cat.shape[0], -1).T.cpu().numpy()  # [N=Ht*Wt, C_total]

        # 6) Adjust dimensions to match the legacy model
        cur_D = patches.shape[1]
        if cur_D != self.legacy_D:
            if cur_D < self.legacy_D:
                repeat = self.legacy_D // cur_D
                remainder = self.legacy_D % cur_D
                blocks = [np.tile(patches, (1, repeat))]
                if remainder > 0:
                    blocks.append(patches[:, :remainder])
                patches = np.hstack(blocks)
            else:
                patches = patches[:, : self.legacy_D]

        cur_N = patches.shape[0]
        expected_N = self.legacy_H * self.legacy_W
        if cur_N != expected_N:
            if cur_N < expected_N:
                idx = np.tile(np.arange(cur_N), (expected_N // cur_N + 1))[:expected_N]
                patches = patches[idx]
            else:
                idx = np.linspace(0, cur_N - 1, expected_N, dtype=int)
                patches = patches[idx]

        # 7) Compute Mahalanobis distance per location
        distances = []
        D = patches.shape[1]

        # Validate legacy stats dimensions once
        if len(self.legacy_mus) != expected_N or len(self.legacy_cov_invs) != expected_N:
            raise ValueError(
                f"Legacy stats mismatch: expected {expected_N} locations, "
                f"got mus={len(self.legacy_mus)}, cov_invs={len(self.legacy_cov_invs)}"
            )

        for i in range(expected_N):
            patch = patches[i]
            mu = self.legacy_mus[i]            # shape (D,) expected
            cov_inv = self.legacy_cov_invs[i]  # shape (D,D) expected

            # Align dimensions defensively
            if mu.shape[0] != D:
                if mu.shape[0] < D:
                    patch = patch[: mu.shape[0]]
                    D_i = mu.shape[0]
                else:
                    mu = mu[:D]
                    cov_inv = cov_inv[:D, :D]
                    D_i = D
            else:
                D_i = D

            # Final safety for cov_inv
            if cov_inv.shape != (D_i, D_i):
                cov_inv = cov_inv[:D_i, :D_i]

            diff = patch[:D_i] - mu[:D_i]
            # Numerically stable Mahalanobis with abs guard
            try:
                val = float(diff.T @ cov_inv @ diff)
                dist = np.sqrt(val) if val >= 0 else np.sqrt(np.abs(val))
            except np.linalg.LinAlgError:
                dist = float(np.linalg.norm(diff))
            distances.append(dist)

        anomaly_score = float(np.max(distances))
        label = "anomaly" if anomaly_score > self.threshold else "normal"
        return anomaly_score, label

    # ------------------------------------------------------------------- #
    # Public inference API
    # ------------------------------------------------------------------- #
    def predict(self, image_path: str) -> Tuple[float, str]:
        """
        Compute the anomaly score for *image_path*.

        Returns
        -------
        tuple (score, label)
            *score* – maximal Mahalanobis distance over all patches.
            *label* – ``"anomaly"`` if ``score > threshold`` else ``"normal"``.
        """
        if self.use_legacy_format:
            return self._predict_legacy(image_path)

        if self.pca is None:
            raise RuntimeError(
                "Model parameters not loaded. Call `load()` before `predict()`."
            )

        # 1️⃣ Extract raw multi‑scale patches
        patches = self._extract_patch_features(image_path)

        # 2️⃣ Reduce dimensionality with the stored PCA matrix
        reduced = self.pca.transform(patches)          # (N, pca_components)

        # 3️⃣ Mahalanobis distance per patch
        dists = np.array(
            [
                mahalanobis(p, self.mean_vector, self.covariance_inv)
                for p in reduced
            ]
        )

        # 4️⃣ Final score & label
        anomaly_score = float(dists.max())
        label = "anomaly" if anomaly_score > self.threshold else "normal"
        return anomaly_score, label


# --------------------------------------------------------------------------- #
# Convenience wrapper – one‑liner for typical usage
# --------------------------------------------------------------------------- #
def detect_anomaly(image_path: str, model_path: str) -> Dict[str, object]:
    """
    Load a PaDiM model and return a dictionary with the detection result.

    Parameters
    ----------
    image_path : str
        Path to the image to be inspected.
    model_path : str
        Path to the ``.npz`` or ``.pkl`` model file produced by the training script.

    Returns
    -------
    dict
        ``{
            "image_path": <str>,
            "anomaly_score": <float>,
            "classification": <"normal" | "anomaly">,
            "is_anomaly": <bool>
        }```
    """
    detector = PaDiMDetector()
    detector.load(model_path)
    score, label = detector.predict(image_path)

    return {
        "image_path": str(image_path),
        "anomaly_score": score,
        "classification": label,
        "is_anomaly": label == "anomaly",
    }
