#!/usr/bin/env python3
"""
padim.py

PaDiM anomaly detection (inference-only).

- Supports NEW dumps (.npz or .pkl) with: pca_components, pca_mean, mean_vector,
  covariance_inv, threshold
- Supports LEGACY dumps (.npz) with: mus, cov_invs, H, W, D, (optional ch_idx), thresh
- Optional preprocessing: bottom-crop percentage + Canny edge emphasis

Requires:
  pip install torch torchvision pillow numpy scikit-learn joblib opencv-python
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import joblib

# Optional OpenCV for Canny/overlays
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# -------------------- helpers --------------------

def _register_feature_hooks(model, layer_names: List[str], features_dict: Dict[str, torch.Tensor]) -> None:
    def _mk(name):
        def hook(_m, _i, out):
            features_dict[name] = out.detach().cpu()
        return hook
    for name, module in model.named_children():
        if name in layer_names:
            module.register_forward_hook(_mk(name))

def _to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])(img)

def _bottom_crop_pil(img: Image.Image, percent: float) -> Image.Image:
    if percent is None or percent <= 0 or percent > 1:
        return img
    w, h = img.size
    top = int(round(h * (1.0 - percent)))
    return img.crop((0, top, w, h))

def _canny_rgb(img_rgb: Image.Image) -> Image.Image:
    """Run Canny in OpenCV space then return back to PIL RGB."""
    if not _HAS_CV2:
        return img_rgb
    bgr = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bgr_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr_edges, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + eps)

def _resize_map01(map_hw: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    w, h = target_wh
    if _HAS_CV2:
        return cv2.resize(map_hw, (w, h), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray((np.clip(map_hw, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.BICUBIC)
    return np.asarray(pil).astype(np.float32) / 255.0

def save_heatmap(heat01: np.ndarray, out_path: Union[str, Path]) -> None:
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    h8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    if _HAS_CV2:
        cv2.imwrite(str(out), cv2.applyColorMap(h8, cv2.COLORMAP_JET))
    else:
        Image.fromarray(h8).save(out)

def save_overlay(rgb: np.ndarray, heat01: np.ndarray, out_path: Union[str, Path], alpha: float = 0.35) -> None:
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    h8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    if _HAS_CV2:
        color = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
        over = cv2.addWeighted(color, alpha, rgb[:, :, ::-1], 1.0 - alpha, 0)
        cv2.imwrite(str(out), over)
    else:
        heat_rgb = np.stack([h8, h8, h8], axis=-1)
        over = (alpha * heat_rgb + (1 - alpha) * rgb).astype(np.uint8)
        Image.fromarray(over).save(out)


# -------------------- model --------------------

@dataclass
class _LegacyParams:
    mus: Optional[np.ndarray] = None
    cov_invs: Optional[np.ndarray] = None
    H: Optional[int] = None
    W: Optional[int] = None
    D: Optional[int] = None
    ch_idx: Optional[np.ndarray] = None


class PaDiMDetector:
    """
    PaDiM anomaly detector (ResNet18 backbone).
    Matches the model formats you outlined in the repo.  [oai_citation:1‡GitHub](https://github.com/songhieng/ml-template/tree/29ecb9a32b4a931011dc4cbb4fca6f85d54155e5)
    """
    def __init__(
        self,
        layers: List[str] = ["layer1", "layer2", "layer3"],
        device: Optional[torch.device] = None,
        bottom_crop: Optional[float] = 0.35,
        use_canny: bool = True,
    ):
        self.layers = list(layers)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bottom_crop = bottom_crop
        self.use_canny = use_canny

        try:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.model = models.resnet18(weights=None)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.features: Dict[str, torch.Tensor] = {}
        _register_feature_hooks(self.model, self.layers, self.features)

        # NEW
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None
        self.covariance_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

        # LEGACY
        self._legacy = _LegacyParams()
        self._use_legacy = False

    # -------- IO --------

    def load(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        ext = p.suffix.lower()

        if ext == ".pkl":
            data = joblib.load(p)
            self.pca = data["pca"]
            self.mean_vector = data["mean_vector"]
            self.covariance_inv = data["covariance_inv"]
            self.threshold = float(data["threshold"])
            self._use_legacy = False
            return

        if ext != ".npz":
            raise ValueError(f"Unsupported model format: {ext} (use .npz or .pkl)")

        data = np.load(p, allow_pickle=True)

        # NEW format
        if {"pca_components", "pca_mean", "mean_vector", "covariance_inv", "threshold"} <= set(data.files):
            d = data["pca_components"].shape[0]
            pca = PCA(n_components=d)
            pca.components_ = data["pca_components"]
            pca.mean_ = data["pca_mean"]
            self.pca = pca
            self.mean_vector = data["mean_vector"]
            self.covariance_inv = data["covariance_inv"]
            self.threshold = float(data["threshold"])
            self._use_legacy = False
            return

        # LEGACY format
        if {"mus", "cov_invs", "H", "W", "D", "thresh"} <= set(data.files):
            L = self._legacy
            L.mus = data["mus"]
            L.cov_invs = data["cov_invs"]
            L.H = int(data["H"]); L.W = int(data["W"]); L.D = int(data["D"])
            self.threshold = float(data["thresh"])
            L.ch_idx = data["ch_idx"] if "ch_idx" in data.files else None
            self._use_legacy = True
            return

        raise ValueError(f"Unknown model format. Keys={list(data.files)}")

    # -------- features --------

    def _extract_patch_features(self, img_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
        # PIL load first, then do optional bottom crop + canny (with proper type round-trips)
        img = Image.open(img_path).convert("RGB")
        if self.bottom_crop:
            img = _bottom_crop_pil(img, self.bottom_crop)
        if self.use_canny:
            img = _canny_rgb(img)

        tensor = _to_tensor(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(tensor)

        ref_h, ref_w = self.features[self.layers[-1]].shape[2:]
        collected = []
        for name in self.layers:
            fm = self.features[name]
            if fm.dim() != 4:
                raise ValueError(f"Expected [N,C,H,W] for {name}, got {fm.shape}")
            fm = fm.squeeze(0)
            if (fm.shape[1], fm.shape[2]) != (ref_h, ref_w):
                fm = F.interpolate(fm.unsqueeze(0), size=(ref_h, ref_w), mode="bilinear", align_corners=False).squeeze(0)
            collected.append(fm)

        cat = torch.cat(collected, dim=0)                      # [C_total, H, W]
        patches = cat.reshape(cat.shape[0], -1).T.cpu().numpy() # [H*W, C_total]
        return patches, (ref_h, ref_w)

    # -------- scoring --------

    def _score_new(self, patches: np.ndarray) -> np.ndarray:
        assert self.pca is not None and self.mean_vector is not None and self.covariance_inv is not None
        reduced = self.pca.transform(patches)  # (N, Dpca)
        return np.array([mahalanobis(v, self.mean_vector, self.covariance_inv) for v in reduced], dtype=np.float32)

    def _score_legacy(self, patches: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        L = self._legacy
        assert L.mus is not None and L.cov_invs is not None and L.H is not None and L.W is not None and L.D is not None
        Ht, Wt = target_hw
        expected = L.H * L.W
        # feature dim align
        curD = patches.shape[1]
        if curD != L.D:
            if curD < L.D:
                reps = L.D // curD
                rem = L.D % curD
                patches = np.tile(patches, (1, reps))
                if rem: patches = np.hstack([patches, patches[:, :rem]])
            else:
                patches = patches[:, :L.D]
        # spatial count align
        curN = patches.shape[0]
        if curN != expected:
            if curN < expected:
                idx = np.tile(np.arange(curN), (expected // curN + 1))[:expected]
                patches = patches[idx]
            else:
                idx = np.linspace(0, curN - 1, expected, dtype=int)
                patches = patches[idx]

        dists = np.empty(expected, dtype=np.float32)
        for i in range(expected):
            mu = L.mus[i]; cov_inv = L.cov_invs[i]
            D = min(len(mu), patches.shape[1], cov_inv.shape[0])
            diff = patches[i][:D] - mu[:D]
            try:
                val = float(diff.T @ cov_inv[:D, :D] @ diff)
                d = np.sqrt(val if val >= 0 else -val)
            except np.linalg.LinAlgError:
                d = float(np.linalg.norm(diff))
            dists[i] = d
        return dists.reshape(L.H, L.W).reshape(-1)  # flattened

    # -------- public --------

    def predict(
        self,
        image_path: Union[str, Path],
        return_map: bool = True,
    ) -> Dict[str, Union[str, float, bool, np.ndarray]]:
        patches, (fh, fw) = self._extract_patch_features(image_path)
        if self._use_legacy:
            dists = self._score_legacy(patches, (fh, fw))
            fmap = dists.reshape(self._legacy.H, self._legacy.W)
        else:
            dists = self._score_new(patches)
            fmap = dists.reshape(fh, fw)

        score = float(np.max(fmap))
        thr = float(self.threshold) if self.threshold is not None else 0.0
        cls = "anomaly" if score >= thr else "normal"

        out: Dict[str, Union[str, float, bool, np.ndarray]] = {
            "image_path": str(image_path),
            "anomaly_score": score,
            "classification": cls,
            "is_anomaly": (cls == "anomaly"),
        }
        if return_map:
            img = Image.open(image_path).convert("RGB")
            heat = _minmax(fmap)
            out["map"] = _resize_map01(heat, img.size)
        return out