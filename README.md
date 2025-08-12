#!/usr/bin/env python3
import os, argparse, random, shutil, math
import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.covariance import LedoitWolf

from ultralytics import YOLO

# ----------------------------
# Globals
# ----------------------------
IMG_SIZE = 224
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
FEATURES = {}
LAYER_NAMES = ['layer2', 'layer3', 'layer4']

# ----------------------------
# Utility: list images
# ----------------------------
def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted([os.path.join(d, f) for f in os.listdir(d)
                   if f.lower().endswith(VALID_EXT)])

# ----------------------------
# Device helper
# ----------------------------
def get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

# ----------------------------
# Feature hook for PaDiM
# ----------------------------
def get_hook(name):
    def hook(module, input, output):
        FEATURES[name] = output.detach()
    return hook

def build_backbone(device):
    try:
        weights = models.ResNet101_Weights.DEFAULT
        net = models.resnet101(weights=weights).to(device).eval()
    except Exception:
        net = models.resnet101(pretrained=True).to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    for name, module in net.named_children():
        if name in LAYER_NAMES:
            module.register_forward_hook(get_hook(name))
    return net

# ----------------------------
# Transforms
# ----------------------------
def build_transforms(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

# ----------------------------
# MRZ Detection (YOLO-OBB) Stub
# ----------------------------

# --- YOLO-OBB MRZ Loader + Detector (Ultralytics) ---
# pip install ultralytics==8.*  (or the YOLO 11 release if you use it)

# If your crop function uses cv2.getRotationMatrix2D(center, angle, ...)
# and you want to align the rotated rect horizontally, you may need
# to rotate by -angle instead of +angle. Adjust this if crops look skewed.
_model = None
_names = None

ROTATE_SIGN = -1  # adjust if crops look rotated incorrectly

def load_mrz_model(weights_path="obb-mrz/best.pt"):
    model = YOLO(weights_path)
    names = model.model.names if hasattr(model, "model") else model.names
    return model, names

def load_mrz_model(weights_path="obb-mrz/best.pt"):
    """
    Loads the Ultralytics YOLO OBB model.
    Returns (model, names) where names is id->class name dict.
    """
    model = YOLO(weights_path)
    names = model.model.names if hasattr(model, "model") else model.names
    return model, names

def _resolve_target_class_id(target_class, names):
    """
    target_class: int class id or str class name (e.g., "mrz")
    names: id->name dict from YOLO
    Returns class_id or None if not filtering.
    """
    if target_class is None:
        return None
    if isinstance(target_class, int):
        return target_class
    if isinstance(target_class, str):
        # find id by name (case-insensitive)
        for cid, cname in names.items():
            if cname.lower() == target_class.lower():
                return cid
        raise ValueError(f"Class '{target_class}' not found in model names: {names}")
    raise ValueError("target_class must be int, str, or None")

def _xywhr_to_tuple(xywhr_row):
    """
    xywhr_row: [cx, cy, w, h, r] with r in radians (ultralytics OBB)
    Returns (cx, cy, w, h, angle_deg)
    """
    cx, cy, w, h, r = xywhr_row.tolist()
    angle_deg = float(np.degrees(r))
    return float(cx), float(cy), float(w), float(h), float(angle_deg)

def _poly_to_minrect(poly8):
    """
    poly8: [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns (cx, cy, w, h, angle_deg) using cv2.minAreaRect
    Angle normalized to (-90, 90] degrees, CCW positive.
    """
    pts = np.array(poly8, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    # OpenCV angle in (-90, 0]; if w < h, swap to keep angle stable
    if w < h:
        w, h = h, w
        angle = angle + 90.0
    # Normalize to (-90, 90]
    if angle <= -90:
        angle += 180.0
    if angle > 90:
        angle -= 180.0
    return float(cx), float(cy), float(w), float(h), float(angle)

@torch.no_grad()
def detect_mrz_obb_ultra(
    image_bgr,
    model,
    names=None,
    device="cpu",
    imgsz=1024,
    conf=0.25,
    iou=0.5,
    target_class=None,  # int id or str name, or None for all
    max_det=50
):
    """
    Runs OBB detection and returns list of (cx, cy, w, h, angle_deg, conf, cls_id).
    All geometry in pixel units, angle in degrees (CCW).
    """
    if names is None:
        names = model.model.names if hasattr(model, "model") else model.names

    class_id = _resolve_target_class_id(target_class, names)

    # Ultralytics can accept numpy arrays; ensure RGB order
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=image_rgb,
        task="obb",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=0 if (device == "auto" and torch.cuda.is_available()) else device,
        verbose=False
    )

    if not results:
        return []

    res = results[0]
    out = []

    # Prefer xywhr if available
    xywhr = getattr(getattr(res, "obb", None), "xywhr", None)
    cls_arr = getattr(getattr(res, "obb", None), "cls", None)
    conf_arr = getattr(getattr(res, "obb", None), "conf", None)

    if xywhr is not None:
        xywhr_np = xywhr.cpu().numpy()
        cls_np = cls_arr.cpu().numpy() if cls_arr is not None else np.zeros(len(xywhr_np))
        conf_np = conf_arr.cpu().numpy() if conf_arr is not None else np.ones(len(xywhr_np))
        for i in range(xywhr_np.shape[0]):
            cx, cy, w, h, angle_deg = _xywhr_to_tuple(xywhr_np[i])
            cid = int(cls_np[i])
            if (class_id is not None) and (cid != class_id):
                continue
            out.append((cx, cy, w, h, angle_deg, float(conf_np[i]), cid))

    else:
        # Fallback: reconstruct from polygons
        polys = getattr(getattr(res, "obb", None), "xyxyxyxy", None)
        cls_arr = getattr(getattr(res, "obb", None), "cls", None)
        conf_arr = getattr(getattr(res, "obb", None), "conf", None)
        if polys is None:
            return out
        polys_np = polys.cpu().numpy()
        cls_np = cls_arr.cpu().numpy() if cls_arr is not None else np.zeros(len(polys_np))
        conf_np = conf_arr.cpu().numpy() if conf_arr is not None else np.ones(len(polys_np))
        for i in range(polys_np.shape[0]):
            cx, cy, w, h, angle_deg = _poly_to_minrect(polys_np[i])
            cid = int(cls_np[i])
            if (class_id is not None) and (cid != class_id):
                continue
            out.append((cx, cy, w, h, angle_deg, float(conf_np[i]), cid))

    return out

# --- Example usage ---
# model, names = load_mrz_model("obb-mrz/best.pt")
# detections = detect_mrz_obb_ultra(bgr_image, model, names, target_class="mrz", conf=0.25, iou=0.5)
# for cx, cy, w, h, angle_deg, conf, cid in detections:
#     print(cx, cy, w, h, angle_deg, conf, names[cid])

def detect_mrz_obb(image_bgr):
    """
    Runs actual YOLO‑OBB MRZ detection using obb-mrz/best.pt.
    Returns list of (cx, cy, w, h, angle_deg) in pixel units.
    """
    global _model, _names
    if _model is None:
        _model, _names = load_mrz_model("obb-mrz/best.pt")

    detections = detect_mrz_obb_ultra(
        image_bgr,
        _model,
        names=_names,
        target_class="mrz_line",  # or None for all classes
        conf=0.25,
        iou=0.5
    )
    # print(detections)

    # Strip to the format you want (drop conf, class id)
    return [(cx, cy, w, h, angle_deg) for cx, cy, w, h, angle_deg, conf, cid in detections]

# ----------------------------
# Crop rotated OBB
# ----------------------------
def crop_obb_region(image_bgr, obb):
    cx, cy, bw, bh, angle_deg = obb[:5]
    M = cv2.getRotationMatrix2D((cx, cy), ROTATE_SIGN * angle_deg, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (image_bgr.shape[1], image_bgr.shape[0]))
    x0, y0 = int(cx - bw/2), int(cy - bh/2)
    x1, y1 = int(cx + bw/2), int(cy + bh/2)
    return rotated[max(0,y0):max(0,y1), max(0,x0):max(0,x1)]



# ----------------------------
# Edge Detection Preprocessing
# ----------------------------

def canny_edge_preprocess(image_bgr):
    """
    Edge detection using Gaussian blur + Canny + dilation + close morphology.
    Returns a 3-channel BGR version of the edge mask.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Light smoothing to reduce noise while preserving edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Dilate to strengthen and connect edges
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Morphological closing to fill small gaps in edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Convert to 3-channel BGR for consistent downstream usage
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr
# ----------------------------
# ELA Preprocessing
# ----------------------------
def ela_preprocess(image_bgr, quality=30, scale=10, dynamic=False):
    """
    Error Level Analysis (ELA) preprocessing.
    - Re-compresses the image at given JPEG quality.
    - Computes absolute difference (residual) between original and JPEG.
    - Scales residual to enhance visibility.
    
    Args:
        image_bgr (np.ndarray): uint8 BGR image.
        quality (int): JPEG quality for re-compression (typ. 85–95).
        scale (float): Fixed amplification factor if dynamic=False.
        dynamic (bool): If True, dynamically stretches residual to full 0–255.

    Returns:
        np.ndarray: uint8 BGR ELA map.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to ela_preprocess")

    # Ensure 3-channel BGR uint8
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    # JPEG re-compression
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    ok, enc = cv2.imencode(".jpg", image_bgr, encode_params)
    if not ok:
        raise RuntimeError("JPEG encoding failed in ela_preprocess")
    jpeg_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Residual (differences in compression error)
    diff = cv2.absdiff(image_bgr, jpeg_bgr).astype(np.float32)

    if dynamic:
        # Stretch to full dynamic range per-image
        max_val = diff.max()
        if max_val < 1e-6:
            ela = np.zeros_like(diff, dtype=np.uint8)
        else:
            ela = np.clip(255.0 * diff / max_val, 0, 255).astype(np.uint8)
    else:
        # Fixed amplification (stable across dataset)
        ela = np.clip(diff * float(scale), 0, 255).astype(np.uint8)

    return ela

# ----------------------------
# PaDiM Feature Extraction
# ----------------------------
@torch.no_grad()
def extract_concat_feats(model, device, img_pil, tf):
    FEATURES.clear()
    x = tf(img_pil).unsqueeze(0).to(device)
    _ = model(x)
    f2 = FEATURES['layer2']
    _, _, H2, W2 = f2.shape
    f3 = FEATURES['layer3']
    f4 = FEATURES['layer4']
    f3_up = F.interpolate(f3, size=(H2, W2), mode='bilinear', align_corners=False)
    f4_up = F.interpolate(f4, size=(H2, W2), mode='bilinear', align_corners=False)
    feat = torch.cat([f2, f3_up, f4_up], dim=1).squeeze(0)
    feat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
    return feat.cpu().numpy(), (H2, W2)

# ----------------------------
# TRAIN
# ----------------------------
def train(args):
    device = get_device(args.device)
    model = build_backbone(device)
    
    # Original train/inference transforms
    TF_TRN = build_transforms(train=True)
    TF_INF = build_transforms(train=False)

    # Extra rotation augment (±15 degrees for example)
    ROT_AUG = transforms.RandomRotation(degrees=(-15, 15))

    train_files = list_images(args.train_dir)
    if not train_files:
        return

    # Process 1 sample for shape
    sample_bgr = cv2.imread(train_files[0])
    obb_list = detect_mrz_obb(sample_bgr)
    crop = canny_edge_preprocess(crop_obb_region(sample_bgr, obb_list[0]))
    sample_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    base_feat, (H, W) = extract_concat_feats(model, device, sample_img, TF_INF)
    P, Ctot = base_feat.shape
    rng = np.random.RandomState(args.seed)
    ch_idx = rng.choice(Ctot, size=args.D, replace=False)

    X_list = []
    for path in tqdm(train_files, desc="Extract feats (train)"):
        bgr = cv2.imread(path)
        for obb in detect_mrz_obb(bgr):
            crop = canny_edge_preprocess(crop_obb_region(bgr, obb))
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # Original (no aug)
            feats0, _ = extract_concat_feats(model, device, img_pil, TF_INF)
            X_list.append(feats0[:, ch_idx])

            # Augmentations
            for _ in range(args.aug_times):
                # Apply rotation BEFORE your other training transforms
                aug_img = ROT_AUG(img_pil)
                
                FEATURES.clear()
                x = TF_TRN(aug_img).unsqueeze(0).to(device)
                _ = model(x)

                f2 = FEATURES['layer2']
                _, _, H2, W2 = f2.shape
                f3 = FEATURES['layer3']
                f4 = FEATURES['layer4']
                f3_up = F.interpolate(f3, size=(H2, W2), mode='bilinear', align_corners=False)
                f4_up = F.interpolate(f4, size=(H2, W2), mode='bilinear', align_corners=False)

                feat = torch.cat([f2, f3_up, f4_up], dim=1).squeeze(0)
                feat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0]).cpu().numpy()
                X_list.append(feat[:, ch_idx])

    X = np.stack(X_list, axis=0)
    mus = np.empty((P, args.D), dtype=np.float32)
    cov_invs = np.empty((P, args.D, args.D), dtype=np.float32)
    for p in tqdm(range(P), desc="Fit Gaussians per location"):
        Xp = X[:, p, :]
        lw = LedoitWolf().fit(Xp)
        mus[p] = lw.location_.astype(np.float32)
        cov_invs[p] = np.linalg.inv(lw.covariance_).astype(np.float32)

    # Threshold calibration
    top_k = max(10, int(0.02 * P))

    def score_image(feats_PD):
        delta = feats_PD - mus
        d2 = np.einsum('pd,pde,pd->p', delta, cov_invs, delta)
        d = np.sqrt(np.clip(d2, 0, None))
        k = min(top_k, d.size)
        return float(np.mean(np.partition(d, -k)[-k:]))

    train_scores = []
    for path in tqdm(train_files, desc="Calibrate on train"):
        bgr = cv2.imread(path)
        for obb in detect_mrz_obb(bgr):
            crop = canny_edge_preprocess(crop_obb_region(bgr, obb))
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            feats, _ = extract_concat_feats(model, device, img_pil, TF_INF)
            train_scores.append(score_image(feats[:, ch_idx]))

    thr = float(np.quantile(np.array(train_scores), 1.0 - args.target_fpr))
    np.savez_compressed(
        args.out,
        mus=mus, cov_invs=cov_invs,
        H=H, W=W, D=args.D, ch_idx=ch_idx,
        thresh=thr, top_k=top_k
    )
    print(f"[TRAIN] Saved model to {args.out}  Threshold={thr:.6f}")

@torch.no_grad()
def infer(args):
    device = get_device(args.device)
    blob = np.load(args.model)
    mus = blob['mus']; cov_invs = blob['cov_invs']
    H = int(blob['H']); W = int(blob['W'])
    D = int(blob['D']); ch_idx = blob['ch_idx']
    thr = float(blob['thresh']); top_k = int(blob['top_k'])

    model = build_backbone(device)
    TF_INF = build_transforms(train=False)

    test_files = list_images(args.test_dir)
    out_norm = os.path.join(args.out_dir, "normal")
    out_anom = os.path.join(args.out_dir, "anomaly")
    os.makedirs(out_norm, exist_ok=True)
    os.makedirs(out_anom, exist_ok=True)

    for path in test_files:
        bgr = cv2.imread(path)
        img_score_total = 0
        label = 'normal'
        for obb in detect_mrz_obb(bgr):
            crop = canny_edge_preprocess(crop_obb_region(bgr, obb))
            # crop = crop_obb_region(bgr, obb)
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            feats, (H2, W2) = extract_concat_feats(model, device, img_pil, TF_INF)
            if (H2, W2) != (H, W):
                print(f"[ERROR] Grid mismatch for {os.path.basename(path)}")
                continue
            feats = feats[:, ch_idx]
            delta = feats - mus
            d2 = np.einsum('pd,pde,pd->p', delta, cov_invs, delta)
            d = np.sqrt(np.clip(d2, 0, None))
            k = min(top_k, d.size)
            score = float(np.mean(np.partition(d, -k)[-k:]))
            img_score_total = max(img_score_total, score)
            if score > thr:
                label = 'anomaly'
        print(f"{os.path.basename(path)} -> {img_score_total:.6f} [{label}]")
        dst = out_anom if label == 'anomaly' else out_norm
        shutil.copy(path, dst)

def parse_args():
    p = argparse.ArgumentParser("OBB + Sobel + PaDiM Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--train_dir", required=True)
    pt.add_argument("--out", default="padim_model_r101.npz")
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--D", type=int, default=100)
    pt.add_argument("--aug_times", type=int, default=20)
    pt.add_argument("--target_fpr", type=float, default=0.01)
    pt.add_argument("--seed", type=int, default=0)

    pi = sub.add_parser("infer")
    pi.add_argument("--model", required=True)
    pi.add_argument("--test_dir", required=True)
    pi.add_argument("--device", default="cpu")
    pi.add_argument("--out_dir", default="infer_outputs")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
