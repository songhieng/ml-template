Got you—here’s a full working FastAPI service that drops into your stack and adds:
	•	Background stabilization → defeats co-rotation.
	•	Residual head-motion gate (face vs. background) after stabilization.
	•	Face-3D geometry checks (FaceMesh): single-homography residual vs. 3D depth correlation + multi-patch homography dispersion.
	•	Clean fusion + richer diagnostics, while keeping your original routes (/detect, /detect_base64, /visualization, /cleanup, /health) and visualizations.

Paste this into app.py (or replace your current file).

✅ Dependencies: fastapi uvicorn numpy opencv-contrib-python scikit-image mediapipe aiofiles
(You already had most of these. opencv-contrib-python gives you SIFT.)

# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
from skimage.metrics import structural_similarity as ssim
import base64
import os
import uuid
from datetime import datetime
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bg-verify")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Background Verify API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (tune for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Config / Folders
# -----------------------------------------------------------------------------
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_WIDTH = 640  # resize for speed/stability

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Thread pool for CPU-bound work
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# For deterministic viz colors
np.random.seed(0)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class DetectionParams(BaseModel):
    # Existing knobs
    min_matches: int = Field(default=30, ge=1, le=2000, description="Minimum good matches for BG")
    num_matches: int = Field(default=100, ge=1, le=2000, description="Matches to draw in visualizations")
    ssim_thresh: float = Field(default=0.90, ge=0.0, le=1.0, description="SSIM threshold (higher means more similar)")
    flow_std_thresh: float = Field(default=1.2, ge=0.0, le=10.0, description="StdDev of optical flow magnitude on BG")
    match_ratio_thresh: float = Field(default=0.10, ge=0.0, le=1.0, description="min good/bg keypoint ratio")
    homography_inlier_thresh: float = Field(default=0.6, ge=0.0, le=1.0, description="BG planar inlier ratio")

    # New (backwards-compatible) knobs — normalized by inter-ocular distance where relevant
    min_stab_inliers: int = Field(default=30, ge=0, le=5000, description="Min inliers to accept BG stabilization")
    max_stab_err: float = Field(default=0.025, ge=0.0, le=1.0, description="Max normalized reprojection err after stabilization")
    mu_face_thresh_norm: float = Field(default=0.015, ge=0.0, le=1.0, description="Face residual flow threshold (norm by IO distance)")
    mu_bg_thresh_norm: float = Field(default=0.008, ge=0.0, le=1.0, description="BG residual flow threshold (norm by IO distance)")
    corrZ_thresh: float = Field(default=0.25, ge=-1.0, le=1.0, description="Corr(residual, |Z|) threshold for 3D")
    eH_face_thresh_norm: float = Field(default=0.012, ge=0.0, le=1.0, description="Face homography residual mean (normalized)")
    disp_face_thresh: float = Field(default=0.10, ge=0.0, le=10.0, description="Multi-homography dispersion threshold")

class Base64DetectionRequest(BaseModel):
    image1: str = Field(..., description="Base64 encoded first image (data URI ok)")
    image2: str = Field(..., description="Base64 encoded second image (data URI ok)")
    params: Optional[DetectionParams] = Field(default=DetectionParams(), description="Detection parameters")

class BackgroundAnalysis(BaseModel):
    total_matches: int
    match_ratio: float
    inlier_ratio: float
    planar_bg: bool
    inconsistent: bool

class ConsistencyMetrics(BaseModel):
    ssim: float
    flow_std: float
    ssim_threshold: float
    flow_threshold: float
    consistency_passed: bool

class FaceMetrics(BaseModel):
    inter_ocular: float
    mu_face_norm: float
    mu_bg_norm: float
    stabilized_inliers: int
    stabilized_error_norm: float
    head_moved: bool
    bg_static: bool
    corr_Z_resid: float
    eH_face_norm: float
    disp_face: float

class Visualizations(BaseModel):
    background_matches: str
    original_matches: str

class DetectionResults(BaseModel):
    is_live: bool
    is_spoof: bool
    background_analysis: BackgroundAnalysis
    consistency_metrics: ConsistencyMetrics
    face_metrics: Optional[FaceMetrics] = None
    visualizations: Visualizations
    session_id: str

class DetectionResponse(BaseModel):
    success: bool
    results: DetectionResults
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class CleanupResponse(BaseModel):
    success: bool
    removed_files: List[str]
    message: str

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def validate_file_extension(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(upload_file: UploadFile, file_path: str):
    content = await upload_file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

def _resize_max(img: np.ndarray, max_w: int = MAX_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _decode_b64(data: str) -> bytes:
    # Allow data URI prefix
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)

# -----------------------------------------------------------------------------
# Core detector
# -----------------------------------------------------------------------------
class FaceSpoofingDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

    # ---------- Face box ----------
    def detect_face_box(self, img: np.ndarray, min_confidence: float = 0.6) -> Optional[Tuple[int,int,int,int]]:
        with self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence
        ) as detector:
            res = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return None
        det = max(res.detections, key=lambda d: d.score[0]).location_data.relative_bounding_box
        h, w = img.shape[:2]
        x = int(max(det.xmin * w, 0))
        y = int(max(det.ymin * h, 0))
        ww = int(min(det.width * w, w - x))
        hh = int(min(det.height * h, h - y))
        # pad a little to be safe
        pad = int(0.05 * max(ww, hh))
        x = max(0, x - pad); y = max(0, y - pad)
        ww = min(w - x, ww + 2 * pad); hh = min(h - y, hh + 2 * pad)
        return (x, y, ww, hh)

    # ---------- FaceMesh landmarks + relative depth (|z|) ----------
    def facemesh_xyZ(self, img: np.ndarray):
        fm = self.mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fm.close()
        if not res.multi_face_landmarks:
            return None, None
        h, w = img.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x*w, p.y*h] for p in lm], np.float32)       # (N,2)
        Z   = np.array([abs(p.z) for p in lm], np.float32)             # (N,)
        return pts, Z

    # ---------- Inter-ocular distance from eye corners (indices 33 and 263) ----------
    @staticmethod
    def inter_ocular(pts: np.ndarray) -> float:
        try:
            pR = pts[33]; pL = pts[263]  # MediaPipe eye outer corners (commonly used)
            return float(np.linalg.norm(pL - pR))
        except Exception:
            return 0.0

    # ---------- Feature extraction (SIFT -> fallback ORB) ----------
    def _extract_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None):
        kp, d = None, None
        # Try SIFT (float descriptors)
        try:
            sift = cv2.SIFT_create()
            kp, d = sift.detectAndCompute(gray, mask)
            if kp is not None and len(kp) > 0:
                return kp, d, "SIFT"
        except Exception:
            pass
        # Fallback ORB (binary descriptors)
        orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10, edgeThreshold=15)
        kp = orb.detect(gray, mask)
        kp, d = orb.compute(gray, kp)
        return kp, d, "ORB"

    # ---------- Descriptor matching with proper norm + Lowe ratio ----------
    def match_descriptors(self, d1: Optional[np.ndarray], d2: Optional[np.ndarray], ratio: float = 0.75) -> List[cv2.DMatch]:
        if d1 is None or d2 is None:
            return []
        norm = cv2.NORM_L2 if d1.dtype in (np.float32, np.float64) else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=False)
        raw = bf.knnMatch(d1, d2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        return good

    # ---------- Homography inlier ratio ----------
    def homography_inlier_ratio(self, kp1, kp2, matches: List[cv2.DMatch], ransac_thresh: float = 3.0) -> float:
        if matches is None or len(matches) < 8:
            return 0.0
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
        if mask is None:
            return 0.0
        return float(mask.ravel().mean())  # 0..1

    # ---------- SSIM (exclude masked-out areas by filling them with mean) ----------
    def compute_ssim(self, gray1: np.ndarray, gray2: np.ndarray, mask_bg: np.ndarray) -> float:
        try:
            g1, g2 = gray1.copy(), gray2.copy()
            excluded = ~mask_bg.astype(bool)  # where we DON'T want to compare
            if np.any(excluded):
                m1 = float(np.mean(g1[excluded]))
                m2 = float(np.mean(g2[excluded]))
                g1[excluded] = m1
                g2[excluded] = m2
            return float(ssim(g1, g2, data_range=255))
        except Exception:
            return 0.0

    # ---------- Flow std computed only on BG region ----------
    def compute_flow_std(self, gray1: np.ndarray, gray2: np.ndarray, mask_bg: np.ndarray) -> float:
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=7, poly_sigma=1.5, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            region = mask_bg.astype(bool)
            if not np.any(region):
                return 0.0
            return float(np.std(mag[region]))
        except Exception:
            return 0.0

    # ---------- Visualization ----------
    def draw_colorful_matches(self, img1, kp1, img2, kp2, matches, out_path):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        if matches:
            for m in matches:
                color = tuple(map(int, np.random.randint(50, 256, 3)))
                p1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                p2 = tuple((np.round(kp2[m.trainIdx].pt).astype(int) + np.array([w1, 0])).tolist())
                cv2.circle(canvas, p1, 4, color, -1)
                cv2.circle(canvas, p2, 4, color, -1)
                cv2.line(canvas, p1, p2, color, 2)
        cv2.imwrite(out_path, canvas)
        return out_path

    # ---------- A) Background stabilization ----------
    def stabilize_by_background(self, img1: np.ndarray, img2: np.ndarray, mask_bg: np.ndarray):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kps1, d1, _ = self._extract_features(g1, mask_bg.astype(np.uint8))
        kps2, d2, _ = self._extract_features(g2, mask_bg.astype(np.uint8))
        matches = self.match_descriptors(d1, d2, ratio=0.75)
        if not matches or not kps1 or not kps2 or len(matches) < 8:
            return img2, 0, None, None
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, inl = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
        if H is None or inl is None or inl.ravel().sum() < 8:
            return img2, 0, None, None
        img2s = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        inliers = int(inl.ravel().sum())
        reproj = cv2.perspectiveTransform(pts2[inl.ravel()==1], H)
        err = float(np.mean(np.linalg.norm(reproj - pts1[inl.ravel()==1], axis=2)))
        return img2s, inliers, err, H

    # ---------- Residual motion after stabilization ----------
    @staticmethod
    def residual_motion(img1: np.ndarray, img2s: np.ndarray, face_mask: np.ndarray, bg_mask: np.ndarray):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2s, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 21, 3, 7, 1.5, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        mu_face = float(mag[face_mask.astype(bool)].mean()) if np.any(face_mask) else 0.0
        mu_bg   = float(mag[bg_mask.astype(bool)].mean())   if np.any(bg_mask)   else 0.0
        return mu_face, mu_bg

    # ---------- B) Face planarity features ----------
    @staticmethod
    def face_planarity_features(pts0: np.ndarray, Z0: np.ndarray, pts1: np.ndarray):
        if pts0 is None or pts1 is None or len(pts0)!=len(pts1) or len(pts0)<20:
            return 0.0, 0.0, 0  # corr, mean residual, inliers
        p0 = pts0.reshape(-1,1,2); p1 = pts1.reshape(-1,1,2)
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if H is None or inl is None or inl.ravel().sum()<20:
            return 0.0, 0.0, 0
        pred = cv2.perspectiveTransform(p0[inl.ravel()==1], H)
        resid = np.linalg.norm(pred - p1[inl.ravel()==1], axis=2).ravel() + 1e-6
        Zsel  = Z0[inl.ravel()==1] + 1e-6
        corr = float(np.corrcoef(resid, Zsel)[0,1])
        eH   = float(np.mean(resid))
        return corr, eH, int(inl.ravel().sum())

    # ---------- C) Multi-patch homography dispersion ----------
    @staticmethod
    def face_multiH_dispersion(pts0: np.ndarray, pts1: np.ndarray, K: int = 8):
        if pts0 is None or pts1 is None or len(pts0)<40:
            return 0.0
        Z = np.float32(pts0)
        K = min(K, max(3, len(pts0)//8))
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        try:
            _, labels, _ = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        except Exception:
            return 0.0
        Hs = []
        p0 = pts0.reshape(-1,1,2); p1 = pts1.reshape(-1,1,2)
        for k in range(K):
            idx = np.where(labels.ravel()==k)[0]
            if len(idx) < 8: continue
            H, inl = cv2.findHomography(p0[idx], p1[idx], cv2.RANSAC, 3.0)
            if H is not None: Hs.append(H.flatten())
        if len(Hs) < 2: return 0.0
        Hs = np.stack(Hs, axis=0)
        return float(np.mean(np.std(Hs, axis=0)))  # dispersion index

    # ---------- Main analysis ----------
    def analyze_images(self, img1_path: str, img2_path: str, params: DetectionParams) -> 'DetectionResults':
        try:
            # Load + downscale
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                raise ValueError("Could not load one or both images")
            img1 = _resize_max(img1, MAX_WIDTH)
            img2 = _resize_max(img2, MAX_WIDTH)

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Build BG masks by excluding face + bottom 30%
            m1 = np.ones_like(gray1, np.uint8)
            m2 = np.ones_like(gray2, np.uint8)

            f0 = self.detect_face_box(img1)
            f1 = self.detect_face_box(img2)
            face_mask1 = np.zeros_like(gray1, np.uint8)
            face_mask2 = np.zeros_like(gray2, np.uint8)
            if f0:
                x, y, w, h = f0; m1[y:y + h, x:x + w] = 0; face_mask1[y:y+h, x:x+w] = 1
            if f1:
                x, y, w, h = f1; m2[y:y + h, x:x + w] = 0; face_mask2[y:y+h, x:x+w] = 1

            bh1 = int(gray1.shape[0] * 0.3)
            bh2 = int(gray2.shape[0] * 0.3)
            m1[-bh1:, :] = 0
            m2[-bh2:, :] = 0

            # Crop to common area for SSIM/flow (BG-only consistency)
            h0 = min(gray1.shape[0], gray2.shape[0])
            w0 = min(gray1.shape[1], gray2.shape[1])
            g1c, g2c = gray1[:h0, :w0], gray2[:h0, :w0]
            mc = (m1[:h0, :w0] & m2[:h0, :w0]).astype(np.uint8)

            # --- BG features + matches ---
            kp1_bg, d1_bg, method_bg = self._extract_features(gray1, mask=m1)
            kp2_bg, d2_bg, _        = self._extract_features(gray2, mask=m2)
            mg_bg = self.match_descriptors(d1_bg, d2_bg, ratio=0.75)

            total_bg = len(mg_bg)
            max_kp_bg = max(len(kp1_bg) if kp1_bg else 0, len(kp2_bg) if kp2_bg else 0)
            ratio_bg = (total_bg / max_kp_bg) if max_kp_bg else 0.0

            # --- Full image features (for viz only) ---
            kp1_full, d1_full, _ = self._extract_features(gray1, mask=None)
            kp2_full, d2_full, _ = self._extract_features(gray2, mask=None)
            mg_full = self.match_descriptors(d1_full, d2_full, ratio=0.75)

            # --- Visualizations ---
            session_id = str(uuid.uuid4())
            bg_viz_path   = os.path.join(RESULTS_FOLDER, f"matches_bg_{session_id}.png")
            orig_viz_path = os.path.join(RESULTS_FOLDER, f"orig_matches_{session_id}.png")

            top_bg   = sorted(mg_bg,   key=lambda x: x.distance)[:params.num_matches] if mg_bg   else []
            top_full = sorted(mg_full, key=lambda x: x.distance)[:params.num_matches] if mg_full else []

            self.draw_colorful_matches(img1, kp1_bg or [], img2, kp2_bg or [], top_bg, bg_viz_path)
            self.draw_colorful_matches(img1, kp1_full or [], img2, kp2_full or [], top_full, orig_viz_path)

            # --- Metrics on BG only ---
            ss = self.compute_ssim(g1c, g2c, mc)
            fs = self.compute_flow_std(g1c, g2c, mc)
            inlier_ratio_bg = self.homography_inlier_ratio(kp1_bg or [], kp2_bg or [], mg_bg)

            # --- A) Stabilize img2 by background + residual motion gate ---
            img2_stab, stab_inliers, stab_err_px, Hbg = self.stabilize_by_background(img1, img2, m1)
            if stab_err_px is None:
                stab_err_px = 1e9  # large

            # --- FaceMesh landmarks for normalization + geometry ---
            pts0, Z0 = self.facemesh_xyZ(img1)
            pts1, Z1 = self.facemesh_xyZ(img2_stab)
            io = self.inter_ocular(pts0) if pts0 is not None else 0.0
            norm = max(io, 1e-6)  # avoid div by zero

            mu_face_px, mu_bg_px = self.residual_motion(img1, img2_stab, face_mask1, m1)
            mu_face_norm = mu_face_px / norm
            mu_bg_norm   = mu_bg_px   / norm
            stab_err_norm = float(stab_err_px) / norm if stab_err_px is not None else 1e6

            head_moved = (mu_face_norm > params.mu_face_thresh_norm)
            bg_static  = (mu_bg_norm   < params.mu_bg_thresh_norm)
            stab_ok    = (stab_inliers >= params.min_stab_inliers) and (stab_err_norm <= params.max_stab_err)

            # --- B) Face geometry features (planarity vs 3D) ---
            corr_Z_resid, eH_face_px, inl_face = self.face_planarity_features(pts0, Z0, pts1)
            eH_face_norm = (eH_face_px / norm) if io > 0 else 0.0
            disp_face = self.face_multiH_dispersion(pts0, pts1)

            # --- Decisions (fusion) ---
            bg_inconsistent = (total_bg < params.min_matches) or (ratio_bg < params.match_ratio_thresh)
            consistency_passed = (ss < params.ssim_thresh) or (fs > params.flow_std_thresh)
            planar_bg = (inlier_ratio_bg > params.homography_inlier_thresh)

            # Live votes from independent cues
            live_votes = 0
            if stab_ok and head_moved and bg_static:
                live_votes += 1
            if corr_Z_resid > params.corrZ_thresh and eH_face_norm > params.eH_face_thresh_norm and inl_face >= 20:
                live_votes += 1
            if disp_face > params.disp_face_thresh:
                live_votes += 1

            # Strong spoof rule for flat screen on static BG
            if planar_bg and (ss >= params.ssim_thresh) and (total_bg >= params.min_matches):
                is_live = False
            else:
                # Conservative fusion: need at least 2 live votes
                is_live = (live_votes >= 2)

            logger.info(
                ("BG(%s): kp1=%d kp2=%d good=%d ratio=%.3f | SSIM=%.3f<th=%.2f | FLOWSTD=%.3f>th=%.2f | "
                 "inliers=%.3f>%.2f | STAB inl=%d errN=%.4f | IO=%.1f | muF=%.4f muB=%.4f | "
                 "corrZ=%.3f>%.2f eHn=%.4f>%.4f disp=%.3f>%.3f | live=%s"),
                method_bg,
                len(kp1_bg) if kp1_bg else 0,
                len(kp2_bg) if kp2_bg else 0,
                total_bg, ratio_bg,
                ss, params.ssim_thresh,
                fs, params.flow_std_thresh,
                inlier_ratio_bg, params.homography_inlier_thresh,
                stab_inliers, stab_err_norm,
                io,
                mu_face_norm, mu_bg_norm,
                corr_Z_resid, params.corrZ_thresh, eH_face_norm, params.eH_face_thresh_norm, disp_face, params.disp_face_thresh,
                str(is_live)
            )

            return DetectionResults(
                is_live=is_live,
                is_spoof=not is_live,
                background_analysis=BackgroundAnalysis(
                    total_matches=int(total_bg),
                    match_ratio=round(float(ratio_bg), 3),
                    inlier_ratio=round(float(inlier_ratio_bg), 3),
                    planar_bg=bool(planar_bg),
                    inconsistent=bool(bg_inconsistent)
                ),
                consistency_metrics=ConsistencyMetrics(
                    ssim=round(float(ss), 3),
                    flow_std=round(float(fs), 3),
                    ssim_threshold=float(params.ssim_thresh),
                    flow_threshold=float(params.flow_std_thresh),
                    consistency_passed=bool(consistency_passed)
                ),
                face_metrics=FaceMetrics(
                    inter_ocular=round(float(io), 3),
                    mu_face_norm=round(float(mu_face_norm), 4),
                    mu_bg_norm=round(float(mu_bg_norm), 4),
                    stabilized_inliers=int(stab_inliers),
                    stabilized_error_norm=round(float(stab_err_norm), 4),
                    head_moved=bool(head_moved),
                    bg_static=bool(bg_static),
                    corr_Z_resid=round(float(corr_Z_resid), 3),
                    eH_face_norm=round(float(eH_face_norm), 4),
                    disp_face=round(float(disp_face), 3),
                ),
                visualizations=Visualizations(
                    background_matches=f"matches_bg_{session_id}.png",
                    original_matches=f"orig_matches_{session_id}.png"
                ),
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"Error analyzing images: {str(e)}")
            raise

# Initialize detector
detector = FaceSpoofingDetector()

def run_detection(img1_path: str, img2_path: str, params: DetectionParams) -> DetectionResults:
    return detector.analyze_images(img1_path, img2_path, params)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_spoofing(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),
    # Existing knobs
    min_matches: int = Form(30),
    num_matches: int = Form(100),
    ssim_thresh: float = Form(0.90),
    flow_std_thresh: float = Form(1.2),
    match_ratio_thresh: float = Form(0.10),
    homography_inlier_thresh: float = Form(0.6),
    # New knobs (optional; safe defaults)
    min_stab_inliers: int = Form(30),
    max_stab_err: float = Form(0.025),
    mu_face_thresh_norm: float = Form(0.015),
    mu_bg_thresh_norm: float = Form(0.008),
    corrZ_thresh: float = Form(0.25),
    eH_face_thresh_norm: float = Form(0.012),
    disp_face_thresh: float = Form(0.10),
):
    """
    Detect face spoofing by uploading two image files (uses BG stabilization + face 3D features).
    """
    try:
        if not validate_file_extension(image1.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type for image1: {image1.filename}")
        if not validate_file_extension(image2.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type for image2: {image2.filename}")

        params = DetectionParams(
            min_matches=min_matches,
            num_matches=num_matches,
            ssim_thresh=ssim_thresh,
            flow_std_thresh=flow_std_thresh,
            match_ratio_thresh=match_ratio_thresh,
            homography_inlier_thresh=homography_inlier_thresh,
            min_stab_inliers=min_stab_inliers,
            max_stab_err=max_stab_err,
            mu_face_thresh_norm=mu_face_thresh_norm,
            mu_bg_thresh_norm=mu_bg_thresh_norm,
            corrZ_thresh=corrZ_thresh,
            eH_face_thresh_norm=eH_face_thresh_norm,
            disp_face_thresh=disp_face_thresh
        )

        session_id = str(uuid.uuid4())
        filepath1 = os.path.join(UPLOAD_FOLDER, f"{session_id}_1_{image1.filename}")
        filepath2 = os.path.join(UPLOAD_FOLDER, f"{session_id}_2_{image2.filename}")

        await save_upload_file(image1, filepath1)
        await save_upload_file(image2, filepath2)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, run_detection, filepath1, filepath2, params)

        # Cleanup uploads (visualizations kept)
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except Exception:
            pass

        return DetectionResponse(
            success=True,
            results=results,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /detect: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_base64", response_model=DetectionResponse)
async def detect_spoofing_base64(request: Base64DetectionRequest):
    """
    Detect face spoofing using base64 encoded images (data URI supported)
    """
    try:
        try:
            img1_data = _decode_b64(request.image1)
            img2_data = _decode_b64(request.image2)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        if len(img1_data) > MAX_FILE_SIZE or len(img2_data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Image file too large")

        session_id = str(uuid.uuid4())
        filepath1 = os.path.join(UPLOAD_FOLDER, f"{session_id}_1.jpg")
        filepath2 = os.path.join(UPLOAD_FOLDER, f"{session_id}_2.jpg")

        async with aiofiles.open(filepath1, 'wb') as f:
            await f.write(img1_data)
        async with aiofiles.open(filepath2, 'wb') as f:
            await f.write(img2_data)

        params = request.params or DetectionParams()

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, run_detection, filepath1, filepath2, params)

        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except Exception:
            pass

        return DetectionResponse(
            success=True,
            results=results,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /detect_base64: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """
    Get visualization image by filename (matches_bg_*.png or orig_matches_*.png)
    """
    try:
        if not (filename.startswith("matches_bg_") or filename.startswith("orig_matches_")) or not filename.endswith(".png"):
            raise HTTPException(status_code=400, detail="Invalid filename")
        filepath = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Visualization not found")
        return FileResponse(filepath, media_type="image/png", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup/{session_id}", response_model=CleanupResponse)
async def cleanup_session(session_id: str):
    """
    Clean up visualization files for a specific session
    """
    try:
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        bg_file = os.path.join(RESULTS_FOLDER, f"matches_bg_{session_id}.png")
        orig_file = os.path.join(RESULTS_FOLDER, f"orig_matches_{session_id}.png")

        removed = []
        if os.path.exists(bg_file):
            os.remove(bg_file)
            removed.append(f"matches_bg_{session_id}.png")
        if os.path.exists(orig_file):
            os.remove(orig_file)
            removed.append(f"orig_matches_{session_id}.png")

        return CleanupResponse(
            success=True,
            removed_files=removed,
            message=f"Cleaned up session {session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

Run it

pip install -U fastapi uvicorn numpy opencv-contrib-python scikit-image mediapipe aiofiles
uvicorn app:app --reload --port 8000

Notes & tuning
	•	Normalization: All pixel thresholds that depend on geometry are normalized by inter-ocular distance → robust across scales.
	•	Gates that matter most:
	•	min_stab_inliers & max_stab_err (how good background stabilization must be),
	•	mu_face_thresh_norm & mu_bg_thresh_norm (head really moved while background stayed put),
	•	corrZ_thresh, eH_face_thresh_norm, disp_face_thresh (3D-ness of the face).
	•	Start with the defaults; then calibrate to your capture set for your desired BPCER@APCER.





You’ve built a background-consistency detector (nice!), but right now it’s mostly deciding “live vs spoof” from what the background does. That’s fast and catches some easy replays, but it leaves big blind spots because it never looks at the face’s 3D behavior.

Below is (1) a blunt assessment of what your current code can/can’t catch, and (2) a surgical upgrade plan with drop-in code snippets you can paste into your file to make it much harder to bypass—without blowing up latency or your API.

⸻

1) What your current code covers (honest take)

Signals you use now
	•	BG keypoint matches (SIFT/ORB), SSIM on BG, Farnebäck flow std on BG, BG homography inlier ratio.
	•	Face is only used to mask it out (no geometry on the face).

What it will catch reliably
	•	Static planar replay in front of a stable background (same room, same camera) → planar BG + high SSIM → mark spoof.
	•	Some obvious edits (frame crops/resizes) when BG keypoints mismatch.

Likely to miss / fragile
	•	Co-rotation: attacker rotates phone + print together → BG moves with the print: your logic can call this “live”.
	•	Textureless BG (blank wall/curtain) → low matches → your rule “bg_inconsistent & movement → live” can false-accept.
	•	Dynamic BG replay (pre-recorded room video behind a screen) → looks “live” to BG metrics.
	•	3D masks / makeup / high-quality screen: no face-3D checks → pass if BG looks “inconsistent”.
	•	Stream injection: unrelated to vision cues (needs origin hardening).

Bottom line
Ballpark: good at easy print/screen in static scenes, weak against co-rotation, dynamic replays, masks. Don’t expect more than “basic PAD” with this alone.

⸻

2) High-impact upgrades (keep your API, same speed class)

Tier A — must-add (cheap + big payoff)
	1.	Background stabilization + residual head motion gate
Stabilize image2 to image1 by the background; then demand non-zero motion on the face but near-zero on the BG. This defeats co-rotation and forces real head motion.
	2.	Face 3D geometry from two frames (camera-motion-invariant)
	•	FaceMesh landmarks (468 pts) for both frames.
	•	“Planarity vs. 3D” test on the face: fit a single homography on face points; compute residuals. The correlation between residual magnitude and FaceMesh depth (|z|) should be high for a real 3D face, low for a flat spoof.
	•	Multi-patch homography dispersion on the face (nose/lips vs cheeks/forehead): 3D → high dispersion; planar → low.
	3.	Simple fusion
Replace the hard if/else with a 5–8 feature score (still no ML if you want), normalized by inter-ocular distance.

Tier B — nice extras (still light)
	•	Moiré / refresh artifact score (FFT peaks) to catch screen replays.
	•	Iris foreshortening delta (ellipse ratio change L/R eye between frames).

Tier C — policy hardening (outside vision)
	•	Do not accept if either IMU rotation > ~5° (phone moved) or “stabilized residual head motion” is too small (user didn’t actually move).
	•	Anti-injection: hash chain frames + per-session challenge token embedded in the UI and visible in pixels.

⸻

3) Drop-in code (pasteable blocks)

Add these helpers near your other utility functions. They reuse your imports and style.

(A) Background stabilization + residual motion

def stabilize_by_background(img1: np.ndarray, img2: np.ndarray, mask_bg: np.ndarray):
    """Warp img2 so the BACKGROUND aligns to img1. Returns (img2_stab, inliers, reproj_err)."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kps1, d1, _ = detector._extract_features(g1, mask_bg.astype(np.uint8))
    kps2, d2, _ = detector._extract_features(g2, mask_bg.astype(np.uint8))
    matches = detector.match_descriptors(d1, d2, ratio=0.75)
    if len(matches) < 30 or not kps1 or not kps2:
        return img2, 0, None  # can't stabilize
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, inl = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    if H is None or inl is None:
        return img2, 0, None
    img2s = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    inliers = int(inl.ravel().sum())
    reproj = cv2.perspectiveTransform(pts2[inl.ravel()==1], H)
    err = float(np.mean(np.linalg.norm(reproj - pts1[inl.ravel()==1], axis=2)))
    return img2s, inliers, err

def residual_motion(img1: np.ndarray, img2s: np.ndarray, face_mask: np.ndarray, bg_mask: np.ndarray):
    """Mean optical-flow magnitude on face vs background AFTER stabilization."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2s, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mu_face = float(mag[face_mask.astype(bool)].mean()) if np.any(face_mask) else 0.0
    mu_bg   = float(mag[bg_mask.astype(bool)].mean())   if np.any(bg_mask)   else 0.0
    return mu_face, mu_bg

(B) FaceMesh 3D residual-depth correlation (planarity vs 3D)

def facemesh_xyZ(img: np.ndarray):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
    res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fm.close()
    if not res.multi_face_landmarks:
        return None, None
    h, w = img.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x*w, p.y*h] for p in lm], np.float32)       # (N,2)
    Z   = np.array([abs(p.z) for p in lm], np.float32)             # (N,)
    return pts, Z

def face_planarity_features(pts0: np.ndarray, Z0: np.ndarray, pts1: np.ndarray):
    """Fit a single homography on face landmarks; correlate residual with |Z| (protrusion)."""
    if pts0 is None or pts1 is None or len(pts0)!=len(pts1) or len(pts0)<20:
        return 0.0, 0.0
    p0 = pts0.reshape(-1,1,2); p1 = pts1.reshape(-1,1,2)
    H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if H is None or inl is None or inl.ravel().sum()<20:
        return 0.0, 0.0
    pred = cv2.perspectiveTransform(p0[inl.ravel()==1], H)
    resid = np.linalg.norm(pred - p1[inl.ravel()==1], axis=2).ravel() + 1e-6
    Zsel  = Z0[inl.ravel()==1] + 1e-6
    # Pearson corr between residual and |Z| (nose/lips should have higher residual on real 3D)
    corr = float(np.corrcoef(resid, Zsel)[0,1])
    eH   = float(np.mean(resid))
    return corr, eH  # higher corr → more 3D; lower corr → more planar

(C) Multi-patch homography dispersion on the face

def face_multiH_dispersion(img0: np.ndarray, img1s: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, K: int = 8):
    """Partition face landmarks into K spatial clusters; fit H per cluster; return parameter dispersion."""
    if pts0 is None or pts1 is None or len(pts0)<40:
        return 0.0
    # k-means on 2D coords (OpenCV expects float32)
    Z = np.float32(pts0)
    K = min(K, len(pts0)//8) if len(pts0)>=64 else min(4, len(pts0)//6)
    if K < 3: return 0.0
    _, labels, centers = cv2.kmeans(Z, K, None,
                                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                                    3, cv2.KMEANS_PP_CENTERS)
    Hs = []
    p0 = pts0.reshape(-1,1,2); p1 = pts1.reshape(-1,1,2)
    for k in range(K):
        idx = np.where(labels.ravel()==k)[0]
        if len(idx) < 8: continue
        H, inl = cv2.findHomography(p0[idx], p1[idx], cv2.RANSAC, 3.0)
        if H is not None: Hs.append(H.flatten())
    if len(Hs) < 2: return 0.0
    Hs = np.stack(Hs, axis=0)
    return float(np.mean(np.std(Hs, axis=0)))  # dispersion index; higher → more 3D

Integrate into your analyze_images(...)

Inside analyze_images, after you create m1/m2 (BG masks) and have img1/img2:

# Build binary masks
face_mask1 = (1 - m1).astype(bool)
face_mask2 = (1 - m2).astype(bool)
bg_mask1   = m1.astype(bool)

# --- A) Stabilize image2 by background ---
img2_stab, stab_inliers, stab_err = stabilize_by_background(img1, img2, m1)
mu_face, mu_bg = residual_motion(img1, img2_stab, face_mask1, m1)

# Gate: require actual head motion after stabilization, and almost no BG motion
head_moved = (mu_face > 1.5)   # tune
bg_static  = (mu_bg   < 0.8)   # tune
if not head_moved or not bg_static:
    # Ask for retake OR mark spoof: here we mark 'retake' via conservative scoring
    pass  # keep computing features but this should heavily penalize acceptance

Then compute face geometry:

# --- B) Face geometry features (planarity vs 3D) ---
pts0, Z0 = facemesh_xyZ(img1)
pts1, Z1 = facemesh_xyZ(img2_stab)
corr_Z_resid, eH_face = face_planarity_features(pts0, Z0, pts1)
disp_face = face_multiH_dispersion(img1, img2_stab, pts0, pts1)

And fuse (simple rule to start):

# Simple fusion (normalized by bg diagnostics)
feat_live_votes = 0
if head_moved and bg_static: feat_live_votes += 1
if corr_Z_resid > 0.25 and eH_face > 1.2: feat_live_votes += 1
if disp_face > 0.10: feat_live_votes += 1

# Keep your old BG planar screen catch (strong spoof signal)
planar_bg = (inlier_ratio_bg > params.homography_inlier_thresh)

if planar_bg and (ss >= params.ssim_thresh) and (total_bg >= params.min_matches):
    is_live = False  # screen/flat replay on static background
else:
    is_live = feat_live_votes >= 2  # need at least 2 independent live cues

All thresholds above are starting points at ~320–640 px face width. Calibrate on your captures (see below).

⸻

4) Thresholds & calibration (quick recipe)
	1.	Collect ~1–2k pairs (balanced bona fide vs attacks you care about: print, screen, co-rotation, dynamic BG replay, mask if possible).
	2.	Normalize geometric pixel errors by inter-ocular distance.
	3.	Grid-search thresholds to minimize BPCER@APCER=1%.
	4.	If you can, fit a tiny LogisticRegression/LightGBM on:

[mu_face, mu_bg, corr_Z_resid, eH_face, disp_face,
 inlier_ratio_bg, ss_bg, flowstd_bg, match_ratio_bg, stab_inliers]

Keep the retake gate separate (when stabilization fails or head_moved/bg_static violated).

⸻

5) Bonus ideas (low risk, good wins)
	•	Iris foreshortening delta: fit ellipse to each iris (FaceMesh eye landmarks), compare major/minor ratio left vs right between frames → real yaw shows asymmetry changes.
	•	FFT moiré score on cheek/forehead crops to snipe OLED/LCD replays (narrowband peaks at 2–4 px/cycle).
	•	Client-side IMU gate: if Δgyro>5°, reject capture before it hits your API.

⸻

TL;DR
	•	Your current system is BG-only → good for basic screens/prints, weak on co-rotation & dynamic replays.
	•	Add BG stabilization + residual head motion, and face 3D planarity checks (3 short functions above).
	•	Fuse with a tiny rule (≥2 live votes) or a 1-page LogReg.
	•	Calibrate on your data. This combo is lightweight, robust, and a huge step toward bank-grade PAD without changing your API.

If you want, I can add a tiny /calibrate route that ingests a labeled CSV of pairs and writes back tuned thresholds—just say the word and I’ll wire it in.