"""
video_utils.py

Video handling, frame extraction utilities, and a simple OpenCV-based
heuristic ambulance detector + overlay functions for visualization.

Functions:
- sample_frames_from_file(file_like, sample_n=5) -> list[(frame_index, rgb_array)]
- extract_frames(file_path, step=30) -> list[(frame_index, frame)]
- frame_to_rgb(frame) -> rgb_array

Heuristic detector functions (OpenCV required):
- detect_ambulance_in_frame(frame_rgb, area_thresh=500, red_only=False) -> (detected: bool, density: float, bbox: Optional[tuple])
- annotate_frame(frame_rgb, bbox=None, label=None) -> annotated_rgb

Notes:
- If OpenCV (cv2) is missing, functions that require it raise RuntimeError with guidance.
- The detector is intentionally simple: it looks for red regions (ambulance lights)
  and checks area; it also returns a crude traffic density estimate (0..1).
"""
from __future__ import annotations
import os
import tempfile
import typing
import warnings
from typing import List, Tuple, Any, Optional

import numpy as np

# Try import cv2; degrade gracefully for non-CV environments
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def _ensure_cv2():
    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV (cv2) is required for video utilities and the ambulance detector. "
            "Install it with `pip install opencv-python` and try again."
        )


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to RGB numpy.ndarray dtype=uint8.
    If conversion fails or cv2 is not present, raise RuntimeError.
    """
    if frame is None:
        raise ValueError("Empty frame passed to frame_to_rgb")
    if not _HAS_CV2:
        raise RuntimeError("OpenCV not available - cannot convert frame to RGB.")
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception:
        # if conversion fails, try to coerce to uint8 and return
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr


def extract_frames(file_path: str, step: int = 30) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames from a video file every `step` frames (BGR frames).
    Returns list of tuples (frame_index, frame_bgr).
    """
    _ensure_cv2()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {file_path}")

    frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append((idx, frame))
            idx += 1
    finally:
        cap.release()
    return frames


def sample_frames_from_file(file_like: typing.Union[str, typing.IO, bytes], sample_n: int = 5) -> List[Tuple[int, np.ndarray]]:
    """
    Sample `sample_n` frames evenly across provided video file-like object or path.

    file_like may be:
    - path string
    - bytes (raw video bytes)
    - file-like object with .read()

    Returns list of (frame_index, rgb_array).
    Ensures temporary files are cleaned up.
    """
    _ensure_cv2()
    temp_path = None
    try:
        if isinstance(file_like, str):
            video_path = file_like
            if not os.path.exists(video_path):
                raise FileNotFoundError(video_path)
        else:
            # Write bytes / file-like to temporary file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_path = tmp.name
            if isinstance(file_like, bytes):
                tmp.write(file_like)
            else:
                # file-like with read()
                data = file_like.read()
                tmp.write(data)
            tmp.flush()
            tmp.close()
            video_path = temp_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open temporary video file.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            # fallback to sequential extraction
            frames_bgr = extract_frames(video_path, step=30)
            selected = frames_bgr[:sample_n]
            result = [(idx, frame_to_rgb(frame)) for idx, frame in selected]
            return result

        indices = list({int(round(i)) for i in np.linspace(0, max(frame_count - 1, 0), num=sample_n)})
        result: List[Tuple[int, np.ndarray]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = frame_to_rgb(frame)
            result.append((idx, rgb))
        cap.release()
        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                warnings.warn("Could not remove temporary video file: %s" % temp_path)


# -----------------------------
# Simple OpenCV-based detector
# -----------------------------
def _red_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Given an RGB frame (H,W,3) uint8, return a binary mask for red-ish pixels.
    Uses HSV thresholds that cover red hues (~0-10 and ~160-180).
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV not available for color masking.")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red range 1
    lower1 = np.array([0, 80, 60])    # H,S,V
    upper1 = np.array([10, 255, 255])
    # Red range 2
    lower2 = np.array([160, 80, 60])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological ops to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_ambulance_in_frame(
    frame_rgb: np.ndarray,
    area_thresh: int = 800,
    red_only: bool = False,
) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    """
    Heuristic detector for ambulances in a single RGB frame.

    Args:
        frame_rgb: HxWx3 uint8 RGB image (numpy array).
        area_thresh: minimum mask area (in pixels) to consider a valid light region.
        red_only: if True, use only color detection; if False, include simple contour heuristics.

    Returns:
        detected (bool): True if ambulance-like lights are detected.
        density (float): crude traffic density estimate in [0,1] (higher => denser).
        bbox (tuple or None): bounding box (x, y, w, h) of the detected region if any.
    """
    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV (cv2) is required for ambulance detection. Install with `pip install opencv-python`."
        )

    if frame_rgb is None:
        return False, 0.0, None

    # Ensure uint8
    arr = np.asarray(frame_rgb)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    frame_area = h * w

    try:
        mask = _red_mask_from_rgb(arr)
    except Exception:
        # If HSV conversion fails, fallback to simple red filter on RGB channels
        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)
        mask = ((r > 150) & (r > (g + 30)) & (r > (b + 30))).astype(np.uint8) * 255

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    best_bbox = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / float(max(1, hh))
        # Heuristic: ambulance light blobs may be wide or tall; accept reasonable aspect ratios
        if area > best_area:
            best_area = area
            best_bbox = (x, y, ww, hh)
            detected = True

    # Crude traffic density estimate:
    # Use edge density (Canny edges) normalized by frame area; higher edges => more objects => denser scene.
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum()) / (255.0 * frame_area)
        # normalize to 0..1
        density = min(1.0, max(0.0, edge_density * 3.0))
    except Exception:
        density = 0.5  # fallback neutral

    return bool(detected), float(density), best_bbox


def annotate_frame(frame_rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None, label: Optional[str] = None) -> np.ndarray:
    """
    Draw bounding box and label onto the provided RGB frame and return annotated RGB frame.
    Keeps original dtype uint8.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required to annotate frames.")

    frame = frame_rgb.copy()
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Convert RGB -> BGR for cv2 drawing, then back to RGB at the end
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if bbox is not None:
        x, y, w, h = bbox
        # draw rectangle
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        # label background
        if label:
            txt = label
            (text_w, text_h), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(bgr, (x, y - text_h - 8), (x + text_w + 8, y), (0, 0, 255), thickness=-1)
            cv2.putText(bgr, txt, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Optionally, overlay a small timestamp or marker
    # convert back to RGB
    annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return annotated


# Small demo helper (not required) to run detector on frames and return annotated frames
def detect_and_annotate_frames(frames_rgb: List[Tuple[int, np.ndarray]], area_thresh: int = 800) -> List[Tuple[int, np.ndarray, bool, float]]:
    """
    Given a list of (frame_index, rgb_array), run detector on each and return list of tuples:
    (frame_index, annotated_rgb, detected_bool, density_float)
    """
    results = []
    for idx, rgb in frames_rgb:
        try:
            detected, density, bbox = detect_ambulance_in_frame(rgb, area_thresh=area_thresh)
            label = None
            if detected:
                label = f"Ambulance (density={density:.2f})"
            annotated = annotate_frame(rgb, bbox=bbox, label=label)
            results.append((idx, annotated, detected, density))
        except Exception:
            # on failure, return original frame with neutral info
            results.append((idx, rgb, False, 0.0))
    return results
