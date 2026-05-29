# ============================================================
# video_features.py
# Real-time video feature extraction utilities
# Used by:
#   - RL Lane Changing
#   - Accident Prediction
# ============================================================

import cv2
import numpy as np


# ============================================================
# RL FEATURE EXTRACTION
# ============================================================
def extract_video_features(frame, prev_frame=None):
    """
    Extract features for RL lane-changing.

    Returns:
        {
            "speed": float,
            "density": float,
            "front_distance": float
        }
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------
    # SPEED PROXY (OPTICAL FLOW)
    # ---------------------------
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        speed = float(np.mean(mag) * 30)
    else:
        speed = 40.0  # default startup value

    # ---------------------------
    # DENSITY PROXY (EDGE COUNT)
    # ---------------------------
    edges = cv2.Canny(gray, 80, 160)
    density = float(np.sum(edges > 0) / edges.size * 200)

    # ---------------------------
    # FRONT DISTANCE PROXY
    # ---------------------------
    front_distance = max(5.0, 70.0 - density * 0.25)

    return {
        "speed": speed,
        "density": density,
        "front_distance": front_distance,
    }


# ============================================================
# ACCIDENT PREDICTION FEATURE EXTRACTION
# ============================================================
def extract_accident_features(frame, prev_frame=None):
    """
    Extract accident-relevant features from video frames with OPTIMIZED SPEED.
    
    ⚡ OPTIMIZED: Uses frame differencing instead of optical flow for speed
    Returns:
        {
            "speed": float,
            "density": float,
            "front_distance": float,
            "motion_variance": float,
            "speed_change": float,
        }
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------
    # ⚡ FAST MOTION DETECTION
    # ---------------------------
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Fast frame differencing (much faster than Farneback optical flow!)
        diff = cv2.absdiff(gray, prev_gray)
        speed_proxy = float(np.mean(diff) * 3)  # Scale to reasonable range
        motion_variance = float(np.std(diff) * 2)
        speed_change = float(np.abs(np.mean(diff)))
    else:
        speed_proxy = 25.0
        motion_variance = 5.0
        speed_change = 0.0

    # ---------------------------
    # FAST DENSITY PROXY (single Canny)
    # ---------------------------
    edges = cv2.Canny(gray, 80, 160)
    density_proxy = float(np.mean(edges > 0) * 200)

    # ---------------------------
    # FRONT DISTANCE PROXY
    # ---------------------------
    front_distance_proxy = max(5.0, 60.0 - density_proxy * 0.2)

    return {
        "speed": speed_proxy,
        "density": density_proxy,
        "front_distance": front_distance_proxy,
        "motion_variance": motion_variance,
        "speed_change": speed_change,
    }


def extract_enhanced_video_features(file_path: str, max_frames: int = 5) -> dict:
    """
    Extract comprehensive features from video file with CRASH DETECTION FOCUS.
    
    ⚡ ULTRA-FAST optimized:
    - 5 frames sampled (optimal for speed + crash detection)
    - Resize to 360p for fastest processing
    - Fast frame differencing (not optical flow)
    - Peak-based crash detection (crashes = spikes)
    - ~1-2 seconds processing time
    
    Args:
        file_path: Path to video file
        max_frames: Number of frames to sample
    
    Returns:
        Dictionary with aggregated features including crash detection
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {
            "speed_score": 0.0, "density_score": 0.0, "distance_score": 100.0,
            "motion_variance": 0.0, "speed_change": 0.0,
            "time_of_day": 12.0, "weather": 0, "confidence": 0.0
        }

    speeds = []
    densities = []
    distances = []
    motion_vars = []
    speed_changes = []
    impact_scores = []  # NEW: Track sudden impacts
    
    prev_frame = None
    prev_gray = None
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, fc // max_frames)

    for i in range(0, fc, skip):
        if len(speeds) >= max_frames:  # Stop if we have enough frames
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # ⚡ Resize to 360p for ULTRA-FAST processing
        h, w = frame.shape[:2]
        if h > 360:
            scale = 360.0 / h
            frame = cv2.resize(frame, (int(w * scale), 360))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ⚡ Fast motion detection
        if prev_gray is not None:
            # Frame difference captures motion/impact
            diff = cv2.absdiff(gray, prev_gray)
            speed = float(np.mean(diff))
            motion_variance = float(np.std(diff))
            
            # Speed change from frame difference
            if prev_frame is not None:
                prev_diff = cv2.absdiff(prev_gray, prev_frame)
                speed_change = float(np.abs(np.mean(diff) - np.mean(prev_diff)))
            else:
                speed_change = 0.0
            
            # NEW: Impact score (high motion + high variance = crash)
            impact_score = (np.mean(diff) + np.std(diff)) / 2
            impact_scores.append(float(impact_score))
        else:
            speed = 25.0
            motion_variance = 5.0
            speed_change = 0.0

        # ⚡ Single Canny pass (faster than multi-scale)
        edges = cv2.Canny(gray, 60, 150)
        density = float(np.mean(edges > 0) * 200)
        distance = max(5.0, 60.0 - density * 0.2)

        speeds.append(speed)
        densities.append(density)
        distances.append(distance)
        motion_vars.append(motion_variance)
        speed_changes.append(speed_change)
        
        prev_frame = gray
        prev_gray = gray

    cap.release()

    if len(speeds) == 0:
        return {
            "speed_score": 0.0, "density_score": 0.0, "distance_score": 100.0,
            "motion_variance": 0.0, "speed_change": 0.0,
            "time_of_day": 12.0, "weather": 0, "confidence": 0.0
        }

    # Use PEAK values (crashes are sudden spikes) not just averages
    max_speed = float(np.max(speeds))
    max_density = float(np.max(densities))
    max_motion_var = float(np.max(motion_vars))
    max_speed_change = float(np.max(speed_changes))
    max_impact = float(np.max(impact_scores)) if impact_scores else 0.0
    
    # Also compute averages for baseline
    avg_speed = float(np.mean(speeds))
    avg_density = float(np.mean(densities))
    avg_motion_var = float(np.mean(motion_vars))
    avg_speed_change = float(np.mean(speed_changes))
    
    # CRASH DETECTION: Use max values weighted with averages
    # If peak motion/impact >> average, it's likely a crash event
    # Lower thresholds for better sensitivity
    crash_indicator = (max_impact > 40.0) or (max_speed_change > 35.0) or (max_speed > 45.0)
    
    if crash_indicator:
        # Boost values for crash videos (they show spikes)
        final_speed = max_speed * 0.8 + avg_speed * 0.2 + 15.0  # Add base boost
        final_density = max_density * 0.8 + avg_density * 0.2 + 10.0
        final_motion_var = max_motion_var * 0.9 + 10.0
        final_speed_change = max_speed_change * 0.95 + 8.0
    else:
        # Use averages for normal traffic
        final_speed = avg_speed
        final_density = avg_density
        final_motion_var = avg_motion_var
        final_speed_change = avg_speed_change
    
    # Confidence: based on frame count and impact detection
    confidence = min(1.0, len(speeds) / max_frames * 0.95)
    
    # Context features
    time_of_day = 12.0

    return {
        "speed_score": final_speed,
        "density_score": final_density,
        "distance_score": max(5.0, 60.0 - final_density * 0.2),
        "motion_variance": final_motion_var,
        "speed_change": final_speed_change,
        "time_of_day": time_of_day,
        "weather": 0,
        "confidence": confidence,
        "crash_detected": crash_indicator  # NEW: Flag for crash video
    }
