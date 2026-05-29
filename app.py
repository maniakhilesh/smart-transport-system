# ============================================================
# app.py
# Unified AI-Based Smart Transport Intelligence Platform
# Professional, Stable, Single-File Dashboard
# ============================================================

import os
import time
import json
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import osmnx as ox
import folium
from streamlit_folium import st_folium
from collections import deque
from traffic_provider import get_live_traffic_snapshot
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_autorefresh import st_autorefresh

import requests
def fetch_live_metrics():
    """
    Simulated live metrics (no expensive API calls)
    """
    rng = np.random.default_rng()
    return {
        "vehicles": int(rng.integers(2500, 3200)),
        "avg_speed": round(rng.uniform(35, 75), 1),
        "risk_alerts": int(rng.integers(0, 5)),
        "emergency_units": int(rng.integers(0, 4)),
    }

# ⚡ OPTIMIZED: Removed expensive TomTom API call (use traffic_provider instead)
# Only compute metrics once per session in init block below


# --------------------------------------------------
# AUTO REFRESH (HARD GUARANTEED) — Every 5 seconds for live updates
# --------------------------------------------------
st_autorefresh(
    interval=5 * 1000,   # 5 seconds for dynamic metric refreshes
    limit=None,          # infinite
    key="global_refresh"
)
# --------------------------------------------------
# LIVE METRICS — Update every refresh for dynamic dashboard
# --------------------------------------------------
st.session_state.live_metrics = fetch_live_metrics()

@st.cache_resource(show_spinner=False)
def load_real_road_graph(lat, lon, dist):
    return ox.graph_from_point(
        (lat, lon),
        dist=dist,
        network_type="drive",
        simplify=True
    )

def interpolate_points(p1, p2, steps=15):
    xs = np.linspace(p1[0], p2[0], steps)
    ys = np.linspace(p1[1], p2[1], steps)
    return list(zip(xs, ys))
ox.settings.use_cache = True
ox.settings.log_console = False


# Optional OpenCV (video processing)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# ------------------------------------------------------------
# LOCAL PROJECT IMPORTS (SAFE, NO PACKAGES)
# ------------------------------------------------------------
from inference import init_models, predict_risk, predict_rl_action
from routing import astar_route, dijkstra_route, load_sample_graph
from video_utils import sample_frames_from_file

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Unified AI-Based Smart Transport",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STATE
# ============================================================
if "logs" not in st.session_state:
    st.session_state.logs = []

if "regen_ts" not in st.session_state:
    st.session_state.regen_ts = time.time()

if "last_eta" not in st.session_state:
    st.session_state.last_eta = None

if "risk_buffer" not in st.session_state:
    st.session_state.risk_buffer = deque(maxlen=20)

if "ambulance_step" not in st.session_state:
    st.session_state.ambulance_step = 0

if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = {
        "vehicles": 2500,
        "avg_speed": 50.0,
        "risk_alerts": 2,
        "emergency_units": 1,
    }



def log_event(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.insert(0, f"{ts} | {msg}")


# ------------------------------------------------------------
# LOAD MODELS (CACHED)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    return init_models()

models = load_models()
acc_model = models.get("acc_model")
rl_model = models.get("rl_model")

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("🚦 Smart Transport Control")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "RL Lane Changing",
        "Accident Prediction",
        "Emergency Routing",
        "Logs & Debug",
    ],
)

st.sidebar.markdown("---")

SIM_SPEED = st.sidebar.slider("Simulation speed", 0.2, 2.0, 1.0, 0.1)
NUM_STEPS = st.sidebar.slider("Timeline length", 30, 200, 60, 10)

if st.sidebar.button("🔄 Regenerate Simulation"):
    st.session_state.regen_ts = time.time()
    log_event("Simulation regenerated")

traffic = get_live_traffic_snapshot()

if traffic["source"] == "paid":
    st.sidebar.success("🌍 Live Traffic: Enterprise API")
else:
    st.sidebar.info("🟢 Traffic: Video + Open Data Mode")


# ------------------------------------------------------------
# GLOBAL HEADER
# ------------------------------------------------------------
st.title("🧠 Unified AI-Based Smart Transport Intelligence Platform")

# ✅ Display live metrics (updated every 5s via autorefresh)
metrics = st.session_state.live_metrics

colA, colB, colC, colD = st.columns(4)

colA.metric("Vehicles Monitored", metrics["vehicles"])
colB.metric("Avg Speed (km/h)", metrics["avg_speed"])
colC.metric("Active Risk Alerts", metrics["risk_alerts"])
colD.metric("Emergency Units", metrics["emergency_units"])

st.caption(f"🔄 Updated at {datetime.now().strftime('%H:%M:%S')}")


st.markdown("---")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.subheader("System Overview")

    st.write(
        """
        This dashboard represents a **unified intelligent transport decision platform**
        integrating **reinforcement learning**, **accident risk prediction**, and
        **graph-based emergency routing**.

        Video streams act as **sensor proxies**, feeding perception data into AI
        decision and optimization modules in real time.
        """
    )
    rng = np.random.default_rng(int(st.session_state.regen_ts))

    t = np.arange(NUM_STEPS)

    rng = np.random.default_rng(int(st.session_state.regen_ts))
    density = np.clip(
            60 + 20 * np.sin(t / 8) + rng.normal(0, 6, NUM_STEPS),
            0, 200
        )

    speed = np.clip(
            70 - density * 0.25 + rng.normal(0, 4, NUM_STEPS),
            5, 120
        )
    df = pd.DataFrame({"Time": t, "Traffic Density": density, "Average Speed": speed})
    fig = px.line(df, x="Time", y=["Traffic Density", "Average Speed"])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 — RL LANE CHANGING
# ============================================================
elif page == "RL Lane Changing":
    if "last_rl_update" not in st.session_state:
        st.session_state.last_rl_update = 0
    
    st.subheader("🚗 Reinforcement Learning — Lane Decision System")

    st.write(
        """
        This module demonstrates **reinforcement learning–based lane-changing**
        using **real video-derived features**.

        Video → Perception → State → RL Policy → Action → Reward → Visualization
        """
    )


    # --------------------------------------------------
    # RNG + Agent Init
    # --------------------------------------------------
    seed = int(st.session_state.regen_ts * 1000) % (2**31 - 1)
    rng = np.random.RandomState(seed)

    from rl.q_agent import QAgent
    from video_features import extract_video_features

    agent = QAgent()

    possible_actions = ["STAY", "LEFT", "RIGHT"]

    # --------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------
    st.markdown("### 🎥 Video Simulation Input")
    video = st.file_uploader(
        "Upload traffic video (mp4 / avi)",
        type=["mp4", "avi"],
        key="rl_video_manual",
    )

    if video:
        st.video(video)

        # Save uploaded file to temporary location for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video.read())
            tmp_video_path = tmp_file.name

        # ----------------------------------------------
        # FRAME EXTRACTION
        # ----------------------------------------------
        try:
            frames = sample_frames_from_file(tmp_video_path, sample_n=5)
        except Exception as e:
            st.warning(f"Video processing failed: {e}")
            frames = []
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)

        if len(frames) == 0:
            st.warning("No frames extracted from video.")
        else:
            prev_frame = None

            st.markdown("### 🚦 RL Decision Over Video Frames")

            # ------------------------------------------
            # PROCESS EACH FRAME
            # ------------------------------------------
            for idx, frame in frames:
                # --------------------------------------
                # REAL VIDEO FEATURE EXTRACTION
                # --------------------------------------
                features = extract_video_features(frame, prev_frame)
                prev_frame = frame

                dummy_state = {
                    "speed": features["speed"],
                    "traffic_density": features["density"],
                    "front_vehicle_distance": features["front_distance"],
                    "lane_id": rng.choice([1, 2, 3]),
                }
                # 🔁 Real-time RL state update
                if time.time() - st.session_state.last_rl_update > 1.5:
                    st.session_state.last_rl_update = time.time()

                    st.session_state.realtime_rl_state = {
                        "speed": np.random.uniform(30, 90),
                        "density": np.random.uniform(20, 150),
                        "front_dist": np.random.uniform(3, 30),
                    }
                traffic = get_live_traffic_snapshot()

                # Light influence only (does not override RL)
                dummy_state["traffic_density"] *= (1 + traffic["congestion_level"])


                # --------------------------------------
                # RL AGENT DECISION
                # --------------------------------------
                action_idx, action_label, action_probs = agent.predict(
                    speed=dummy_state["speed"],
                    lane=dummy_state["lane_id"] - 1,  # 0-based
                    front_dist=dummy_state["front_vehicle_distance"],
                    density=dummy_state["traffic_density"],
                )
                state = st.session_state.realtime_rl_state

                st.metric("Speed (km/h)", f"{state['speed']:.1f}")
                st.metric("Traffic Density", f"{state['density']:.1f}")
                st.metric("Front Gap (m)", f"{state['front_dist']:.1f}")


                # --------------------------------------
                # REWARD SIMULATION (UI-ONLY)
                # --------------------------------------
                front_dist = dummy_state["front_vehicle_distance"]
                lane_id = dummy_state["lane_id"]

                if front_dist < 6:
                    reward = -1.0
                elif action_label == "STAY":
                    reward = 0.5
                elif action_label in ["LEFT", "RIGHT"]:
                    reward = 0.3
                else:
                    reward = -0.2

                if action_label == "LEFT" and lane_id == 1:
                    reward -= 0.5
                if action_label == "RIGHT" and lane_id == 3:
                    reward -= 0.5

                # --------------------------------------
                # VISUAL OVERLAY
                # --------------------------------------
                overlay = frame.copy()
                h, w, _ = overlay.shape

                cv2.putText(
                    overlay, f"Action: {action_label}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2
                )

                cv2.putText(
                    overlay, f"Speed: {features['speed']:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2
                )

                cv2.putText(
                    overlay, f"Density: {features['density']:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2
                )

                cv2.putText(
                    overlay, f"Front Gap: {features['front_distance']:.1f} m",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 2
                )

                st.image(overlay, caption=f"Frame {idx}")

            # ------------------------------------------
            # SUMMARY METRICS (LAST FRAME)
            # ------------------------------------------
            st.markdown("### 🧠 RL Decision Summary")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Selected Action", action_label)
                st.metric("Reward", round(reward, 2))

            with col2:
                prob_df = pd.DataFrame({
                    "Action": possible_actions,
                    "Confidence": action_probs,
                })
                st.bar_chart(prob_df.set_index("Action"))

            # ------------------------------------------
            # VIDEO INFLUENCE EXPLANATION
            # ------------------------------------------
            st.info(
                "🎥 Video input directly influences RL perception via "
                "speed, density, and front-gap estimation per frame."
            )

    else:
        st.info("Upload a traffic video to activate RL-based lane decisions.")

# ============================================================
# PAGE 3 — ACCIDENT PREDICTION (VIDEO-DOMINANT, FIXED)
# ============================================================
elif page == "Accident Prediction":
    st.subheader("⚠️ Accident Risk Prediction")

    st.write(
        """
        **🎯 IMPROVED: Enhanced ML + Multi-Scale Video Analysis**

        The AI analyzes:
        • Motion patterns & instability (optical flow + variance)
        • Multi-scale edge detection (vehicles + lane markings)
        • Temporal features (speed changes, acceleration)
        • Traffic density & proximity
        • Advanced RandomForest (150 trees, 7 features)
        • Confidence scoring for reliability
        """
    )

    # --------------------------------------------------
    # BASELINE (REFERENCE ONLY)
    # --------------------------------------------------
    st.markdown("### 🔢 Enhanced Model - Baseline Test")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        base_speed = st.slider("Speed (km/h)", 0, 150, 65)
    with c2:
        base_density = st.slider("Traffic Density", 0, 200, 45)
    with c3:
        base_distance = st.slider("Front Distance (m)", 0, 100, 15)
    with c4:
        base_motion_var = st.slider("Motion Instability", 0, 100, 12)

    # Additional baseline features
    base_speed_change = 0.0  # neutral
    base_time = 12.0  # midday

    baseline_risk = predict_risk(acc_model, base_speed, base_density, base_distance,
                                 base_motion_var, base_speed_change, base_time, 0)
    if baseline_risk is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline ML Risk", f"{baseline_risk:.2f}")
        with col2:
            risk_pct = baseline_risk * 100
            status = "🟢 Safe" if risk_pct < 35 else "🟡 Caution" if risk_pct < 70 else "🔴 Danger"
            st.metric("Status", status)

    st.markdown("---")

    # --------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------
    st.markdown("### 🎥 Video-Based Accident Detection")

    video = st.file_uploader(
        "Upload traffic video for comprehensive analysis",
        type=["mp4", "avi"],
        key="accident_auto_route_video_v2"
    )

    if not video:
        st.info("⬆️ Upload a video to activate enhanced accident detection.")
        st.stop()

    st.video(video)

    # --------------------------------------------------
    # SAVE UPLOADED FILE TO TEMPORARY LOCATION
    # --------------------------------------------------
    # OpenCV requires a file path, not a file-like object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video.read())
        tmp_video_path = tmp_file.name

    # --------------------------------------------------
    # ENHANCED FRAME EXTRACTION (10 FRAMES)
    # --------------------------------------------------
    from video_features import extract_accident_features, extract_enhanced_video_features

    st.markdown("### 🔬 Extracting Enhanced Video Features...")
    
    with st.spinner("⚡ Fast video analysis (5 frames)..."):
        try:
            # Get comprehensive video-level features (FAST)
            video_features = extract_enhanced_video_features(tmp_video_path, max_frames=5)
            
            # Extract fewer frames for visualization (speed)
            frames = sample_frames_from_file(tmp_video_path, sample_n=4)
        except Exception as e:
            st.error(f"Video processing failed: {e}")
            # Clean up temporary file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
            st.stop()
        finally:
            # Clean up temporary file after processing
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)

    # --------------------------------------------------
    # DISPLAY VIDEO-LEVEL FEATURES
    # --------------------------------------------------
    st.markdown("### 📊 Video Analysis Results")
    st.success("✅ Video analysis complete! (~1-2 seconds)")
    
    # Display crash detection status
    if video_features.get("crash_detected", False):
        st.warning("⚠️ **CRASH PATTERN DETECTED** - Video shows sudden motion/impact indicators typical of accidents")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Speed", f"{video_features['speed_score']:.1f} km/h")
    with col2:
        st.metric("Avg Density", f"{video_features['density_score']:.1f}")
    with col3:
        st.metric("Motion Instability", f"{video_features['motion_variance']:.1f}")
    with col4:
        st.metric("Confidence", f"{video_features['confidence']:.0%}")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Speed Variation", f"{video_features['speed_change']:.1f}")
    with col6:
        st.metric("Avg Distance", f"{video_features['distance_score']:.1f} m")

    # --------------------------------------------------
    # ML PREDICTION WITH ENHANCED FEATURES
    # --------------------------------------------------
    ml_risk = predict_risk(
        acc_model,
        video_features['speed_score'],
        video_features['density_score'],
        video_features['distance_score'],
        video_features['motion_variance'],
        video_features['speed_change'],
        video_features['time_of_day'],
        0
    )

    # --------------------------------------------------
    # FRAME-BY-FRAME ANALYSIS
    # --------------------------------------------------
    prev_frame = None
    frame_risks = []
    frame_ids = []
    max_risk = 0.0
    crash_frame = None
    crash_frame_id = None

    st.markdown("### 🧠 Frame-wise Risk Analysis")

    for idx, frame in frames:
        if prev_frame is None:
            prev_frame = frame
            continue

        # -----------------------------
        # ⚡ FAST VIDEO SIGNALS (no ML per frame)
        # -----------------------------
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_now, gray_prev)
        motion_energy = np.mean(diff) / 255.0

        # Single Canny pass (faster)
        edges = cv2.Canny(gray_now, 60, 150)
        edge_density = np.mean(edges > 0)

        # -----------------------------
        # FAST RISK CALCULATION (no ML call)
        # -----------------------------
        # Direct calculation - much faster than ML per frame
        video_risk = np.clip(
            (motion_energy / 0.35) * 0.6 +
            (edge_density / 0.30) * 0.4,
            0, 1
        )
        
        # Use video-level ML risk as baseline (already computed)
        final_risk = video_risk * 0.55 + ml_risk * 0.45

        # HARD CRASH OVERRIDE (extreme motion + high edges)
        if motion_energy > 0.45 and edge_density > 0.35:
            final_risk = np.clip(0.88 + video_risk * 0.12, 0.88, 1.0)

        frame_risks.append(final_risk)
        frame_ids.append(idx)

        if final_risk > max_risk:
            max_risk = final_risk
            crash_frame = frame.copy()
            crash_frame_id = idx

        # -----------------------------
        # OVERLAY WITH MORE INFO
        # -----------------------------
        overlay = frame.copy()
        color = (0, 255, 0) if final_risk < 0.35 else (0, 165, 255) if final_risk < 0.7 else (0, 0, 255)
        label = "LOW" if final_risk < 0.35 else "MODERATE" if final_risk < 0.7 else "HIGH"

        cv2.rectangle(overlay, (10, 10), (320, 110), color, 2)
        cv2.putText(overlay, f"Risk: {final_risk:.2f} ({label})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay, f"Motion: {motion_energy:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(overlay, f"Edges: {edge_density:.2f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        st.image(overlay, caption=f"Frame {idx}", width=400)
        prev_frame = frame

    # --------------------------------------------------
    # CRASH FRAME FREEZE
    # --------------------------------------------------
    if crash_frame is not None and max_risk >= 0.85:
        st.markdown("## 🚨 HIGH RISK FRAME DETECTED")
        cv2.rectangle(crash_frame, (0, 0), (crash_frame.shape[1]-1, crash_frame.shape[0]-1), (0, 0, 255), 8)
        st.image(crash_frame, caption=f"⚠️ High Risk Frame {crash_frame_id}", use_column_width=True)

    # --------------------------------------------------
    # RISK TIMELINE
    # --------------------------------------------------
    if frame_risks:
        st.session_state.risk_buffer.extend(frame_risks)
        risk_df = pd.DataFrame({
            "Frame": frame_ids,
            "Risk": frame_risks
        })
        st.markdown("### 📈 Risk Timeline")
        st.line_chart(risk_df.set_index("Frame"))

    # --------------------------------------------------
    # FINAL RISK CALCULATION
    # --------------------------------------------------
    # Combine video-level ML prediction with max frame risk
    final_risk = float(max(max_risk, ml_risk * 0.9))

    # External traffic data fusion
    traffic = get_live_traffic_snapshot()
    if traffic["incidents"]:
        final_risk = max(final_risk, 0.92)
    else:
        final_risk = max(final_risk, traffic["congestion_level"] * 0.5)

    st.markdown("### 🚨 Final Accident Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**Final Risk Score**", f"{final_risk:.3f}", 
                 delta=f"{(final_risk - 0.5):.2f}" if final_risk > 0.5 else None)
    with col2:
        risk_category = "🟢 LOW" if final_risk < 0.35 else "🟡 MODERATE" if final_risk < 0.70 else "🔴 HIGH" if final_risk < 0.85 else "🚨 CRITICAL"
        st.metric("Risk Category", risk_category)
    with col3:
        st.metric("Analysis Confidence", f"{video_features['confidence']:.0%}")

    # --------------------------------------------------
    # AUTO EMERGENCY ROUTING TRIGGER
    # --------------------------------------------------
    if final_risk >= 0.85:
        st.session_state["auto_emergency"] = True
        st.session_state["crash_node"] = "A"
        st.session_state["hospital_node"] = "D"
        st.error("🚑 **CRITICAL RISK DETECTED** - Emergency Routing Activated Automatically")
    else:
        st.session_state["auto_emergency"] = False

    # --------------------------------------------------
    # CONFIDENCE GAUGE
    # --------------------------------------------------
    import plotly.graph_objects as go

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=final_risk * 100,
        title={"text": "Risk Level (%)"},
        delta={"reference": 50, "increasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred"},
            "steps": [
                {"range": [0, 35], "color": "lightgreen"},
                {"range": [35, 70], "color": "yellow"},
                {"range": [70, 85], "color": "orange"},
                {"range": [85, 100], "color": "red"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 85
            }
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --------------------------------------------------
    # FEATURE IMPORTANCE VISUALIZATION
    # --------------------------------------------------
    st.markdown("### 📊 Feature Contribution to Risk")
    
    feature_contrib = {
        "Speed": video_features['speed_score'] / 150 * final_risk,
        "Density": video_features['density_score'] / 200 * final_risk,
        "Motion Instability": video_features['motion_variance'] / 100 * final_risk,
        "Speed Variation": video_features['speed_change'] / 30 * final_risk,
        "Proximity": (100 - video_features['distance_score']) / 100 * final_risk,
    }
    
    contrib_df = pd.DataFrame({
        "Feature": list(feature_contrib.keys()),
        "Contribution": list(feature_contrib.values())
    })
    st.bar_chart(contrib_df.set_index("Feature"))

    st.success("✅ Enhanced accident prediction analysis complete!")

# ============================================================
# PAGE 4 — EMERGENCY ROUTING (STABLE & VISUAL)
# ============================================================
elif page == "Emergency Routing":
    # --- SAFE INITIALIZATION ---
    if "route_path" not in st.session_state:
        st.session_state.route_path = None

    if "last_eta" not in st.session_state:
        st.session_state.last_eta = None

    st.session_state.ambulance_step = 0

    st.markdown("### 🗺️ Live Emergency Route Map (Persistent)")
        
    # --------------------------------------------------
    # READ PERSISTENT REAL-MAP ROUTE
    # --------------------------------------------------
    if "real_route" not in st.session_state or "real_graph" not in st.session_state:
        st.info("Compute a real map route to display it here.")
    else:
        route = st.session_state.real_route
        G_map = st.session_state.real_graph

        # Build coordinates from SAME GPS graph used for routing
        route_coords = [
            (G_map.nodes[n]["y"], G_map.nodes[n]["x"])
            for n in route
        ]

        # --------------------------------------------------
        # LEAFLET MAP (PERSISTENT)
        # --------------------------------------------------
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=route_coords[0], zoom_start=13)

        folium.PolyLine(
            route_coords,
            color="red",
            weight=5,
            opacity=0.9,
        ).add_to(m)

        folium.Marker(
            route_coords[0],
            tooltip="🚑 Ambulance Start",
            icon=folium.Icon(color="red"),
        ).add_to(m)

        folium.Marker(
            route_coords[-1],
            tooltip="🏥 Hospital",
            icon=folium.Icon(color="green"),
        ).add_to(m)

        st_folium(m, width=750, height=500)



    st.subheader("🚑 Emergency Vehicle Routing & Response")
    st.markdown("### 🔀 Routing Mode")

    routing_mode = st.radio(
        "Select routing engine",
        ["Demo Graph (Simulation)", "🌍 Real Map (GPS-based)"],
        horizontal=True
    )


    # --------------------------------------------------
    # LOAD GRAPH (ALWAYS)
    # --------------------------------------------------
    graph = load_sample_graph()
    nodes = graph["nodes"]

    import networkx as nx
    import matplotlib.pyplot as plt

    # Build NetworkX graph ONCE
    G = nx.Graph()
    for e in graph["edges"]:
        G.add_edge(e["u"], e["v"], weight=e.get("weight", 1.0))
    # --------------------------------------------------
    # STABLE GRAPH LAYOUT (CRITICAL)
    # --------------------------------------------------
    if "graph_pos" not in st.session_state:
        st.session_state.graph_pos = nx.spring_layout(G, seed=42)

    pos = st.session_state.graph_pos

    path = None

    # --------------------------------------------------
    # AUTO EMERGENCY ROUTING
    # --------------------------------------------------
    auto_mode = st.session_state.get("auto_emergency", False)

    if auto_mode:
        st.info("🚨 Auto-routing active (triggered by accident)")
        crash_node = st.session_state.get("crash_node")
        hospital_node = st.session_state.get("hospital_node")

        if crash_node and hospital_node:
            path, cost = astar_route(graph, crash_node, hospital_node)
            st.session_state.route_path = path
            st.session_state.last_eta = round(cost * 0.6, 2)

            st.success(f"🚑 Auto Route: {' → '.join(path)}")
            st.metric("Auto Emergency ETA (min)", st.session_state.last_eta)

    # --------------------------------------------------
    # MANUAL ROUTING (ALWAYS VISIBLE)
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🧭 Manual Emergency Routing")

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.selectbox("Start Node", nodes, key="manual_start")
    with c2:
        goal = st.selectbox("Goal Node", nodes, key="manual_goal")
    with c3:
        algo = st.selectbox("Algorithm", ["A*", "Dijkstra"], key="manual_algo")

    if st.button("Compute Route", key="manual_route_btn"):
        if algo == "A*":
            path, cost = astar_route(graph, start, goal)
        else:
            path, cost = dijkstra_route(graph, start, goal)

        st.session_state.route_path = path
        traffic = get_live_traffic_snapshot()

        delay_factor = 1.0 + traffic["congestion_level"]
        eta = round(cost * 0.5 * delay_factor, 2)
        st.session_state.last_eta = eta

        st.success(f"✅ Route Computed: {' → '.join(path)}")
        st.metric("ETA (minutes)", eta)

    # --------------------------------------------------
    # 🚑 LIVE AMBULANCE ANIMATION
    # --------------------------------------------------
    st.markdown("### 🚑 Live Ambulance Movement")

    path = st.session_state.get("route_path")

    if path is None or len(path) < 2:
        st.warning("Compute a route to see live animation.")
    else:
        fig_placeholder = st.empty()
        progress = st.progress(0)

        for i in range(len(path) - 1):

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.axis("off")
            ax.set_title("🚑 Emergency Vehicle Navigation")

            # draw base graph
            nx.draw(
                G, pos,
                node_color="#CBD5E1",
                node_size=500,
                edge_color="#94A3B8",
                width=1.5,
                with_labels=True,
                ax=ax
            )

            # highlight completed route
            route_edges = list(zip(path[:i+1], path[1:i+2]))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=route_edges,
                edge_color="red",
                width=3,
                ax=ax
            )

            # ambulance marker
            x, y = pos[path[i+1]]
            ax.text(x, y, "🚑", fontsize=22, ha="center", va="center")

            fig_placeholder.pyplot(fig)
            plt.close(fig)

            progress.progress(int((i + 1) / len(path) * 100))
            time.sleep(0.6)
        st.session_state.ambulance_step += 1
        current_node = path[min(
            st.session_state.ambulance_step,
            len(path)-1
        )]
        st.metric("Current Ambulance Node", current_node)
        st.metric(
            "Remaining Nodes",
            len(path) - st.session_state.ambulance_step
        )



        st.success("🏥 Emergency vehicle reached destination")    

    # --------------------------------------------------
    # 🎥 VIDEO-ASSISTED ETA (ALWAYS VISIBLE)
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎥 Video-Assisted Traffic Awareness")

    route_video = st.file_uploader(
        "Upload traffic video for routing",
        type=["mp4", "avi"],
        key="route_video_panel"
    )

    if route_video:
        st.video(route_video)

        if path is not None and st.session_state.get("last_eta") is not None:
            # Save uploaded file to temporary location for OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(route_video.read())
                tmp_route_video_path = tmp_file.name

            try:
                frames = sample_frames_from_file(tmp_route_video_path, sample_n=4)
                congestion = []

                for idx, frame in frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 80, 160)
                    congestion.append(np.mean(edges > 0))

                factor = np.clip(1.0 + np.mean(congestion) * 1.5, 0.8, 1.8)
                adj_eta = round(st.session_state.last_eta * factor, 2)

                st.metric("Adjusted Emergency ETA (minutes)", adj_eta)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_route_video_path):
                    os.unlink(tmp_route_video_path)

    # --------------------------------------------------
    # Demo graph doesn't have GPS coordinates
    # Real GPS routing is handled separately below
    # --------------------------------------------------
    # ============================================================
    # 🌍 REAL MAP + GPS EMERGENCY ROUTING
    # ============================================================
    if routing_mode == "🌍 Real Map (GPS-based)":

        st.markdown("---")
        st.subheader("🌍 Real Map Emergency Routing (GPS-Based)")

        st.write(
            """
            This mode uses **real road networks** from OpenStreetMap.
            Routing is computed using **Dijkstra / A\*** on actual city roads.
            """
        )

        # -----------------------------
        # CITY SELECTION
        # -----------------------------
        city = st.selectbox(
            "Select City",
            [
                "Bengaluru, India",
                "Hyderabad, India",
                "Chennai, India",
                "Delhi, India",
                "Mumbai, India",
            ],
        )

        # -----------------------------
        # LOAD REAL ROAD NETWORK
        # -----------------------------
        st.markdown("### 📍 Emergency Location (GPS)")

        col1, col2 = st.columns(2)

        with col1:
            start_lat = st.number_input(
                "Start Latitude",
                value=12.9716,
                format="%.6f",
                key="realmap_start_lat"
            )
            start_lon = st.number_input(
                "Start Longitude",
                value=77.5946,
                format="%.6f",
                key="realmap_start_lon"
            )

        with col2:
            dest_lat = st.number_input(
                "Destination Latitude",
                value=12.9750,
                format="%.6f",
                key="realmap_dest_lat"
            )
            dest_lon = st.number_input(
                "Destination Longitude",
                value=77.6030,
                format="%.6f",
                key="realmap_dest_lon"
            )

        # -----------------------------
        # ROUTE COMPUTATION
        # -----------------------------
        if st.button("🚑 Compute Real Map Route"):
            
            with st.spinner("Loading road network and computing route..."):
                try:
                    # Load road network around start point
                    G_map = load_real_road_graph(start_lat, start_lon, 3000)

                    # Find nearest nodes to start and destination
                    orig_node = ox.nearest_nodes(G_map, start_lon, start_lat)
                    dest_node = ox.nearest_nodes(G_map, dest_lon, dest_lat)

                    # Compute route using NetworkX
                    route = nx.shortest_path(
                        G_map,
                        orig_node,
                        dest_node,
                        weight="length",
                        method="dijkstra",
                    )

                    # ✅ STORE ROUTE PERSISTENTLY
                    st.session_state.real_route = route
                    st.session_state.real_graph = G_map

                    st.success("✅ Route computed successfully")

                    # ✅ Extract GPS coordinates from the route nodes
                    route_coords = [
                        (G_map.nodes[n]["y"], G_map.nodes[n]["x"])  # (lat, lon)
                        for n in route
                    ]

                    # Calculate route length
                    route_length = 0.0
                    for u, v in zip(route[:-1], route[1:]):
                        edge_data = G_map.get_edge_data(u, v)
                        if edge_data:
                            # Handle multigraph safely
                            edge = list(edge_data.values())[0]
                            route_length += edge.get("length", 0.0)

                    # Calculate ETA
                    eta = round((route_length / 1000) / 40 * 60, 2)  # 40 km/h avg
                    st.metric("🚑 Estimated ETA (minutes)", eta)
                    
                    # Create interactive map
                    st.markdown("### 🗺️ Live Emergency Route Map")
                    m = folium.Map(
                        location=route_coords[0],
                        zoom_start=13,
                        tiles="OpenStreetMap"
                    )
                    
                    # Add route line
                    folium.PolyLine(
                        route_coords,
                        color="red",
                        weight=5,
                        opacity=0.9
                    ).add_to(m)
                    
                    # Add start marker
                    folium.Marker(
                        route_coords[0],
                        tooltip="🚑 Ambulance Start",
                        icon=folium.Icon(color="red", icon="plus-sign")
                    ).add_to(m)

                    # Add destination marker
                    folium.Marker(
                        route_coords[-1],
                        tooltip="🏥 Hospital",
                        icon=folium.Icon(color="green", icon="info-sign")
                    ).add_to(m)

                    st_folium(m, width=800, height=520)
                    st.success("🏥 Emergency route successfully visualized")
                    
                except Exception as e:
                    st.error(f"Route computation failed: {e}")
                    st.info("💡 Try adjusting the coordinates or selecting a different city region.")

# ============================================================
# PAGE 5 — LOGS
# ============================================================
elif page == "Logs & Debug":
    st.subheader("System Logs & Debug")

    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[:200]))
    else:
        st.info("No logs yet.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Unified AI-Based Smart Transport Intelligence Platform | "
    "Stable Professional Base | Ready for Real Model Integration"
)
