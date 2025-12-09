"""
app.py - Unified AI-Based Smart Transport Dashboard (Single-file Streamlit app)

Requirements:
    pip install streamlit plotly pandas numpy
    # For video features (optional): pip install opencv-python

Run:
    streamlit run app.py

This file is intentionally self-contained and uses dummy/simulated data so it runs
without any external backend. Replace TODO comments with actual ML/RL model calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import io
import tempfile
import random
from datetime import datetime

# Optional: OpenCV for video frame extraction.
# If you don't have OpenCV installed, the app still runs; video processing will be disabled.
try:
    import cv2  # pip install opencv-python
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# ---------------------------
# Page configuration & style
# ---------------------------
st.set_page_config(layout="wide", page_title="Smart Transport Dashboard")
st.title("Unified AI-Based Smart Transport Dashboard")

# Global in-session state for logs
if "logs" not in st.session_state:
    st.session_state.logs = []

def log_event(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.insert(0, f"{timestamp} : {msg}")

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Home / Overview", "RL Lane-Changing", "Accident Prediction", "Emergency Routing", "Logs & Debug")
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation controls**")
SIM_SPEED = st.sidebar.slider("Simulation speed (affects generated time series)", 0.1, 2.0, 1.0, 0.1)
NUM_TSTEPS = st.sidebar.slider("Time series length", 30, 200, 60, 10)

# ---------------------------
# Helper data-generation
# ---------------------------
def generate_traffic_timeseries(n=60, base_density=50, variation=10):
    t = np.arange(n)
    noise = np.random.randn(n) * variation
    series = np.clip(base_density + np.sin(t / 6.0) * (variation * 0.8) + noise, 0, 200)
    return series

def generate_speed_timeseries(n=60, base_speed=60, variation=8):
    t = np.arange(n)
    noise = np.random.randn(n) * variation
    speed = np.clip(base_speed + np.cos(t / 8.0) * (variation * 0.7) + noise, 0, 140)
    return speed

# ---------------------------
# Shared layout functions
# ---------------------------
def metric_card(col, label, value, delta=None):
    # wrapper to present consistent metric styling
    if delta is not None:
        col.metric(label, value, delta)
    else:
        col.metric(label, value)

# ---------------------------
# PAGE: Home / Overview
# ---------------------------
if page == "Home / Overview":
    st.header("Overview")
    st.write(
        """
        **Unified AI-Based Smart Transport System** — dashboard to visualize RL lane-changing decisions,
        accident risk prediction, and emergency vehicle priority routing. All data shown here is simulated
        and intended as placeholders until you plug your real models and data pipelines.
        """
    )

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    total_vehicles = int(np.random.randint(500, 2500))
    avg_speed = round(float(generate_speed_timeseries(1, base_speed=60, variation=8)[0]), 1)
    active_emergency = int(np.random.randint(0, 8))
    high_risk_events = int(np.random.randint(0, 12))

    metric_card(col1, "Total Vehicles", f"{total_vehicles}")
    metric_card(col2, "Average Speed (km/h)", f"{avg_speed}")
    metric_card(col3, "Active Emergency Vehicles", f"{active_emergency}")
    metric_card(col4, "Current High-Risk Events", f"{high_risk_events}")

    st.markdown("---")
    # Simulation snapshot placeholder
    st.subheader("Simulation Snapshot")
    st.info("Placeholder: Add your traffic simulator screenshot here. Example:\n\n# st.image('simulation_placeholder.png')")
    st.write("If you have a simulation image, replace the placeholder above with st.image(...)")

    # Combined Plotly line chart
    st.subheader("Traffic Analytics")
    cols = st.columns([3, 1])
    with cols[0]:
        n = NUM_TSTEPS
        density = generate_traffic_timeseries(n=n, base_density=80, variation=15)
        avg_speed_ts = generate_speed_timeseries(n=n, base_speed=60, variation=6)

        df = pd.DataFrame({
            "t": np.arange(n),
            "Traffic Density": density,
            "Average Speed": avg_speed_ts
        })

        fig = px.line(df, x="t", y=["Traffic Density", "Average Speed"],
                      labels={"t": "Time step", "value": "Value", "variable": "Metric"},
                      title="Traffic Density and Average Speed Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        st.subheader("Quick Controls")
        st.write("Adjust simulation parameters from the sidebar.")
        if st.button("Regenerate Overview Data"):
            st.experimental_rerun()

# ---------------------------
# PAGE: RL Lane-Changing
# ---------------------------
elif page == "RL Lane-Changing":
    st.header("RL Lane-Changing Agent")
    st.write(
        """
        This module simulates an RL agent deciding whether to change lanes. Replace the simulated
        parts marked `TODO` with actual environment state and agent outputs.
        """
    )

    # Scenario selection
    scenario = st.selectbox("Scenario", ["Light Traffic", "Moderate Traffic", "Heavy Traffic"])
    if scenario == "Light Traffic":
        density_base = 30
        speed_base = 80
    elif scenario == "Moderate Traffic":
        density_base = 80
        speed_base = 60
    else:  # Heavy Traffic
        density_base = 140
        speed_base = 35

    # Simulated current state
    # TODO: Replace this dummy_state with real RL environment state
    dummy_state = {
        "speed": round(float(np.random.normal(speed_base, 3)), 1),
        "lane_id": int(np.random.choice([1, 2, 3])),
        "front_vehicle_distance": round(abs(np.random.normal(15 if scenario=="Light Traffic" else 8, 4)), 1),
        "rear_vehicle_distance": round(abs(np.random.normal(12 if scenario=="Light Traffic" else 6, 3)), 1),
        "traffic_density": int(np.random.normal(density_base, 8)),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.subheader("Current State")
        st.json(dummy_state)

        # Simulated action (TODO: Replace with actual agent action output)
        possible_actions = ["STAY", "CHANGE_LEFT", "CHANGE_RIGHT"]
        action = np.random.choice(possible_actions, p=[0.6, 0.2, 0.2])

        # Safety status logic
        # Green for safe, Yellow for caution, Red for risky
        safety = "Safe"
        safety_color = "green"
        risk_level = 0.0
        # simple heuristic for dummy safety
        if dummy_state["front_vehicle_distance"] < 5 or dummy_state["rear_vehicle_distance"] < 4:
            safety = "Risky"
            safety_color = "red"
            risk_level = 0.9
        elif dummy_state["front_vehicle_distance"] < 10:
            safety = "Caution"
            safety_color = "orange"
            risk_level = 0.5
        else:
            safety = "Safe"
            safety_color = "green"
            risk_level = 0.1

        st.markdown("### Selected Action")
        st.markdown(f"<div style='font-size:22px;padding:8px;border-radius:6px;background-color:#f0f2f6'>**Action: {action.replace('_',' ')}**</div>", unsafe_allow_html=True)

        st.markdown("### Safety Status")
        if safety == "Safe":
            st.success("Safe")
        elif safety == "Caution":
            st.warning("Caution")
        else:
            st.error("Risky")

    with right_col:
        st.subheader("RL Training Curve (Simulated)")
        # Simulate episode reward curve (50-100 episodes)
        num_episodes = st.slider("Episodes", 50, 200, 100)
        # smooth cumulative-like rewards
        rand = np.random.randn(num_episodes) * (2.0 / SIM_SPEED)
        rewards = np.cumsum(rand) + np.linspace(0, 50, num_episodes)
        df_rewards = pd.DataFrame({"episode": np.arange(1, num_episodes + 1), "reward": rewards})
        fig_r = px.line(df_rewards, x="episode", y="reward", title="Episode Reward vs Episode Number")
        fig_r.update_layout(hovermode="x unified")
        st.plotly_chart(fig_r, use_container_width=True)

        st.subheader("Agent Debugging")
        # Action probabilities (dummy)
        # TODO: Replace this random action_probs with actual agent.policy(state)
        action_probs = np.random.dirichlet(np.ones(len(possible_actions)), size=1)[0]
        prob_df = pd.DataFrame({
            "action": possible_actions,
            "probability": np.round(action_probs, 3)
        })
        st.table(prob_df)

        # Provide ability to print states/actions for debugging
        if st.button("Print current debug info"):
            st.write("STATE ->", dummy_state)
            st.write("ACTION ->", action)
            st.write("ACTION PROBS ->", action_probs)
            log_event(f"DEBUG: Printed RL state/action for scenario={scenario}")

    st.markdown("---")
    # Video integration for RL Lane-Changing
    st.subheader("Video Integration — RL Lane-Changing")
    st.write("Upload a highway/road clip to visualize frames and dummy lane/RL overlays.")
    rl_video_file = st.file_uploader("Upload video for Lane-Changing (mp4/avi)", type=["mp4", "avi"], key="rl_video")

    if rl_video_file is not None:
        log_event("Loaded video for Lane-Changing module")
        # We can display the uploaded video directly
        st.video(rl_video_file)

        # Extract frames using OpenCV if available
        if OPENCV_AVAILABLE:
            tcol1, tcol2 = st.columns([1, 2])
            with tcol1:
                st.write("Extracted Frames (sampled)")
            with tcol2:
                st.write("Frame overlays (dummy values)")

            # Save to a temp file so cv2 can open it
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(rl_video_file.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            log_event(f"Loaded video for Lane-Changing module ({total_frames} frames, fps={fps})")

            sample_frames = []
            # Read first 5 frames or every Nth frame if many
            sample_n = min(5, total_frames) if total_frames > 0 else 5
            step = max(1, total_frames // sample_n) if total_frames > 0 else 1

            extracted = 0
            frame_index = 0
            frames_processed = 0
            while extracted < sample_n and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index % step == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sample_frames.append((frame_index, frame_rgb))
                    extracted += 1
                frame_index += 1
                frames_processed += 1
            cap.release()

            # Display frames with dummy overlays
            cols = st.columns(len(sample_frames) or 1)
            for (idx, frm), c in zip(sample_frames, cols):
                c.image(frm, caption=f"Frame #{idx}", use_column_width=True)
                # Dummy overlay info beneath
                detected_lane = np.random.choice([1, 2, 3])
                front_dist = round(np.random.uniform(5, 30), 1)
                rl_action = np.random.choice(["Change Left", "Stay", "Change Right"])
                c.markdown(f"**Detected lane:** {detected_lane}")
                c.markdown(f"**Front vehicle distance:** {front_dist} m")
                c.markdown(f"**RL Action (simulated):** {rl_action}")
                log_event(f"Processed frame #{idx} for RL Lane-Changing")
        else:
            st.warning("OpenCV not available. Install opencv-python to enable frame extraction and processing.")
            st.info("You can still view the uploaded video above.")

# ---------------------------
# PAGE: Accident Prediction
# ---------------------------
elif page == "Accident Prediction":
    st.header("Accident Risk Prediction")
    st.write(
        """
        This module simulates accident risk scoring over time and per-frame (if a video is uploaded).
        Replace the simulated risk values with calls to your trained model (e.g., model.predict(trajectory)).
        """
    )

    # Generate base risk timeseries
    n = NUM_TSTEPS
    base_risk_noise = np.abs(np.random.randn(n)) * 0.08
    base_risk = np.clip(np.sin(np.linspace(0, 6.0, n)) * 0.2 + 0.3 + base_risk_noise, 0, 1)

    st.subheader("Current Risk Score")
    current_risk = float(np.round(base_risk[-1], 3))
    st.metric("Current Risk Score (0-1)", f"{current_risk}")

    if current_risk < 0.3:
        st.success("Safe")
    elif current_risk < 0.7:
        st.warning("Moderate Risk")
    else:
        st.error("HIGH RISK – Possible Collision!")

    # Risk vs Time plot
    df_risk = pd.DataFrame({"t": np.arange(n), "risk": base_risk})
    fig_risk = px.line(df_risk, x="t", y="risk", title="Risk vs Time")
    fig_risk.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_risk, use_container_width=True)

    # Recent events table (dummy)
    st.subheader("Recent Events")
    num_events = 7
    times = [(datetime.now() - pd.to_timedelta(i, unit='m')).strftime("%H:%M:%S") for i in range(num_events)]
    vehicle_ids = [f"V-{1000 + i}" for i in range(num_events)]
    risk_scores = np.round(np.random.rand(num_events), 3)
    recommended = [np.random.choice(["Brake", "Change Lane Left", "Change Lane Right", "Maintain"]) for _ in range(num_events)]
    df_events = pd.DataFrame({
        "Time": times,
        "Vehicle ID": vehicle_ids,
        "Risk Score": risk_scores,
        "Recommended Action": recommended
    })
    st.table(df_events)

    # TODO: Replace random risk values with model.predict(trajectory)

    st.markdown("---")
    # Video integration for Accident Prediction
    st.subheader("Video Integration — Accident Prediction")
    st.write("Upload a traffic/crossroad video to generate per-frame simulated risk scores.")
    ap_video_file = st.file_uploader("Upload video for Accident Prediction (mp4/avi)", type=["mp4", "avi"], key="ap_video")

    if ap_video_file is not None:
        log_event("Loaded video for Accident Prediction module")
        st.video(ap_video_file)

        if OPENCV_AVAILABLE:
            # Save to temp file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(ap_video_file.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            log_event(f"Loaded video for Accident Prediction ({total_frames} frames, fps={fps})")

            sampled_frames = []
            sample_n = min(10, total_frames) if total_frames > 0 else 10
            step = max(1, total_frames // sample_n) if total_frames > 0 else 1
            idx = 0
            extracted = 0
            simulated_frame_risks = []
            while extracted < sample_n and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Simulate a risk score for this frame
                    frame_risk = float(np.clip(np.random.rand() * 0.9, 0, 1))
                    sampled_frames.append((idx, frame_rgb, frame_risk))
                    simulated_frame_risks.append(frame_risk)
                    extracted += 1
                    log_event(f"Processed frame #{idx} for Accident Prediction (sim risk={frame_risk:.3f})")
                idx += 1
            cap.release()

            # Show frames + table with simulated risk
            cols = st.columns([2, 1])
            with cols[0]:
                st.write("Sampled Frames")
                frame_cols = st.columns(len(sampled_frames) or 1)
                for (fr_idx, frm, fr_risk), c in zip(sampled_frames, frame_cols):
                    c.image(frm, caption=f"Frame #{fr_idx}", use_column_width=True)
                    c.markdown(f"**Simulated risk:** {fr_risk:.3f}")
            with cols[1]:
                st.write("Frame Risk Table")
                fdf = pd.DataFrame([
                    {"Frame #": fr_idx, "Simulated Risk Score": round(fr_risk, 3),
                     "Risk Level": ("Safe" if fr_risk < 0.3 else "Moderate" if fr_risk < 0.7 else "High")}
                    for fr_idx, _, fr_risk in sampled_frames
                ])
                st.table(fdf)

                # Plot Risk vs Frame Number (from sampled frames)
                if len(simulated_frame_risks) > 0:
                    fig_ap = px.line(x=[s[0] for s in sampled_frames], y=[s[2] for s in sampled_frames],
                                     labels={"x": "Frame #", "y": "Simulated Risk"},
                                     title="Simulated Risk vs Frame #")
                    fig_ap.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_ap, use_container_width=True)
        else:
            st.warning("OpenCV not available. Install opencv-python to extract frames and compute per-frame risk.")
            st.info("You can still view the uploaded video above.")

# ---------------------------
# PAGE: Emergency Routing
# ---------------------------
elif page == "Emergency Routing":
    st.header("Emergency Vehicle Routing System")
    st.write(
        """
        Simulated emergency routing outputs. Replace dummy routing computations with your path planning module (A*, Dijkstra, or ML-based).
        """
    )

    # Simulated path choices
    dummy_paths = [
        ["Junction A", "Junction C", "Junction F", "Hospital"],
        ["Entrance", "Roundabout", "Central", "Hospital"],
        ["Junction 12", "Junction 9", "Junction 4", "Hospital"]
    ]
    # Use Python's random.choice to avoid numpy's multidimensional choice issue
    path = random.choice(dummy_paths)
    path_display = " → ".join(path)

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Planned Route")
        st.markdown(f"**Route:** {path_display}")
        # Simple network-like scatter using plotly
        nodes = [{"id": i, "label": p} for i, p in enumerate(path)]
        node_x = list(range(len(nodes)))
        node_y = [0] * len(nodes)
        net_df = pd.DataFrame({
            "x": node_x,
            "y": node_y,
            "label": [n["label"] for n in nodes]
        })
        fig_route = px.scatter(net_df, x="x", y="y", text="label", title="Route Visualization (simple)")
        fig_route.update_traces(textposition="bottom center")
        st.plotly_chart(fig_route, use_container_width=True)

    with cols[1]:
        st.subheader("ETA Metrics")
        # Base ETA
        base_eta = round(np.random.uniform(4.0, 12.0), 2)  # without priority
        with_priority_eta = round(base_eta * np.random.uniform(0.4, 0.7), 2)  # with priority
        st.metric("Estimated Arrival Time (with priority) (min)", f"{with_priority_eta}")
        st.markdown(f"*Without priority: ~{base_eta} min*")

        # Bar chart comparing times
        df_eta = pd.DataFrame({
            "Mode": ["Without Priority", "With Priority"],
            "Travel Time (min)": [base_eta, with_priority_eta]
        })
        fig_eta = px.bar(df_eta, x="Mode", y="Travel Time (min)",
                         title="Travel Time: Without Priority vs With Priority")
        st.plotly_chart(fig_eta, use_container_width=True)

    st.markdown("---")
    # Video integration for Emergency Routing
    st.subheader("Video Integration — Emergency Routing")
    st.write("Upload an ambulance-in-traffic video to simulate detection and ETA adjustment.")
    er_video_file = st.file_uploader("Upload video for Emergency Routing (mp4/avi)", type=["mp4", "avi"], key="er_video")

    adjusted_eta = with_priority_eta
    detected_ambulance = False
    traffic_density_label = "Medium"

    if er_video_file is not None:
        log_event("Loaded video for Emergency Routing module")
        st.video(er_video_file)

        if OPENCV_AVAILABLE:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(er_video_file.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            log_event(f"Loaded video for Emergency Routing ({total_frames} frames, fps={fps})")

            # Sample a few frames to infer dummy boolean and density
            sample_n = min(6, total_frames) if total_frames > 0 else 6
            step = max(1, total_frames // sample_n) if total_frames > 0 else 1
            idx = 0
            ambulance_votes = 0
            density_vals = []
            extracted = 0
            while extracted < sample_n and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    # Dummy detection: random boolean per frame
                    amb_detected = np.random.choice([0, 1], p=[0.7, 0.3])
                    ambulance_votes += amb_detected
                    # Dummy traffic density metric
                    density_vals.append(np.random.choice([0.2, 0.5, 0.8], p=[0.5, 0.3, 0.2]))
                    extracted += 1
                    log_event(f"Processed frame #{idx} for Emergency Routing (amb_detect={amb_detected})")
                idx += 1
            cap.release()

            detected_ambulance = ambulance_votes > (0.4 * sample_n)
            avg_density = np.mean(density_vals) if len(density_vals) > 0 else 0.5
            if avg_density < 0.35:
                traffic_density_label = "Low"
            elif avg_density < 0.65:
                traffic_density_label = "Medium"
            else:
                traffic_density_label = "High"

            # Adjust ETA slightly based on dummy detections
            if detected_ambulance:
                adjusted_eta = round(with_priority_eta * np.random.uniform(0.9, 1.05), 2)
            else:
                # no ambulance detected -> maybe traffic causes delay
                adjusted_eta = round(with_priority_eta * np.random.uniform(1.05, 1.35), 2)

            # Display detection summary
            cols = st.columns(2)
            cols[0].markdown(f"**Ambulance detected:** {'Yes' if detected_ambulance else 'No'}")
            cols[1].markdown(f"**Traffic density:** {traffic_density_label}")
            cols[0].metric("Adjusted ETA (min)", f"{adjusted_eta}")
            # Update bar chart with adjusted ETA
            df_eta2 = pd.DataFrame({
                "Mode": ["Without Priority", "With Priority (orig)", "With Priority (adjusted)"],
                "Travel Time (min)": [base_eta, with_priority_eta, adjusted_eta]
            })
            fig_eta2 = px.bar(df_eta2, x="Mode", y="Travel Time (min)", title="Travel Time Comparison (adjusted)")
            st.plotly_chart(fig_eta2, use_container_width=True)

            log_event(f"Updated ETA based on Emergency Routing video (detected_ambulance={detected_ambulance}, density={traffic_density_label})")
        else:
            st.warning("OpenCV not available. Install opencv-python to enable frame extraction for emergency routing.")
            st.info("You can still view the uploaded video above.")

    # TODO: Replace this dummy path with output of A* / Dijkstra
    st.info("Note: Replace dummy path and ETA with outputs from your routing algorithm (A*, Dijkstra, RL planner, etc.)")

# ---------------------------
# PAGE: Logs & Debug
# ---------------------------
elif page == "Logs & Debug":
    st.header("System Logs & Debug View")
    st.write(
        """
        Use this page to inspect runtime logs and debug arrays/states. Toggle debug mode to reveal
        additional raw state vectors and dummy probabilities.
        """
    )

    debug_mode = st.checkbox("Enable debug mode", value=False)

    # Show logs
    st.subheader("Event Logs")
    if len(st.session_state.logs) == 0:
        st.info("No events logged yet. Logs will appear here as you interact with the dashboard (e.g., upload videos, process frames).")
    else:
        # Show as code block for easy copying
        log_block = "\n".join(st.session_state.logs[:200])
        st.code(log_block)

    st.markdown("---")
    st.subheader("Quick Debug Controls")
    if st.button("Generate Dummy Debug Entry"):
        # Simulate a log entry
        t = round(time.time() % 1000, 2)
        entry = f"t={t}s : Lane change RIGHT (Safe)"
        log_event(entry)
        st.experimental_rerun()

    # Dummy raw state vectors and action summary charts
    st.subheader("Action & Risk Summaries")
    # Actions taken pie (dummy)
    actions = ["Stay", "Left", "Right"]
    action_counts = np.random.randint(10, 200, size=len(actions))
    df_actions = pd.DataFrame({"action": actions, "count": action_counts})
    fig_actions = px.pie(df_actions, names="action", values="count", title="Actions Taken Distribution")
    st.plotly_chart(fig_actions, use_container_width=True)

    # Risk categories bar chart
    risk_cats = ["Safe", "Moderate", "High"]
    risk_counts = np.random.randint(5, 50, size=3)
    df_riskcats = pd.DataFrame({"risk": risk_cats, "count": risk_counts})
    fig_riskcats = px.bar(df_riskcats, x="risk", y="count", title="Risk Categories Summary")
    st.plotly_chart(fig_riskcats, use_container_width=True)

    if debug_mode:
        st.markdown("### Raw State Vectors (sample)")
        sample_vector = np.round(np.random.randn(12), 3).tolist()
        st.code(f"state_vector = {sample_vector}")

        st.markdown("### Action Probabilities (sample)")
        sample_probs = np.round(np.random.dirichlet(np.ones(3)), 3).tolist()
        st.code(f"action_probs = {sample_probs}")

        st.markdown("### Recent Logs (table view)")
        df_logs = pd.DataFrame({"log": st.session_state.logs[:50]})
        st.table(df_logs)

# ---------------------------
# End of pages
# ---------------------------

# Footer
st.markdown("---")
st.caption("Dashboard demo — all values are simulated. Replace TODO items with your ML/RL model outputs and processing pipelines.")
