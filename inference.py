"""
inference.py

Model loading, caching and inference wrappers for:
- Accident prediction (scikit-learn RandomForest)
- RL-based lane-changing decision (SB3 or PyTorch fallback)
- Route computation wrapper that calls routing.py functions

Exports:
from inference import load_accident_model, load_rl_model, predict_risk, predict_rl_action, compute_route, init_models
"""
from __future__ import annotations
import os
import logging
import tempfile
from typing import Any, Dict, Optional, Tuple, Sequence
import joblib
import numpy as np

# Streamlit caching utilities (used by app.py)
import streamlit as st

# local routing utilities
from routing import astar_route, dijkstra_route, load_sample_graph



# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@st.cache_resource
def init_models(
    acc_path: str = "models/accident_model.pkl",
    rl_path: str = "models/rl_model.zip",
    rl_algo: str = "DQN",
) -> Dict[str, Any]:
    """
    Load or (if missing) create the accident model and attempt to load an RL model.
    Returns a dict with keys: 'acc_model', 'rl_model'
    
    ⚡ OPTIMIZED: Model training is deferred to background; app loads immediately with None.
    """
    os.makedirs("models", exist_ok=True)
    acc_model = load_accident_model(acc_path)
    
    if acc_model is None:
        logger.warning(
            "⚠️ Accident model not found. Will train in background on first use. "
            "For now, app continues with None (safe fallback)."
        )
        # ✅ DO NOT BLOCK: return None immediately, let user interact
        # Training will happen on first prediction call (lazy load)
    
    rl_model = load_rl_model(rl_path, algo=rl_algo)
    return {"acc_model": acc_model, "rl_model": rl_model}


def load_accident_model(path: str) -> Optional[Any]:
    """
    Load a scikit-learn model saved with joblib. Returns None if not found or load fails.
    """
    try:
        if not os.path.exists(path):
            logger.warning("Accident model path does not exist: %s", path)
            return None
        model = joblib.load(path)
        logger.info("Loaded accident model from %s", path)
        return model
    except Exception as e:
        logger.exception("Failed to load accident model: %s", e)
        return None


def train_and_save_accident_model(path: str, n: int = 2000) -> None:
    """
    Train an IMPROVED RandomForestClassifier with enhanced features:
    Features: speed, density, distance, motion_variance, time_of_day, weather_sim, speed_change
    Label: accident risk derived from realistic traffic accident patterns
    Saves model to `path` using joblib.
    
    🎯 IMPROVED: More realistic data, larger model, better features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import matplotlib.pyplot as plt

    import numpy as np

    rng = np.random.RandomState(42)
    
    # Generate realistic synthetic features
    speed = rng.normal(loc=65, scale=18, size=n).clip(5, 150)  # km/h
    density = rng.normal(loc=35, scale=12, size=n).clip(5, 180)  # vehicles/km
    distance = rng.exponential(scale=18, size=n).clip(1.0, 150)  # meters
    
    # NEW: Additional features for better prediction
    motion_variance = rng.exponential(scale=15, size=n).clip(0, 100)  # motion instability
    speed_change = rng.normal(loc=0, scale=8, size=n).clip(-30, 30)  # acceleration/braking
    time_of_day = rng.uniform(0, 24, size=n)  # hour (night = higher risk)
    weather_sim = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])  # 0=clear, 1=rain, 2=fog
    
    # Night time factor (10pm-6am = higher risk)
    night_factor = np.where((time_of_day >= 22) | (time_of_day <= 6), 1.3, 1.0)
    
    # Weather multiplier
    weather_factor = np.where(weather_sim == 0, 1.0, 
                             np.where(weather_sim == 1, 1.4, 1.6))
    
    X = np.vstack([speed, density, distance, motion_variance, 
                   speed_change, time_of_day, weather_sim]).T

    # IMPROVED: More realistic accident probability model
    # Factors: high speed + high density + short distance + sudden braking + night + bad weather
    logits = (
        -4.5 +  # base threshold
        0.04 * speed +  # speed contribution
        0.06 * density +  # congestion contribution
        -0.12 * distance +  # proximity risk
        0.03 * motion_variance +  # erratic motion
        0.05 * np.abs(speed_change) +  # sudden speed changes
        0.5 * (night_factor - 1) +  # night driving
        0.4 * (weather_factor - 1)  # adverse weather
    )
    
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < probs).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🎯 IMPROVED: Larger, better-tuned model
    model = RandomForestClassifier(
        n_estimators=150,  # More trees for better accuracy
        max_depth=12,  # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    # Ensure directories exist
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    logger.info("✅ IMPROVED Accident model trained: acc=%.3f auc=%.3f (n=%d, trees=%d)", 
                acc, auc, n, 150)
    
    # Feature importance logging
    feature_names = ['speed', 'density', 'distance', 'motion_var', 'speed_change', 'time', 'weather']
    importances = model.feature_importances_
    logger.info("Feature importances: %s", dict(zip(feature_names, importances.round(3))))


def predict_risk(model: Any, speed: float, density: float, distance: float, 
                 motion_variance: float = 10.0, speed_change: float = 0.0,
                 time_of_day: float = 12.0, weather: int = 0) -> float:
    """
    Predict accident risk probability in [0,1] with ENHANCED features.
    
    Args:
        model: Trained RandomForest model
        speed: Vehicle speed (km/h)
        density: Traffic density (vehicles/km)
        distance: Front vehicle distance (m)
        motion_variance: Motion instability measure
        speed_change: Speed change rate (km/h)
        time_of_day: Hour of day (0-24)
        weather: Weather condition (0=clear, 1=rain, 2=fog)
    
    Returns:
        Risk probability in [0, 1]
    """
    try:
        X = np.array([[float(speed), float(density), float(distance), 
                      float(motion_variance), float(speed_change),
                      float(time_of_day), int(weather)]])
    except Exception as e:
        logger.exception("Invalid input to predict_risk: %s", e)
        raise

    if model is None:
        # ✅ Lazy train on first prediction
        if not hasattr(predict_risk, "_trained") or not predict_risk._trained:
            logger.info("⏳ Training IMPROVED accident model on first use...")
            try:
                train_and_save_accident_model("models/accident_model.pkl", n=2000)
                model = load_accident_model("models/accident_model.pkl")
                predict_risk._trained = True
            except Exception as e:
                logger.warning("Model training failed, returning safe default: %s", e)
                return 0.3  # conservative default
        else:
            logger.warning("No accident model available - returning safe risk estimate")
            return 0.3

    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
            prob = max(0.0, min(1.0, prob))
            return prob
        # fallback to predict
        pred = model.predict(X)[0]
        return float(1.0 if pred else 0.0)
    except Exception as e:
        logger.exception("Accident prediction failed: %s", e)
        return 0.3  # safe default


def load_rl_model(path: str, algo: str = "DQN") -> Optional[Any]:
    """
    Attempts to load an RL model.
    - If a Stable Baselines3 model zip is present and SB3 is installed, load it.
    - If a PyTorch .pth is present and torch is installed, load state_dict and return module if possible.
    - If nothing present or loading fails, return a DummyPolicy instance.

    Note: The returned object may be:
      - SB3 model with .predict(obs, deterministic=True)
      - PyTorch nn.Module
      - DummyPolicy with .predict(state)
    """
    # Lazy imports
    try:
        if os.path.exists(path):
            # attempt SB3 load
            try:
                from stable_baselines3.common.base_class import BaseAlgorithm
                from stable_baselines3 import DQN, PPO  # type: ignore
                if path.endswith(".zip"):
                    if algo.upper().startswith("DQN"):
                        model = DQN.load(path)
                        logger.info("Loaded SB3 DQN model from %s", path)
                        return model
                    else:
                        model = PPO.load(path)
                        logger.info("Loaded SB3 PPO model from %s", path)
                        return model
            except Exception as e:
                logger.info("SB3 load failed or SB3 not installed: %s", e)

            # attempt torch
            if path.endswith(".pth") or path.endswith(".pt"):
                try:
                    import torch
                    # We return the raw state_dict for demo; user should replace with their model wrapper.
                    state = torch.load(path, map_location="cpu")
                    logger.info("Loaded PyTorch checkpoint from %s (returned raw object)", path)
                    return state
                except Exception as e:
                    logger.info("Torch load failed: %s", e)
        else:
            logger.info("RL model file does not exist: %s", path)
    except Exception as e:
        logger.exception("Unexpected error in load_rl_model: %s", e)

    logger.info("Returning DummyPolicy as RL model fallback.")
    return DummyPolicy()


def predict_rl_action(rl_model: Any, state: Any) -> Any:
    """
    Given an RL model and a state, return an action index or label.
    Handles SB3 models, PyTorch state_dicts (not runnable), and DummyPolicy.
    """
    if rl_model is None:
        logger.warning("No RL model provided, using DummyPolicy.")
        rl_model = DummyPolicy()

    try:
        # SB3 models have predict method
        if hasattr(rl_model, "predict"):
            # SB3 predict signature: predict(obs, deterministic=True)
            try:
                action, _ = rl_model.predict(state, deterministic=True)  # type: ignore
                return action
            except TypeError:
                action = rl_model.predict(state)  # fallback
                return action

        # PyTorch state dict - not runnable; return deterministic fallback
        import torch

        if isinstance(rl_model, dict) and "state_dict" in rl_model:
            logger.info("RL model is a PyTorch state dict - returning fallback action 0")
            return 0
    except Exception as e:
        logger.exception("Error while predicting RL action: %s", e)

    # Final fallback to DummyPolicy
    return DummyPolicy().predict(state)


class DummyPolicy:
    """
    Simple deterministic policy used when no RL model is available.
    Predict returns 0 for any input (e.g., 'stay in lane').
    """

    def predict(self, state: Any) -> int:
        # Attempt to use simple heuristics from state if it's array-like
        try:
            import numpy as np

            arr = np.array(state)
            # If lateral_velocity high -> suggest lane change (1)
            if arr.ndim > 0 and arr.size > 0:
                val = float(arr.ravel()[0])
                return int(val > 0)  # simple placeholder
        except Exception:
            pass
        return 0


def compute_route(
    graph: Any, start: str, goal: str, weights: Optional[Dict[str, float]] = None, algo: str = "astar"
) -> Tuple[Sequence[str], float, float]:
    """
    Compute a path using astar or dijkstra (calls routing.py).
    Returns (path, cost, eta_minutes).

    ETA conversion: eta_minutes = cost * 0.5 (per project spec).
    """
    try:
        if algo.lower().startswith("astar"):
            path, cost = astar_route(graph, start, goal)
        else:
            path, cost = dijkstra_route(graph, start, goal)
        # simple ETA conversion
        eta_minutes = float(cost) * 0.5
        logger.info("Computed route %s cost=%.3f eta=%.3fmin", path, cost, eta_minutes)
        return path, float(cost), eta_minutes
    except Exception as e:
        logger.exception("Route computation failed: %s", e)
        return [], float("inf"), float("inf")


# If run as script, provide a tiny demo
if __name__ == "__main__":
    # quick demo to ensure imports and basic functions work
    models = init_models()
    acc = models.get("acc_model")
    print("Accident model available:", acc is not None)
    rl = models.get("rl_model")
    print("RL model available:", type(rl))
    graph = load_sample_graph()
    path, cost, eta = compute_route(graph, "A", "D")
    print("Demo route:", path, cost, eta)
