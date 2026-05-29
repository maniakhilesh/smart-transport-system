# traffic_provider.py
import os
import random

def get_traffic_provider():
    """
    Decide which traffic source to use.
    PAID → if API key exists
    FREE → default fallback
    """
    if os.getenv("TRAFFIC_API_KEY"):
        return "PAID"
    return "FREE"


def get_live_traffic_snapshot(location=None):
    """
    Unified traffic interface.
    Returns normalized values used by all modules.
    """
    provider = get_traffic_provider()

    if provider == "PAID":
        # Placeholder for future real API
        return {
            "source": "paid",
            "avg_speed": 25,            # km/h
            "congestion_level": 0.75,   # 0–1
            "incidents": ["ACCIDENT"],
        }

    # FREE fallback (safe demo mode)
    return {
        "source": "free",
        "avg_speed": random.uniform(30, 55),
        "congestion_level": random.uniform(0.1, 0.6),
        "incidents": [],
    }
