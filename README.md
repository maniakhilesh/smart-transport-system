# Smart Transport Dashboard

AI-assisted Streamlit dashboard for transport monitoring, route planning, accident risk analysis, and reinforcement-learning-based lane decisions.

## Features

- Live dashboard metrics with auto-refresh
- Traffic snapshot abstraction with free and paid provider modes
- RL lane-changing demo using the Q-table in `rl/`
- Accident risk prediction from video-derived features
- Emergency routing with demo graph and real OpenStreetMap routing
- Cached model loading and reusable routing helpers

## Project Structure

- `app.py` - Streamlit entry point and dashboard UI
- `inference.py` - Model loading and inference helpers
- `routing.py` - Demo graph routing utilities
- `traffic_provider.py` - Traffic source selection and normalized live snapshot
- `video_features.py` - Video feature extraction helpers
- `video_utils.py` - Video frame sampling utilities
- `rl/` - RL agent, environment, and Q-table assets
- `models/` - Trained model assets used by the app

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Copy `.env.example` to `.env` if you want to use a traffic API key.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Environment Variables

The current app only reads one optional variable:

- `TRAFFIC_API_KEY` - Enables the paid traffic-provider path in `traffic_provider.py`

The app still runs without any key by falling back to the free/demo mode.

## Run the App

```bash
streamlit run app.py
```

If Streamlit is already installed in your active environment, this starts the dashboard in your browser.

## Notes

- The repository includes trained model assets in `models/` and RL assets in `rl/` so the app can start without retraining.
- `README.txt` is a legacy setup note file. Use this `README.md` as the main project documentation.
- Generated local notes, caches, and secret files are ignored through `.gitignore`.

## Troubleshooting

- If a module cannot be imported, verify the virtual environment is active.
- If traffic APIs are unavailable, the dashboard automatically uses free/demo traffic snapshots.
- If you update dependencies, reinstall with `pip install -r requirements.txt`.