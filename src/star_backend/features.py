# src/star_backend/features.py

from mlforecast.lag_transforms import RollingMean

# Registry of allowed recursive transforms
# We keep this extremely small due to only ~60 points
TRANSFORM_REGISTRY = {
    "rolling_mean_3": lambda: RollingMean(window_size=3),
}

def build_lag_transforms(config_dict: dict) -> dict:
    """
    Builds Nixtla-compatible lag transforms.

    Why this is declarative:
    - Prevents data leakage
    - Ensures recursive recomputation during forecasting
    - Keeps feature definitions centralized and auditable

    Example config:
    {
        1: ["rolling_mean_3"]
    }
    """
    transforms = {}

    for lag, transform_names in config_dict.items():
        transforms[lag] = [
            TRANSFORM_REGISTRY[name]()
            for name in transform_names
            if name in TRANSFORM_REGISTRY
        ]

    return transforms
