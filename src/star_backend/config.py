from functools import lru_cache
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class TuningConfig(BaseModel):
    """
    Configuration for hyperparameter tuning.
    """

    enabled: bool = False
    n_trials: int = 20
    cv_folds: int = 3

class ForecastConfig(BaseModel):
    """
    Scientific configuration for forecasting.
    Defines HOW forecasting is done.
    """

    # ---- Model selection ----
    model_type: str = Field(
        default="lightgbm",
        description="Forecasting engine to use",
    )

    # ---- Temporal setup ----
    freq: str = "MS"
    horizon: int = 6

    # ---- Feature design ----
    lags: List[int] = Field(default_factory=lambda: [1, 2, 3, 12])

    lag_transforms: Dict[int, List[str]] = Field(
        default_factory=lambda: {
            1: ["rolling_mean_3"],
        }
    )

    # ---- ML defaults ----
    lgbm_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 5,
            "random_state": 42,
            "verbosity": -1,
        }
    )

    # ---- Tuning ----
    tuning: TuningConfig = Field(default_factory=TuningConfig)

class Config:
        env_prefix = "ST_" # Looks for ST_MODEL_TYPE in .env
        
class Settings(BaseSettings):
    app_name: str = "Star Backend"
    environment: str = "development"
    debug: bool = True
    port: int = 8000
    host: str = "127.0.0.1"
    log_level: str = "DEBUG"

    # data file path
    clinical_file: Path = Path("data/raw/ST_CLINICAL.xlsx")
    weather_file: Path = Path("data/raw/ST_WEATHER.xlsx")
    output_dir: Path = Path("data/processed/")
    

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache()
def get_settings() -> Settings:
    return Settings()