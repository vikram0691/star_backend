from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import structlog
import json

from star_backend.data import (
    load_clinical,
    load_weather,
    impute_weather,
    aggregate_cases,
    aggregate_weather,
)
from star_backend.schemas.observations import TrainingObservation
from star_backend.exceptions import DataValidationError


logger = structlog.get_logger()

def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the physical reality of each monthly observation using Pydantic.
    This function is the ONLY defensive gatekeeper in the system.
    """
    try:
        records = df.to_dict(orient="records")
        validated = [
            TrainingObservation(**rec).model_dump()
            for rec in records
        ]
        logger.info("observation_validation_success", rows=len(validated))
        return pd.DataFrame(validated)

    except Exception as e:
        logger.error("observation_validation_failed", error=str(e))
        raise DataValidationError(
            "Gold dataset failed physical reality validation"
        ) from e

def generate_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived cyclical features AFTER validation.
    These are mathematical encodings, not physical observations.
    """
    ts = pd.to_datetime(df["time_stamp"])

    df["sin_month"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * ts.dt.month / 12)

    return df

def build_gold_dataset(
    clinical_path: Path,
    weather_path: Path,
) -> pd.DataFrame:
    """
    Builds the validated monthly gold dataset used for all modeling.
    """

    logger.info("loading_raw_data")

    clinical_df = load_clinical(clinical_path)
    weather_df = load_weather(weather_path)

    weather_df = impute_weather(weather_df)

    cases_monthly = aggregate_cases(clinical_df)
    weather_monthly = aggregate_weather(weather_df)

    gold_df = pd.merge(
        cases_monthly,
        weather_monthly,
        on="time_stamp",
        how="left",
    )

    # Domain flags (still reality, not modeling tricks)
    gold_df["is_outbreak_2019"] = (
        gold_df["time_stamp"].dt.year == 2019
    )
    gold_df["is_covid_period"] = gold_df["time_stamp"].between(
        "2020-03-01", "2021-12-31"
    )

    # 🔐 SINGLE VALIDATION POINT
    gold_df = validate_observations(gold_df)

    # Derived features come AFTER validation
    gold_df = generate_cyclical_features(gold_df)

    return gold_df

def run_training_pipeline(
    gold_df: pd.DataFrame,
    config,
):
    """
    Trains the selected forecasting model.
    Assumes `gold_df` has already passed validation.
    """

    logger.info(
        "starting_training_pipeline",
        model_type=config.model_type,
    )

    # Nixtla schema alignment
    nixtla_df = gold_df.rename(
        columns={
            "time_stamp": "ds",
            "case_count": "y",
        }
    )
    nixtla_df["unique_id"] = "bilaspur_HP"

    # # Engine resolution via registry
    # engine_cls = ModelRegistry.get_model(config.model_type)
    # engine = engine_cls(config)

    # engine.fit(nixtla_df)

    # logger.info("model_training_complete")

    # return engine
    return nixtla_df  # Placeholder since model training is not implemented



def process_forecast(
    clinical_path: Path,    
    weather_path: Path,
    output_dir: Path,
    config,
) -> Dict[str, Any]:
    """
    High-level orchestration used by CLI and API.
    """

    gold_df = build_gold_dataset(
        clinical_path=clinical_path,
        weather_path=weather_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "gold_dataset.csv"
    gold_df.to_csv(gold_path, index=False)

    engine = run_training_pipeline(gold_df, config)
    engine.to_csv(output_dir / "nixtla_ready_dataset.csv", index=False)
    
    logger.info(
        "forecast_pipeline_complete",
        gold_rows=len(gold_df),
        output=str(gold_path),
    )

    return {
        "status": "success",
        "gold_dataset": str(gold_path),
        "rows": len(gold_df),
        "model_type": config.model_type,
    }
