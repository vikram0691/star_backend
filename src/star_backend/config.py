from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

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