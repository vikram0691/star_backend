import structlog
import warnings

import pandas as pd

from typing import Union, Dict, List, Optional, Iterable
from pathlib import Path

from star_backend.exceptions import DataValidationError

logger = structlog.get_logger()

def _load_and_preprocess_excel(file_path: Path, sheet_name: Union[str, int, List[Union[str, int]], None] = 0) -> pd.DataFrame:
    """
    Load data from an Excel file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the Excel file.
        sheet_name (str|int|None): The name of the sheet to load.

    Returns:
        pd.DataFrame: The loaded data.
    """

    # 1. Handle list input
    if isinstance(sheet_name, list):
        if not sheet_name:
            first_sheet = 0
            logger.warning(f"Empty sheet_name list provided. Defaulting to first sheet: {first_sheet}")
        else:
            first_sheet = sheet_name[0] 
            logger.warning(
                f"Multiple sheet names provided: {sheet_name}"
                f"Only the first sheet '{first_sheet}' will be loaded"
            )
        sheet_name = first_sheet

    if sheet_name is None:
        sheet_name = 0  # Default to the first sheet

    try:
        logger.info(f"Reading '{file_path}' (Sheet: {sheet_name})")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        df.columns = (
            df.columns.str.strip()
            .str.replace(r"\s+", "_", regex=True)  # Spaces to underscores
            .str.replace(r"[^0-9a-zA-Z_]", "", regex=True) # Remove special chars (like $ or .)
            .str.lower()
            .str.strip("_") # Remove leading/trailing underscores
        )

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].where(
                df[col].isna(), 
                df[col].astype(str).str.strip()
            )

        return df
    except Exception as e:
        logger.exception(f"Error reading Excel file for '{file_path}'")
        raise DataValidationError(f"Failed to load Excel file: {file_path}") from e

def load_clinical(path: Path, *, sheet_name: Union[str, int, None] = 0) -> pd.DataFrame:
    """Loads Clinical Data."""
    # 1. Generic Load
    df = _load_and_preprocess_excel(path, sheet_name)
    return df

def load_weather(path: Path, *, sheet_name: Union[str, int, None] = 0) -> pd.DataFrame:
    """Loads Weather Data."""
    # 1. Generic Load
    df = _load_and_preprocess_excel(path, sheet_name)
    return df