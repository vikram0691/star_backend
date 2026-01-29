import structlog
import warnings
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from rapidfuzz import process, fuzz

from typing import Union, Dict, List, Optional, Iterable
from pathlib import Path

from star_backend.exceptions import DataValidationError

logger = structlog.get_logger()

DATE_FORMAT_DETECTION_THRESHOLD = 0.8
CONVERSION_FRACTION_THRESHOLD = 0.5
DATE_LIKE_SAMPLE = 200


# PUBLIC: data.py needs this for the looks_date_like() function
DATE_LIKE_REGEX = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b)"
    r"|[\/\-\:]"
    r"|(\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b)"
    r"|(\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b)",
    re.IGNORECASE,
)

COMMON_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d",
    "%m/%d/%Y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y",
    "%B %d %Y", "%Y%m%d", "%d%m%Y",
]

# PUBLIC: data.py needs this for difflib.get_close_matches()
MONTHS_CANONICAL = [
    "january", "jan", "february", "feb", "march", "mar",
    "april", "apr", "may", "june", "jun", "july", "jul",
    "august", "aug", "september", "sep", "sept", "october", "oct",
    "november", "nov", "december", "dec",
]

def _build_month_map() -> Dict[str, str]:
    """
    Builds the normalization map.
    """
    mapping: Dict[str, str] = {}

    for m in MONTHS_CANONICAL:
        mapping[m] = m[:3]
    return mapping

MONTH_CORRECTIONS = _build_month_map()

VALID_MONTH_KEYS = list(MONTH_CORRECTIONS.keys())

def looks_date_like(series: pd.Series) -> bool:
    """Heuristic check if a column contains date-like strings."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    sample = non_null.head(DATE_LIKE_SAMPLE)
    for v in sample:
        if DATE_LIKE_REGEX.search(v):
            return True
    return False

def fix_month_spelling_in_string(s: Optional[str], cutoff: float = 0.85) -> str:
    """
    Optimized specifically for 'Month Year' format (e.g. 'Januery 2023').
    Splits the string, fixes the first word, and reassembles it.
    """
    if not isinstance(s, str) or not s:
        return f"{s}"
    
    parts = s.strip().split(maxsplit=1)

    if len(parts) < 2:
        return s
    
    month_raw = parts[0]
    year_raw = parts[1]

    if not month_raw.isalpha():
        return f"{s}"

    lower = month_raw.lower()
    fixed_month = month_raw

    if lower in MONTH_CORRECTIONS:
        fixed_month = MONTH_CORRECTIONS[lower]
    else:
        match = process.extractOne(
            lower, 
            VALID_MONTH_KEYS, 
            scorer=fuzz.ratio, 
            score_cutoff=cutoff * 100
        )
        if match:
            fixed_month = MONTH_CORRECTIONS[match[0]]

    return f"{fixed_month} {year_raw}"

def fix_month_spelling(series: pd.Series, cutoff: float = 0.78) -> pd.Series:
    """Optimized month-fixer on unique values."""
    if series.empty:
        return series
    unique_vals = series.dropna().astype(str).unique()
    mapping = {val: fix_month_spelling_in_string(val, cutoff) for val in unique_vals}
    return series.astype(str).map(mapping).where(series.notna())

def safe_infer_and_coerce_column(
    df: pd.DataFrame, col: str, threshold: float = CONVERSION_FRACTION_THRESHOLD
) -> None:
    """Smart Inference Engine."""
    series = df[col]
    n_nonnull = series.notna().sum()
    if n_nonnull == 0:
        return

    # 1. Try Numeric
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() >= max(1, int(n_nonnull * threshold)):
        df[col] = num
        print(col, df[col].head(5))
        return

    if looks_date_like(series):
        sample_raw = series.dropna().astype(str).head(DATE_LIKE_SAMPLE)
        sample_fixed = sample_raw.map(fix_month_spelling_in_string)
        corrected_series = fix_month_spelling(series)

        dt = pd.to_datetime(
            corrected_series, format="%m-%Y", errors="coerce", dayfirst=False
        )
        
        df[col] = dt
        print(col, df[col].head(5))
        return

    print(col, df[col].head(5))
    return

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

        # 3. Smart Inference
        for col in df.columns:
            if ptypes.is_numeric_dtype(df[col]) or ptypes.is_datetime64_any_dtype(df[col]):
                continue
            safe_infer_and_coerce_column(df, col)

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