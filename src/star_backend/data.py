import pandas as pd
from typing import Union
import structlog
from pathlib import Path


logger = structlog.get_logger()

def load_excel(file_path: Path, sheet_name: Union[str, int, None] = 0) -> pd.DataFrame:
    """
    Load data from an Excel file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the Excel file.
        sheet_name (str|int|None): The name of the sheet to load.

    Returns:
        pd.DataFrame: The loaded data.
    """

    if sheet_name is None:
        sheet_name = 0  # Default to the first sheet
        
    logger.info(f"Loading Excel file from {file_path}, sheet: {sheet_name}")

    return pd.read_excel(file_path, sheet_name=sheet_name)