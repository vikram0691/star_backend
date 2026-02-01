import typer
import structlog
import json
from pathlib import Path
from typing import Annotated, Optional

from star_backend.logging_conf import configure_logging
from star_backend.config import get_settings, ForecastConfig
from star_backend.pipeline import process_forecast


app = typer.Typer(add_completion=False)
logger = structlog.get_logger()

@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")):
    """
    This function runs ALWAYS, before any specific command.
    """
    configure_logging(verbose = verbose)
    if verbose:
        logger.info("Verbose mode enabled")
        
@app.command()
def run(
    clinical_file: Annotated[
        Optional[Path], 
        typer.Option(
            "--clinical-file", "-cf", 
            help="Path to the Excel file containing clinical case data (individual records)."
        )
    ] = None,
    weather_file: Annotated[
        Optional[Path], 
        typer.Option(
            "--weather-file", "-wf", 
            help="Path to the weather data file containing daily temperature and rainfall measures."
        )
    ] = None,
    output_dir: Annotated[
        Optional[Path], 
        typer.Option(
            "--output-dir", "-o", 
            help="Directory where the validated gold_dataset.csv and forecast results will be saved."
        )
    ] = None,
    ):
    """
    Main command that greets the user.
    """
    settings = get_settings()
    config = ForecastConfig()


    clinical_file = clinical_file or settings.clinical_file
    weather_file = weather_file or settings.weather_file
    output_dir = output_dir or settings.output_dir

    if output_dir is None:
        output_dir = settings.output_dir

    try:
        result = process_forecast(clinical_file, weather_file, output_dir, config)
        typer.secho("Forecast processing completed successfully!", fg=typer.colors.GREEN, bold = True)
    except Exception as e:
        logger.error(f"Error processing forecast:", error=str(e))
        raise typer.Exit(code=1)

    logger.info(f"Data files loaded successfully. Output will be saved to {output_dir}")