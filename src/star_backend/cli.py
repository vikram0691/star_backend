import typer
import structlog
import json
from pathlib import Path
from typing import Annotated, Optional

from star_backend.logging_conf import configure_logging
from star_backend.config import get_settings
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
    clinical_file: Annotated[Path, typer.Option(..., "--clinical-file", "-cf", help="Path to clinical data file", rich_help_panel="Input Data")],
    weather_file: Annotated[Path, typer.Option(..., "--weather-file", "-wf", help="Path to weather data file", rich_help_panel="Input Data")],
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Output directory", rich_help_panel="Output")] = None,
    ):
    """
    Main command that greets the user.
    """
    settings = get_settings()

    if output_dir is None:
        output_dir = settings.output_dir

    try:
        result = process_forecast(clinical_file, weather_file, output_dir)
        typer.secho("Forecast processing completed successfully!", fg=typer.colors.GREEN, bold = True)
    except Exception as e:
        logger.error(f"Error processing forecast:", error=str(e))
        raise typer.Exit(code=1)

    logger.info(f"Data files loaded successfully. Output will be saved to {output_dir}")