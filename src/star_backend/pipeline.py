import structlog
import json
from pathlib import Path
from star_backend.data import load_excel
from typing import Dict, Any


logger = structlog.get_logger()


def process_forecast(clinical_path: Path, weather_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process clinical and weather data to generate forecasts.
    """

    logger.info("Loading clinical and weather data...")

    clinical_data = load_excel(clinical_path)
    weather_data = load_excel(weather_path)


    # 2. Build the Result Dictionary
    result = {
        "status": "success",
        "meta": {
            "clinical_file": str(clinical_path),
            "weather_file": str(weather_path)
        },
        "clinical_summary": {
            "rows": len(clinical_data),
            "preview": clinical_data.head(5).to_dict(orient="records")
        },
        "weather_summary": {
            "rows": len(weather_data),
            "preview": weather_data.head(5).to_dict(orient="records")
        }
    }

    # 3. SAVE TO FILE (The New Part)
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_file = output_dir / "forecast_results.json"
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
        
    logger.info(f"✅ Results saved to: {output_file}")
    
    # 4. Return it (optional, but good for confirmation)
    return result