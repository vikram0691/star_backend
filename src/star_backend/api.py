# src/star_backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from star_backend.logging_conf import configure_logging
from star_backend.config import get_settings
from star_backend.pipeline import process_forecast


logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    configure_logging()
    settings = get_settings()

    logger.info(f"Starting application: {settings.app_name} in {settings.environment} mode")
    if not settings.clinical_file.exists():
        logger.warning(f"⚠️ Clinical file not found at: {settings.clinical_file}")
    if not settings.weather_file.exists():
        logger.warning(f"⚠️ Weather file not found at: {settings.weather_file}")

    logger.info(f" API Ready. targeting data in: {settings.clinical_file.parent}")
    yield
    logger.info("🛑 API Shutting down")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

@app.post("/forecast")
async def forecast():
    """
    Trigger the forecast using the files defined in settings.
    No file upload required.
    """
    settings = get_settings()
    
    # 1. Validation: Ensure files exist before running
    if not settings.clinical_file.exists():
        raise HTTPException(status_code=404, detail=f"Clinical file missing: {settings.clinical_file}")
    if not settings.weather_file.exists():
        raise HTTPException(status_code=404, detail=f"Weather file missing: {settings.weather_file}")

    try:
        logger.info("Triggering forecast run...")
        result = process_forecast(
            clinical_path=settings.clinical_file, 
            weather_path=settings.weather_file, 
            output_dir=settings.output_dir
        )
        return result

    except Exception as e:
        logger.error("Forecast run failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    """
    Serve the last saved forecast result to the frontend.
    """
    settings = get_settings()
    result_path = settings.output_dir / "forecast_results.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="No results found. Run /forecast first.")
    
    # Return the file directly
    return FileResponse(result_path, media_type="application/json")