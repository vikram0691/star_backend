import sys
import logging
import structlog
from star_backend.config import get_settings

def configure_logging(verbose: bool = False):
    settings = get_settings()

    log_level = "DEBUG" if verbose else settings.log_level.upper()

    # Processors applied in both Dev and Prod
    shared_processors = [
        structlog.contextvars.merge_contextvars, # Merge async context (request_id)
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        # Add source info (module, line number) automatically
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    if settings.environment == "production":
        # JSON output for machines
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty colored output for humans
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]

    # Configure standard logging to capture Uvicorn/other libs
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )