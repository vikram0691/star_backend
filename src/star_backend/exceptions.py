class StarBackendError(Exception):
    """Base exception for the application."""
    pass

class ConfigurationError(StarBackendError):
    """Raised when critical settings (like API keys) are missing."""
    pass

class DataValidationError(StarBackendError):
    """Raised when input data (Excel) is missing columns or empty."""
    pass

class ModelTrainingError(StarBackendError):
    """Raised when SARIMAX or Nixtla fails to converge."""
    pass