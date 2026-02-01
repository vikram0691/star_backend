class StarBackendError(Exception):
    """Base exception for the star backend forecasting application."""
    pass

class ConfigurationError(StarBackendError):
    """Raised when application configuration is missing or invalid."""
    pass

class DataValidationError(StarBackendError):
    """Raised when input data violates schema or physical reality constraints."""
    pass

class FeatureEngineeringError(StarBackendError):
    """Raised when feature generation or data alignment fails."""
    pass

class ModelTrainingError(StarBackendError):
    """Raised when model training or Nixtla/LightGBM convergence fails."""
    pass

class InferenceError(StarBackendError):
    """Raised when model prediction or model bundle loading fails."""
    pass