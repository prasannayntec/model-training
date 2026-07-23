"""Custom exceptions for the LoRA training project."""


class ModelTrainingError(Exception):
    """Base exception for all training-project failures."""


class DatasetValidationError(ModelTrainingError):
    """Raised when dataset files are missing or invalid."""


class ConfigurationError(ModelTrainingError):
    """Raised when configuration files are invalid."""


class EvaluationError(ModelTrainingError):
    """Raised when evaluation cannot be completed successfully."""
