class MLPredictorError(Exception):
    """Базовый класс для всех ошибок пакета."""
    pass

class ModelLoadError(MLPredictorError):
    """Ошибка загрузки модели или препроцессора."""
    pass

class PredictionError(MLPredictorError):
    """Ошибка во время выполнения предсказания."""
    pass

class ValidationError(MLPredictorError):
    """Ошибка валидации входных данных."""
    pass