import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any

from .config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_PREPROCESSOR_PATH, 
    DEFAULT_METADATA_PATH,
    FORECAST_POINTS,
    REQUIRED_COLUMNS
)
from .preprocess import DataPreprocessor
from .exceptions import ModelLoadError, PredictionError, ValidationError

class Predictor:
    def __init__(self, model_path: Optional[str] = None, 
                 preprocessor_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Инициализация Predictor.
        Загружает модель и препроцессор.
        """
        # Переопределение путей через config или аргументы
        self.model_path = model_path or (config.get('model_path') if config else None) or DEFAULT_MODEL_PATH
        self.preprocessor_path = preprocessor_path or (config.get('preprocessor_path') if config else None) or DEFAULT_PREPROCESSOR_PATH
        self.metadata_path = config.get('metadata_path', DEFAULT_METADATA_PATH) if config else DEFAULT_METADATA_PATH

        # Выбор модели (пользовательский ввод, если не задано)
        if not Path(self.model_path).exists():
            print(f"[ML] Модель по пути {self.model_path} не найдена.")
            user_path = input("Введите путь к файлу модели (.pkl): ")
            if user_path:
                self.model_path = user_path
            else:
                raise ModelLoadError("Путь к модели не указан и файл не найден.")

        try:
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load(self.preprocessor_path)
        except Exception as e:
            raise ModelLoadError(f"Ошибка загрузки препроцессора: {str(e)}")

        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise ModelLoadError(f"Ошибка загрузки модели: {str(e)}")

        # Загрузка метаданных (опционально)
        self.metadata = {}
        if Path(self.metadata_path).exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

    def _validate_input(self, df: pd.DataFrame):
        """Валидация входных данных."""
        # Проверка колонок
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Отсутствуют необходимые колонки: {missing_cols}")
        
        # Проверка пропусков в id
        if 'id' in df.columns and df['id'].isnull().any():
            raise ValidationError("Обнаружены пропуски (NaN) в колонке 'id'")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Основной метод предсказания.
        Возвращает DataFrame с колонками: id, y_pred.
        Каждая строка входа порождает FORECAST_POINTS строк выхода.
        """
        try:
            self._validate_input(df)
            
            # Предобработка
            X_processed = self.preprocessor.transform(df)
            
            # Предикт
            predictions = self.model.predict(X_processed)

            ids = np.array(df['id'].values)
            expanded_ids = np.repeat(ids, FORECAST_POINTS)
            expanded_preds = predictions.ravel()

            result_df = pd.DataFrame({
                'id': expanded_ids,
                'y_pred': expanded_preds
            })

            return result_df

        except ValidationError:
            raise
        except Exception as e:
            raise PredictionError(f"Ошибка при генерации прогноза: {str(e)}")

    def predict_single(self, row: pd.Series) -> np.ndarray:
        """Предсказание для одной строки (для отладки)."""
        df = row.to_frame().T
        res_df = self.predict(df)
        return np.array(res_df['y_pred'].values)

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        return {}