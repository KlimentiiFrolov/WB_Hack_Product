import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


from .config import (
    CATEGORICAL_FEATURES, 
    NUMERIC_FEATURES, 
    DEFAULT_PREPROCESSOR_PATH
)
from .exceptions import ModelLoadError

class DataPreprocessor:
    def __init__(self):
        self.pipeline = None

    def fit(self, df: pd.DataFrame):
        """Обучение препроцессора."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ]
        )
        self.pipeline.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Трансформация данных."""
        if self.pipeline is None:
            raise ModelLoadError("Препроцессор не загружен")
        return self.pipeline.transform(df)

    def save(self, path: str):
        """Сохранение состояния."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print("[ML] Препроцессор успешно загружен")

    def load(self, path: str):
        """Загрузка состояния."""
        if not Path(path).exists():
            raise ModelLoadError(f"Файл препроцессора не найден: {path}")
        self.pipeline = joblib.load(path)
        return self

def preprocess_data(df: pd.DataFrame, is_training: bool = False, 
                    path: str = DEFAULT_PREPROCESSOR_PATH) -> np.ndarray:
    """
    Основная функция предобработки.
    Если is_training=True: обучается и сохраняет препроцессор, потом трансформирует данные.
    Если is_training=False: загружает сохраненный и трансформирует.
    """
    preprocessor = DataPreprocessor()
    
    if is_training:
        preprocessor.fit(df)
        preprocessor.save(path)
    else:
        preprocessor.load(path)
    
    return preprocessor.transform(df)