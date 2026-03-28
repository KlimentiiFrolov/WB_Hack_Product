import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import metrics

from ml_predictor.preprocess import DataPreprocessor
from ml_predictor.config import DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH, DEFAULT_METADATA_PATH
from ml_predictor.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FORECAST_POINTS

def train(model_class, df):
    print("[ML] Загрузка данных...")
    # Разделение по времени (первые 80% - train, последние 20% - val)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    # Подготовка таргетов
    target_cols = [f'target_step_{i}' for i in range(1, FORECAST_POINTS + 1)]
    y_train = train_df[target_cols].values
    y_val = val_df[target_cols].values
    
    X_train = train_df.drop(columns=['id', 'timestamp'] + target_cols)
    X_val = val_df.drop(columns=['id', 'timestamp'] + target_cols)

    print("[ML] Обучение препроцессора...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train)
    preprocessor.save(DEFAULT_PREPROCESSOR_PATH)
    
    print("[ML] Трансформация данных...")
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    print(f"[ML] Обучение модели ({model_class})...")
    base_model = model_class(alpha=1.0)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_proc, y_train)
    
    # Оценка качества (MSE)
    preds = model.predict(X_val_proc)
    mse = np.mean((preds - y_val) ** 2)
    wape = metrics.wape(y_val, preds)
    
    print(f"[ML] Сохранение модели в {DEFAULT_MODEL_PATH}...")
    Path(DEFAULT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEFAULT_MODEL_PATH)
    
    metadata = {
        "hyperparameters": model.get_params(),
        "validation_metrics": {"wape": float(wape), "steps": FORECAST_POINTS},
        "train_samples": len(train_df),
        "val_samples": len(val_df)
    }
    
    with open(DEFAULT_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Метрики:\n WAPE = {wape:.4f}")
