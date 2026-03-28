import os
from pathlib import Path

# Базовый путь к папке пакета
PACKAGE_ROOT = Path(__file__).parent.resolve()
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "models"

# Конфигурационные переменные (с приоритетом env vars)
DEFAULT_MODEL_PATH = os.getenv(
    "ML_MODEL_PATH", 
    str(DEFAULT_MODEL_DIR / "ridge_model.pkl")
)
DEFAULT_PREPROCESSOR_PATH = os.getenv(
    "ML_PREPROCESSOR_PATH", 
    str(DEFAULT_MODEL_DIR / "preprocessor.pkl")
)
DEFAULT_METADATA_PATH = os.getenv(
    "ML_METADATA_PATH", 
    str(DEFAULT_MODEL_DIR / "metadata.json")
)

FORECAST_POINTS = int(os.getenv("ML_FORECAST_POINTS", "10"))
TARGET_COL = os.getenv("ML_TARGET_COL", "target")

# Признаки
CATEGORICAL_FEATURES = ['office_from_id', 'route_id']
NUMERIC_FEATURES = [f'status_{i}' for i in range(1, 9)]
REQUIRED_COLUMNS = ['id'] + CATEGORICAL_FEATURES + NUMERIC_FEATURES