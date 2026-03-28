# Использование пакета `ml_predictor`

Пакет интегрирован в проект и не требует установки через `pip`. Все модели и артефакты хранятся внутри папки `ml_predictor/models/`.

## 1. Инициализация

Создайте экземпляр предиктора. При инициализации автоматически загружаются модель и препроцессор из конфигурации по умолчанию.

```python
from ml_predictor import Predictor

predictor = Predictor()
```

## 2. Формат входных данных

Метод `predict` принимает `pandas.DataFrame`. 
**Обязательные колонки:**
- `id` (уникальный идентификатор, без пропусков)
- `office_from_id`, `route_id` (категориальные)
- `status_1` ... `status_8` (числовые)

```python
import pandas as pd

data = {
    'id': [101, 102],
    'office_from_id': [0, 1],
    'route_id': [0, 1],
    'status_1': [10, 20],
    # ... status_2 до status_8
}
df_input = pd.DataFrame(data)
```

## 3. Получение прогноза

```python
try:
    df_result = predictor.predict(df_input)
except Exception as e:
    # Обработка ошибок (ValidationError, PredictionError, ModelLoadError)
    logger.error(f"Ошибка предсказания: {e}")
```

**Формат выхода (`pd.DataFrame`):**
- Колонки: `id`, `y_pred`.
- **Важно:** Каждая строка входа порождает **10 строк** выхода (по количеству шагов прогноза).
- Пример: Если на входе 2 строки, на выходе будет 20 строк.

## 4. Конфигурация

Параметры можно переопределять через переменные окружения (например, в `.env` или Docker):

- `ML_MODEL_PATH` — путь к файлу модели.
- `ML_PREPROCESSOR_PATH` — путь к файлу препроцессора.
- `ML_FORECAST_POINTS` — количество шагов прогноза (по умолчанию 10).

## 5. Диагностика

Получить информацию о текущей версии модели и метриках:

```python
info = predictor.get_model_info()
print(info['validation_metrics'])
```