# Описание

## Этот API позволяет:

* Загружать .parquet файлы с данными.
* Проверять структуру и обязательные колонки.
* Получать предсказания модели.
* Скачивать результаты в формате CSV.

Также есть health check для проверки статуса модели.



## Запуск

### Запустите FastAPI сервер:

uvicorn main:app --reload

Сервер будет доступен по адресу: http://127.0.0.1:8000
## Эндпоинты

1. ### POST /predict

    Загрузка файла для предсказаний.

    Формат файла: .parquet
    Максимальный размер файла: 100 MB

   #### Пример запроса с curl:
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
    -F "file=@data.parquet" \
    -o predicted_data.csv 
    ```
    Возвращает: CSV файл с предсказаниями.
2. ### GET /health

    Проверка статуса модели.

    Пример запроса:
    ```bash
    curl http://127.0.0.1:8000/health
   ```

    Пример ответа:
   ```json
    {
     "status": "ok",
     "model_flag": {
       "validation_metrics": { ... }
     }
   }
   ```
## Настройки
   * Разрешённые расширения: .parquet
   * Максимальный размер файла: 100 MB
   * Обязательные колонки: указаны в REQUIRED_COLUMNS (в WB_Hack_Product/ml_predictor/config.py)
## Зависимости
   * fastapi
   * uvicorn
   * pandas
   * starlette
