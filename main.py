from io import BytesIO, StringIO

from fastapi import FastAPI,File,UploadFile,HTTPException
from pathlib import Path

from starlette.responses import StreamingResponse

from ml_predictor.config import REQUIRED_COLUMNS
from ml_predictor import Predictor
import pandas as pd
app = FastAPI()
predictor = Predictor()

ALLOWED_EXTENSIONS = [".parquet"]
MAX_FILE_SIZE = 100*1024*1024




@app.post("/predict")
async def get_predict(file:UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,detail=f"Недопустимый тип файла: {ext}")
    contents = await file.read()
    if len(contents)>MAX_FILE_SIZE:
        raise HTTPException(status_code=400,detail=f"Файл слишком большой. Максимальный размер:{MAX_FILE_SIZE}")
    try:
        df = pd.read_parquet(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Ошибка чтения файла: {str(e)}")
    # missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    # if missing_columns:
    #     raise HTTPException(status_code=400,detail=f"Неверная структура файла. Отсутствуют обязательные колонки: {missing_columns}")
    try:
        df_result = predictor.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Ошибка предсказания: {str(e)}")
    buffer = StringIO()
    df_result.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue().encode()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predicted_{file.filename}.csv"}
    )
@app.get("/health")
def get_health():
    info = predictor.get_model_info()
    return {"status":"ok",
            "model_flag":info['validation_metrics']
            }