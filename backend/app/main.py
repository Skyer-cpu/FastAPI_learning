from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .ml_handler import model_handler

# Создание экземпляра FastAPI
app = FastAPI()

# Настройка CORS для разрешения запросов от фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Принимает файл изображения и возвращает предсказание."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Читаем байты файла и передаем в обработчик модели
    image_bytes = await file.read()
    prediction = await model_handler.predict(image_bytes)
    
    return prediction



