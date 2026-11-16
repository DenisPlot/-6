from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Безопасная загрузка модели
model = None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"ОШИБКА загрузки модели: {e}. Запустите train_model.py!")

IRIS_INFO = {
    "setosa": {
        "ru_name": "Ирис щетинистый (Iris setosa)",
        "desc": "Небольшой цветок с короткими и широкими лепестками."
    },
    "versicolor": {
        "ru_name": "Ирис разноцветный (Iris versicolor)",
        "desc": "Средние размеры, лепестки фиолетово-синие с желтым пятном."
    },
    "virginica": {
        "ru_name": "Ирис виргинский (Iris virginica)",
        "desc": "Крупный цветок с длинными лепестками."
    }
}
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict(
        request: Request,
        sepal_length: float = Form(...),
        sepal_width: float = Form(...),
        petal_length: float = Form(...),
        petal_width: float = Form(...)
):
    if model is None:
        return templates.TemplateResponse(request=request, name="index.html",
                                          context={"error": "Модель не загружена. Запустите train_model.py!"})

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(data)[0]
    class_key = CLASS_NAMES[prediction]
    result_info = IRIS_INFO[class_key]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result_name": result_info["ru_name"],
            "result_desc": result_info["desc"],
            "error": None
        }
    )