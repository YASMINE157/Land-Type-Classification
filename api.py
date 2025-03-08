import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from torchvision import transforms
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

model = YOLO("yolo_last.pt")
model.to("cpu")
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_path = f"static/{file.filename}"
    with open(image_path, "wb") as image_file:
        image_bytes = await file.read()
        image_file.write(image_bytes)

    img = np.array(Image.open(io.BytesIO(image_bytes)))
    results = model(img)
    output = results[0]

    predictions = []
    labels = []
    scores = []

    for box in output.boxes:
        class_id = int(box.cls)
        score = float(box.conf)
        predictions.append({"class_id": class_id, "score": score})

        labels.append(f"Class {class_id}")
        scores.append(score)

    colors = cm.PuBu(np.linspace(0.5, 1, len(scores)))
    plt.figure(figsize=(5, 3))
    plt.bar(labels, scores, color=colors)
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")

    plot_path = f"static/plot_{file.filename}.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "prediction": predictions,
        "image_url": f"http://127.0.0.1:8000/static/{file.filename}",
        "plot_url": f"http://127.0.0.1:8000/static/plot_{file.filename}.png",
    }
