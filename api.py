import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import io
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# تحميل الموديل
model = YOLO("yolo_last.pt")
model.to("cpu")
model.eval()

# إنشاء التطبيق
app = FastAPI()

# السماح بالتواصل مع أي موقع (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء مجلد static إذا لم يكن موجودًا
os.makedirs("static", exist_ok=True)

# تحميل الملفات الثابتة من مجلد static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), request: Request = None):
    # حفظ الصورة الأصلية
    image_path = f"static/{file.filename}"
    with open(image_path, "wb") as image_file:
        image_bytes = await file.read()
        image_file.write(image_bytes)

    # تحميل الصورة من الملف بدلاً من bytes
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # تمرير الصورة إلى YOLO
    results = model(img_np)
    output = results[0]

    draw = ImageDraw.Draw(img)  # لإنشاء نسخة مرسومة من الصورة
    predicted_classes = []

    # التحقق مما إذا كان هناك كائنات مكتشفة
    if not hasattr(output, "boxes") or len(output.boxes) == 0:
        return JSONResponse(content={
            "prediction": "No objects detected",
            "image_url": f"{request.base_url._url.rstrip('/')}/static/{file.filename}"
        })

    for box in output.boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]  # استخراج اسم الكلاس
        predicted_classes.append(class_name)

        # استخراج إحداثيات البوكس
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{class_name}", fill="red")

    # حفظ الصورة بعد الرسم
    processed_filename = f"processed_{file.filename}"
    processed_image_path = f"static/{processed_filename}"
    img.save(processed_image_path)

    # تحديد الـ base URL تلقائيًا
    base_url = request.base_url._url.rstrip("/") if request else "http://127.0.0.1:8000"

    return JSONResponse(content={
        "prediction": predicted_classes,
        "image_url": f"{base_url}/static/{processed_filename}"
    })
