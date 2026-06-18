from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI()

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("agrovision_model.h5")

print("✅ Model Loaded")
print("📊 Output Shape:", model.output_shape)

# ==========================
# LOAD DISEASE INFO
# ==========================
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# ==========================
# LOAD CLASSES FROM DATASET
# ==========================
DATASET_PATH = "dataset"

class_names = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("🔥 Classes Loaded:", class_names)
print("📦 Total Classes:", len(class_names))

IMG_SIZE = 224

# ==========================
# SEVERITY LOGIC
# ==========================
def get_severity(confidence):
    if confidence < 0.60:
        return "Mild"
    elif confidence < 0.85:
        return "Moderate"
    else:
        return "Severe"

# ==========================
# PREDICTION API
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Image Processing
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        prediction = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Safety Check
        if predicted_index >= len(class_names):
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Model classes do not match dataset classes."
                }
            )

        predicted_class = class_names[predicted_index]

        print("\n======================")
        print("Predicted Index :", predicted_index)
        print("Predicted Class :", predicted_class)
        print("Confidence      :", round(confidence * 100, 2), "%")
        print("======================\n")

        severity = get_severity(confidence)

        info = disease_info.get(predicted_class, {})

        return JSONResponse({
            "status": "success",
            "prediction": {
                "disease": info.get("display_name", predicted_class),
                "confidence": round(confidence, 4),
                "severity": severity
            },
            "details": {
                "cause": info.get("cause", "N/A"),
                "treatment": info.get("treatment", "N/A"),
                "prevention": info.get("prevention", "N/A")
            }
        })

    except Exception as e:
        print("❌ Error:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Prediction failed",
                "details": str(e)
            }
        )