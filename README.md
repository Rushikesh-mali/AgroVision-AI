 # 🌿 AgroVision AI  
### Intelligent Plant Disease Detection System

AgroVision is an AI-powered plant disease detection platform that helps farmers, students, and researchers instantly identify plant diseases using image analysis.

---

## 🚀 Features

- 📸 Upload or capture plant leaf images
- 🤖 AI-powered disease detection using deep learning (MobileNetV2)
- 📊 Confidence score + severity prediction
- 💊 Detailed treatment, cause & prevention suggestions
- 🌍 Multi-language support (EN / HI / MR)
- ⚡ Fast & responsive UI

---

## 🧠 Supported Plants

### 🍆 Eggplant
- Healthy
- Insect Pest
- Leaf Spot
- Mosaic Virus
- Small Leaf
- White Mold
- Wilt

### 🍇 Grapes
- Black Rot
- ESCA
- Healthy
- Leaf Blight

### 🍊 Orange
- Citrus Canker
- Nutrient Deficiency
- Healthy

### 🥔 Potato
- Early Blight
- Late Blight
- Healthy

### 🍅 Tomato
- Early Blight
- Late Blight
- Leaf Mold
- Healthy

---

## 🏗️ Tech Stack

### Frontend
- HTML5
- Tailwind CSS
- JavaScript

### Backend
- FastAPI (Python)

### AI / ML
- TensorFlow
- MobileNetV2 (Transfer Learning)

---

## ⚙️ Project Structure
AgroVisionAI/
│
├── agrovision-frontend/ # Frontend UI
├── dataset/ # (Not included in repo)
├── agrovision_model.h5 # Trained model (optional)
├── disease_info.json # Disease details
├── main.py # FastAPI backend
├── train_model.py # Model training script
├── .gitignore
└── README.md


---

## 🧪 How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Rushikesh-mali/AgroVision-AI.git
cd AgroVision-AI


### 2️⃣ Setup Python Environment
python -m venv agrovision_env
agrovision_env\Scripts\activate
pip install -r requirements.txt


###3️⃣ Run Backend (FastAPI)
uvicorn main:app --reload

👉 Backend runs at:
http://127.0.0.1:8000

4️⃣ Run Frontend
cd agrovision-frontend
npx live-server
📡 API Endpoint
POST /predict

Request:

Image file (form-data)

Response:

{
  "prediction": {
    "disease": "Eggplant Insect Pest",
    "confidence": 0.87,
    "severity": "Moderate"
  },
  "details": {
    "cause": "...",
    "treatment": "...",
    "prevention": "..."
  }
}
📊 Model Details
Architecture: MobileNetV2 (Transfer Learning)
Input Size: 224x224
Classes: 21+
Accuracy: ~98% (validation)
⚠️ Dataset

Dataset is not included in this repository due to size.

👉 You can use:

Kaggle Plant Disease Dataset
Custom datasets (Eggplant, Grapes, Orange, etc.)
💡 Future Improvements
📱 Mobile App (React Native / Flutter)
☁️ Cloud deployment (AWS / GCP)
🧠 Real-time disease tracking
🌾 Farmer recommendation system
📈 Analytics dashboard

👨‍💻 Author
Rushikesh Mali
Electronics & Telecommunication Engineer
Startup Builder | AI Enthusiast

⭐ Support

If you like this project:

👉 Star the repo ⭐
👉 Share with others 🚀

🔥 Vision

AgroVision aims to empower farmers with AI-driven agriculture intelligence to reduce crop loss and increase productivity.