# Multimodal Emotion Recognition

A web-based **Multimodal Emotion Recognition System** that detects human emotions by analyzing **facial expressions (video)** and **audio signals**.  
The project leverages computer vision and deep learning techniques to provide real-time emotion recognition through an interactive web interface.

---

## 📌 Features

- 🎥 **Facial Emotion Recognition**
  - Detects faces using Haar Cascade
  - Classifies emotions from facial expressions using a trained deep learning model

- 🎧 **Audio Emotion Analysis**
  - Processes audio input for emotion-related features
  - Supports multimodal inference (audio + video)

- 🌐 **Web Application**
  - User-friendly interface built with HTML templates
  - Real-time emotion prediction display

- ⚡ **Real-Time Processing**
  - Uses OpenCV for live video capture and inference

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- Jinja Templates (Flask)

### Backend
- Python
- Flask

### Machine Learning & Computer Vision
- OpenCV
- DeepFace
- Haar Cascade Classifier
- NumPy

---

## 📂 Project Structure
Multimodal-Emotion-Recognition/
│
├── templates/
│ ├── index.html
│ ├── emotion_detection.html
│ └── logs.html
│
├── emotion.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Nehaselvaraj23/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
###2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the application
python emotion.py
5️⃣ Open in browser
http://127.0.0.1:5000/
📊 Output

Displays detected emotion labels in real time

Works with live webcam input

Logs emotion predictions for analysis

🚀 Future Enhancements

Improve emotion accuracy using CNN/LSTM models

Add speech-to-text emotion analysis

Deploy using Docker or cloud platforms

Support multiple faces simultaneously

Add emotion analytics dashboard




