# 🎭 AI Fitness Tracker - Emotion Detection

## 📌 Overview
The **AI Fitness Tracker** is a deep learning-powered system that detects **facial and vocal emotions** in real-time and recommends personalized workouts based on the user's emotional state. It integrates **computer vision, speech processing, and Flask-based web deployment**.

## 🚀 Features
- **Facial Emotion Detection** using **MediaPipe** and a pre-trained CNN model.
- **Voice Emotion Recognition** using **Librosa** and a CNN trained on the **RAVDESS dataset**.
- **Live Video Streaming** via **Flask** with real-time emotion overlays.
- **Workout Recommendations** based on detected emotions.
- **Automatic Logging** of emotions in a CSV file for historical analysis.

## 🛠 Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MediaPipe
- **Speech Processing**: Librosa, PyAudio
- **Machine Learning**: Scikit-Learn, StandardScaler
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas, CSV
- **Threading**: Multi-threaded voice and face processing

## 📂 Project Structure
/AI-Fitness-Tracker │── app2.py # Main Flask application for real-time tracking │── emotion_model.h5 # Pre-trained deep learning model for face emotions │── voice_emotion_model.h5 # Pre-trained voice emotion recognition model │── voice_scaler.pkl # StandardScaler for normalizing voice features │── Face_train.ipynb # Jupyter Notebook for face emotion model training │── Voice_train.ipynb # Jupyter Notebook for voice emotion model training │── requirements.txt # Dependencies list for the project │── templates/ │ └── index.html # Web UI for Flask app (if applicable) │── static/ │ └── assets/ # CSS, JS, or media files (if applicable)

## 🏗 Installation & Setup
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/AI-Fitness-Tracker.git
cd AI-Fitness-Tracker
2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python app2.py
Then, open http://localhost:5000/ in your browser.
