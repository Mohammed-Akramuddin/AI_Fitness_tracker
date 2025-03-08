# 🎭 AI Fitness Tracker - Emotion Detection

## 📌 Overview
The **AI Fitness Tracker** is a deep learning-powered system that detects **facial and vocal emotions** in real-time and recommends personalized workouts based on the user's emotional state. It integrates **computer vision, speech processing, and Flask-based web deployment**.

---

## 📥 Download Datasets

To train and run the **AI Fitness Tracker**, you need two main datasets:  
one for **voice emotion recognition** and another for **facial emotion detection**.

### 🔊 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
The **RAVDESS dataset** contains speech and song recordings with emotional expressions.  
📥 **Download Link**:  
➡️ [RAVDESS Dataset - Zenodo](https://zenodo.org/record/1188976)  

### 📸 Facial Emotion Recognition Dataset (FER 2013)
For facial emotion recognition, we use the **FER 2013** dataset, which contains thousands of labeled facial expressions.  
📥 **Download Links**:  
➡️ [FER 2013 Dataset - Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
➡️ [FER 2013 Dataset - Papers with Code](https://paperswithcode.com/dataset/fer2013)  

### 📌 How to Use These Datasets
1. **Download the datasets** from the links above.  
2. **Extract the files** and place them in the appropriate directories (`datasets/` or any preferred folder).  
3. **Modify the dataset path** in `app2.py` or Jupyter notebooks (`Face_train.ipynb` and `Voice_train.ipynb`) to point to the correct location.  
4. **Train the models** if needed before running the application.  

---

## 🚀 Features
✅ **Facial Emotion Detection** using **MediaPipe** and a pre-trained CNN model.  
✅ **Voice Emotion Recognition** using **Librosa** and a CNN trained on the **RAVDESS dataset**.  
✅ **Live Video Streaming** via **Flask** with real-time emotion overlays.  
✅ **Workout Recommendations** based on detected emotions.  
✅ **Automatic Logging** of emotions in a CSV file for historical analysis.  
✅ **Multi-threaded Processing** for efficient face and voice emotion detection.  

---

## 🛠 Technologies Used
- **Deep Learning**: TensorFlow, Keras  
- **Computer Vision**: OpenCV, MediaPipe  
- **Speech Processing**: Librosa, PyAudio  
- **Machine Learning**: Scikit-Learn, StandardScaler  
- **Web Framework**: Flask  
- **Data Processing**: NumPy, Pandas, CSV  
- **Threading**: Multi-threaded voice and face processing  

---



## 📂 Project Structure
/AI-Fitness-Tracker │── app2.py # Main Flask application for real-time tracking │── emotion_model.h5 # Pre-trained deep learning model for face emotions │── voice_emotion_model.h5 # Pre-trained voice emotion recognition model │── voice_scaler.pkl # StandardScaler for normalizing voice features │── Face_train.ipynb # Jupyter Notebook for face emotion model training │── Voice_train.ipynb # Jupyter Notebook for voice emotion model training │── requirements.txt # Dependencies list for the project │── templates/ │ └── index.html # Web UI for Flask app (if applicable) │── static/ │ └── assets/ # CSS, JS, or media files (if applicable)

### 1️⃣ Clone the Repository
To get started, clone this repository using the following command:

```bash
git clone https://github.com/your-username/AI-Fitness-Tracker.git
cd AI-Fitness-Tracker
```
### 2️⃣ Create a Virtual Environment and Activate It
```bash
# For Linux/Mac
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Flask App
```bash
python app2.py
```
### 🏋️‍♂️ How It Works

#### 1️⃣ Facial Emotion Detection:
- **Uses a pre-trained CNN model** to analyze real-time facial expressions.
- **MediaPipe processes face landmarks** for better accuracy.

#### 2️⃣ Voice Emotion Recognition:
- **Uses Librosa** to extract voice features.
- **CNN model classifies voice into emotional categories.**

#### 3️⃣ Workout Recommendation System:
- **Based on detected emotions, the app suggests workouts dynamically.**
- **Example:**
  - If the user feels **stressed**, it may recommend **meditation or yoga**.
  - If the user is **excited**, it may suggest **high-intensity exercises**.

### 🛠 Future Enhancements

- 🔹 **Mobile App Integration** for a seamless experience.
- 🔹 **More Emotion Categories** for deeper analysis.
- 🔹 **Support for Wearable Devices** (e.g., Fitbit, Apple Watch).
- 🔹 **User Authentication** to track progress over time.

