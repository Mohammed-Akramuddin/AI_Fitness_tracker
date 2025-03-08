import cv2
import mediapipe as mp
import numpy as np
import librosa
import pyaudio
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import joblib
import os
import threading
import queue
from flask import Flask, Response, render_template_string, request
import csv
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Define emotion labels for face and voice
FACE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
VOICE_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
FEATURE_LENGTH = 180

# Load pre-trained face emotion model
try:
    face_model = load_model("emotion_model.h5")
    print("Face model loaded successfully.")
except Exception as e:
    print(f"Error loading face model: {e}")
    exit()

# Function to extract voice features
def extract_voice_features(audio, sample_rate, mfcc=True, chroma=True, mel=True):
    try:
        features = []
        if mfcc:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            features.append(mfccs)
        if chroma:
            chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            chroma_stft = np.mean(chroma_stft.T, axis=0)
            features.append(chroma_stft)
        if mel:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            mel_spec = np.mean(mel_spec.T, axis=0)
            features.append(mel_spec)

        feature_vector = np.concatenate(features, axis=0)
        if len(feature_vector) < FEATURE_LENGTH:
            feature_vector = np.pad(feature_vector, (0, FEATURE_LENGTH - len(feature_vector)))
        else:
            feature_vector = feature_vector[:FEATURE_LENGTH]
        return feature_vector
    except Exception as e:
        print(f"Error extracting voice features: {e}")
        return None

# Build voice CNN model
def build_voice_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and train voice model
def train_voice_model(data_path):
    X, y = [], []
    unique_emotions = list(VOICE_EMOTIONS.values())

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                emotion_code = file.split("-")[2]
                emotion = VOICE_EMOTIONS.get(emotion_code)
                if emotion:
                    try:
                        audio, sr = librosa.load(file_path, sr=22050)
                        features = extract_voice_features(audio, sr)
                        if features is not None:
                            X.append(features)
                            y.append(unique_emotions.index(emotion))
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    if not X:
        print("No valid voice data found!")
        return None, None, None

    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    voice_model = build_voice_model(input_shape=FEATURE_LENGTH, num_classes=len(unique_emotions))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    voice_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping], verbose=1)
    voice_model.save("voice_emotion_model.h5")
    joblib.dump(scaler, "voice_scaler.pkl")
    return voice_model, unique_emotions, scaler

# Global variables
emotion_queue = queue.Queue()
cap = cv2.VideoCapture(0)
running = True
voice_stream = None
p = None
emotion_log = []  # Store emotion data over time

# Workout recommendations based on emotions
def get_workout_recommendation(face_emotion, voice_emotion):
    if "Happy" in [face_emotion, voice_emotion] or "happy" in [face_emotion, voice_emotion]:
        return "Great mood! Try a high-energy cardio session."
    elif "Sad" in [face_emotion, voice_emotion] or "sad" in [face_emotion, voice_emotion]:
        return "Feeling down? How about some gentle yoga or a walk?"
    elif "Angry" in [face_emotion, voice_emotion] or "angry" in [face_emotion, voice_emotion]:
        return "Channel that energy with a boxing workout!"
    elif "Fear" in [face_emotion, voice_emotion] or "fearful" in [face_emotion, voice_emotion]:
        return "Take it easy with some deep breathing exercises."
    else:
        return "Neutral mood? A balanced strength workout sounds perfect."

# Continuous voice emotion detection
def continuous_voice_emotion(voice_model, emotions_list, scaler):
    global voice_stream, p, running
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    WINDOW_SECONDS = 1

    p = pyaudio.PyAudio()
    voice_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Continuous voice recording started...")
    buffer = np.array([], dtype=np.float32)

    while running:
        try:
            data = voice_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.append(buffer, audio_chunk)

            if len(buffer) >= RATE * WINDOW_SECONDS:
                audio_window = buffer[:RATE * WINDOW_SECONDS]
                buffer = buffer[RATE * WINDOW_SECONDS:]

                features = extract_voice_features(audio_window, RATE)
                if features is not None:
                    features = scaler.transform(features.reshape(1, -1))
                    prediction = voice_model.predict(features, verbose=0)
                    emotion_idx = np.argmax(prediction)
                    confidence = prediction[0][emotion_idx]
                    if confidence >= 0.5:
                        emotion = emotions_list[emotion_idx]
                    else:
                        emotion = "Uncertain"
                    emotion_queue.put(("voice", emotion))
                else:
                    emotion_queue.put(("voice", "Unknown"))

        except Exception as e:
            print(f"Voice processing error: {e}")
            emotion_queue.put(("voice", "Error"))
            break

    if voice_stream:
        voice_stream.stop_stream()
        voice_stream.close()
    if p:
        p.terminate()
    print("Voice recording stopped.")

# Video and emotion processing
def generate_frames(voice_model, voice_emotions, scaler):
    global running, emotion_log
    voice_thread = threading.Thread(target=continuous_voice_emotion,
                                    args=(voice_model, voice_emotions, scaler),
                                    daemon=True)
    voice_thread.start()

    face_emotion = "No face detected"
    voice_emotion = "N/A"
    last_log_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Face detection and emotion prediction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min = max(0, int(bboxC.xmin * w))
                y_min = max(0, int(bboxC.ymin * h))
                x_max = min(w, int((bboxC.xmin + bboxC.width) * w))
                y_max = min(h, int((bboxC.ymin + bboxC.height) * h))

                face = frame[y_min:y_max, x_min:x_max]
                if face.shape[0] > 10 and face.shape[1] > 10:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48)) / 255.0
                    face_input = np.reshape(face_resized, (1, 48, 48, 1))
                    face_pred = face_model.predict(face_input, verbose=0)
                    face_emotion = FACE_EMOTIONS[np.argmax(face_pred)]

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face: {face_emotion}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Get latest voice emotion from queue
        try:
            source, emotion = emotion_queue.get_nowait()
            if source == "voice":
                voice_emotion = emotion
        except queue.Empty:
            pass

        # Log emotions every 10 seconds
        current_time = time.time()
        if current_time - last_log_time >= 10:
            emotion_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_emotion": face_emotion,
                "voice_emotion": voice_emotion
            })
            with open("emotion_log.csv", "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "face_emotion", "voice_emotion"])
                if os.stat("emotion_log.csv").st_size == 0:
                    writer.writeheader()
                writer.writerow(emotion_log[-1])
            last_log_time = current_time

        # Overlay emotions and workout recommendation
        recommendation = get_workout_recommendation(face_emotion, voice_emotion)
        cv2.putText(frame, f"Voice: {voice_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Workout: {recommendation}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("Video feed stopped.")

# Flask routes
@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>AI Fitness Tracker - Emotion Detection</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; }
                h1 { color: #333; }
                img { max-width: 100%; height: auto; }
                button { padding: 10px 20px; margin: 10px; background-color: #ff4444; color: white; border: none; cursor: pointer; }
                button:hover { background-color: #cc0000; }
            </style>
        </head>
        <body>
            <h1>Fitness Tracker - Emotion Detection</h1>
            <img src="{{ url_for('video_feed') }}" alt="Video Feed">
            <br>
            <form action="/shutdown" method="post">
                <button type="submit">Stop Tracker</button>
            </form>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(voice_model, voice_emotions, scaler),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global running
    running = False
    time.sleep(1)  # Give time for threads to stop
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print("Server shutdown not supported in this environment.")
    else:
        func()
    return "Shutting down..."

# Main execution
if __name__ == "__main__":
    dataset_path = r"C:\Users\akram\Downloads\Audio_Song_Actors_01-24"  # Update this path

    # Train or load voice model
    try:
        voice_model = load_model("voice_emotion_model.h5")
        scaler = joblib.load("voice_scaler.pkl")
        print("Voice model and scaler loaded successfully.")
        voice_emotions = list(VOICE_EMOTIONS.values())
    except:
        print("Training new voice model...")
        voice_model, voice_emotions, scaler = train_voice_model(dataset_path)
        if voice_model is None:
            print("Failed to train voice model. Exiting.")
            exit()

    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        running = False
        cap.release()
        if voice_stream:
            voice_stream.stop_stream()
            voice_stream.close()
        if p:
            p.terminate()
        print("Application stopped via keyboard interrupt.")