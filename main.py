from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import threading
import pickle
import tensorflow as tf

web = Flask(__name__)

model = tf.keras.models.load_model('Model/asl_model.h5', compile=False)

with open('Model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Model/labels.txt', 'r') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

print("Labels loaded:", labels)

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_prediction = {"top3": [], "detected": "", "confidence": 0, "hand_present": False}
prediction_lock = threading.Lock()


def extract_landmarks(hand_landmarks):
    """Extract raw 63 values — no flipping, no offset subtraction"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def gen_frames():
    global latest_prediction

    while True:
        success, img = camera.read()
        if not success or img is None:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]

            #Draw skeleton
            mp_draw.draw_landmarks(
                img, hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            #Extract landmarks
            landmarks = extract_landmarks(hand_lms)

            #Scale ONCE, matches training
            landmarks_scaled = scaler.transform([landmarks])

            #Predict
            prediction = model.predict(landmarks_scaled, verbose=0)[0]

            top3_indices = prediction.argsort()[-3:][::-1]
            top3 = [
                {"label": labels[i].upper(), "confidence": round(float(prediction[i]) * 100, 1)}
                for i in top3_indices
            ]

            detected_label = labels[np.argmax(prediction)].upper()
            top_confidence = float(np.max(prediction)) * 100

            cv2.putText(img, f"{detected_label}  {top_confidence:.0f}%",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 150), 3, cv2.LINE_AA)

            with prediction_lock:
                latest_prediction = {
                    "top3": top3,
                    "detected": detected_label,
                    "confidence": top_confidence,
                    "hand_present": True
                }
        else:
            with prediction_lock:
                latest_prediction = {
                    "top3": [], "detected": "",
                    "confidence": 0, "hand_present": False
                }

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@web.route('/')
def index():
    return render_template('index.html')

@web.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@web.route('/prediction')
def prediction():
    with prediction_lock:
        return jsonify(latest_prediction)

if __name__ == '__main__':
    web.run(debug=True, use_reloader=False)