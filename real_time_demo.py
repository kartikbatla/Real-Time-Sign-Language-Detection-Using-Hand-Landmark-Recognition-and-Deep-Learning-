import cv2
import numpy as np
import mediapipe as mp
import pickle
import pyttsx3
from keras.models import load_model

# Load model and encoders
model = load_model("saved_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables
sentence = ""
current_word = ""
last_pred = ""
cooldown_frames = 0
detect_mode = False

# Mouse click coordinates
click_x, click_y = -1, -1


def click_event(event, x, y, flags, param):
    """Capture mouse clicks for button interaction."""
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y


cv2.namedWindow("Sign Language To Speech Conversion", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Sign Language To Speech Conversion", click_event)


def speak_sentence(text):
    """Speak the given English text."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def draw_button(frame, x1, y1, x2, y2, text, active=False):
    """Draws clean buttons with optional highlight for active state."""
    color = (180, 255, 180) if active else (230, 230, 230)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) if detect_mode else None

    # Base UI canvas
    ui = np.ones((h, w, 3), dtype=np.uint8) * 245
    ui[0:h, 0:w] = frame

    all_hands_data = []

    # Detection logic
    if detect_mode and result and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                ui, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            all_hands_data.append(data)

        if len(all_hands_data) == 1:
            all_hands_data.append([0.0] * len(all_hands_data[0]))

        combined_data = np.array(all_hands_data).flatten().reshape(1, -1)
        combined_data = scaler.transform(combined_data)

        prediction = model.predict(combined_data)
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        if pred_label != last_pred and cooldown_frames == 0:
            if pred_label.lower() == "blank":
                sentence += " "
            else:
                current_word = pred_label
                sentence += pred_label
            last_pred = pred_label
            cooldown_frames = 15

    if cooldown_frames > 0:
        cooldown_frames -= 1

    # Text area background
    cv2.rectangle(ui, (0, h - 150), (w, h), (245, 245, 245), -1)

    # Detected character
    cv2.putText(ui, "Detected Char :", (50, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(ui, current_word, (280, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # English sentence
    cv2.putText(ui, "Sentence :", (50, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(ui, sentence.strip(), (200, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    # ---------------- BUTTON AREA ---------------- #
    btn_w, btn_h = 120, 40
    gap = 10
    start_x = w - (btn_w + gap) * 4 - 40
    y_bottom = h - 45

    buttons = {
        "detect": (start_x, y_bottom - btn_h, start_x + btn_w, y_bottom),
        "space": (start_x + (btn_w + gap), y_bottom - btn_h, start_x + 2 * (btn_w + gap) - gap, y_bottom),
        "clear": (start_x + 2 * (btn_w + gap), y_bottom - btn_h, start_x + 3 * (btn_w + gap) - gap, y_bottom),
        "speak": (start_x + 3 * (btn_w + gap), y_bottom - btn_h, start_x + 4 * (btn_w + gap) - gap, y_bottom)
    }

    draw_button(ui, *buttons["detect"], "DETECT" if not detect_mode else "STOP", active=detect_mode)
    draw_button(ui, *buttons["space"], "SPACE")
    draw_button(ui, *buttons["clear"], "CLEAR")
    draw_button(ui, *buttons["speak"], "SPEAK")

    # Handle button clicks
    if click_x != -1 and click_y != -1:
        if buttons["detect"][0] <= click_x <= buttons["detect"][2] and buttons["detect"][1] <= click_y <= buttons["detect"][3]:
            detect_mode = not detect_mode
        elif buttons["space"][0] <= click_x <= buttons["space"][2] and buttons["space"][1] <= click_y <= buttons["space"][3]:
            sentence += " "
        elif buttons["clear"][0] <= click_x <= buttons["clear"][2] and buttons["clear"][1] <= click_y <= buttons["clear"][3]:
            sentence = ""
        elif buttons["speak"][0] <= click_x <= buttons["speak"][2] and buttons["speak"][1] <= click_y <= buttons["speak"][3]:
            if sentence.strip():
                speak_sentence(sentence.strip())
        click_x, click_y = -1, -1

    # Header title
    cv2.rectangle(ui, (0, 0), (w, 50), (245, 245, 245), -1)
    cv2.putText(ui, "Sign Language To Speech Conversion (English Only)", (40, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Sign Language To Speech Conversion", ui)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
