import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import os
from time import sleep

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hands_list):
   
    coords = []

    # Process up to 2 hands
    for i in range(2):
        if i < len(hands_list):
            for lm in hands_list[i].landmark:
                coords.extend([lm.x, lm.y, lm.z])
        else:
            # Fill zeros for missing hand
            coords.extend([0.0] * 63)

    return coords

def main(label, samples, delay):
    csv_path = "dataset.csv"
    cols = [f"f{i}" for i in range(126)] + ["label"]  # 63 per hand × 2
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

    cap = cv2.VideoCapture(0)
    print(f"\n=== Starting data collection for label: '{label}' ===")
    print("Press 'q' anytime to stop early.\n")
    print("Get ready...")
    sleep(delay)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        count = 0
        while count < samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Capture only when at least one hand is visible
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                feats = extract_landmarks(results.multi_hand_landmarks)
                row = feats + [label]
                df = pd.DataFrame([row])
                df.to_csv(csv_path, mode='a', header=False, index=False)
                count += 1

                cv2.putText(frame, f"Label: {label} Count: {count}/{samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hands detected. Hold gesture steady.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Collecting Data (Two Hands Supported)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Finished! Collected {count} samples for label '{label}' saved in {csv_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, help="Label name, e.g. 'A', 'hello', or 'blank'")
    parser.add_argument("--samples", type=int, default=300, help="Number of frames to capture")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay before starting (seconds)")
    args = parser.parse_args()
    main(args.label, args.samples, args.delay)
