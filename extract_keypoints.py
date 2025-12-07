import mediapipe as mp
import cv2
import numpy as np
import os
import glob

mp_hands = mp.solutions.hands

INPUT_FOLDER = "piano_set_1_processed"
OUTPUT_FOLDER = "keypoints"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                pts = []
                for lm in hand.landmark:
                    pts.extend([lm.x, lm.y, lm.z])
            else:
                # If no hand detected → use zeros
                pts = [0]*63

            keypoints.append(pts)
    
    cap.release()
    return np.array(keypoints)

for video in glob.glob(f"{INPUT_FOLDER}/*.mp4"):
    kp = extract_keypoints(video)
    video_name = os.path.splitext(os.path.basename(video))[0]
    np.save(os.path.join(OUTPUT_FOLDER, f"{video_name}.npy"), kp)


