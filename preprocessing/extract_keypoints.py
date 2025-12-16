import mediapipe as mp
import cv2
import numpy as np
import os
import glob
import argparse

mp_hands = mp.solutions.hands

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,  required=True, help="Path to input video folder")
    parser.add_argument("--output_folder", type=str, default="keypoints", help="Path to output keypoints folder")
    args = parser.parse_args()

    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for video in glob.glob(f"{INPUT_FOLDER}/*.mp4"):
        kp = extract_keypoints(video)
        video_name = os.path.splitext(os.path.basename(video))[0]
        np.save(os.path.join(OUTPUT_FOLDER, f"{video_name}.npy"), kp)
