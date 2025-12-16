import os
import numpy as np
import pandas as pd

KEYPOINT_FOLDER = "keypoints_new"
META_FILE = "metadata.csv"

FPS = 30
WINDOW_SECONDS = 4
WINDOW = FPS * WINDOW_SECONDS
STRIDE = int(WINDOW * 0.2) 

OUTPUT_X = "X_tcnn.npy"
OUTPUT_Y = "y_tcnn.npy"
OUTPUT_P = "players.npy"

LABEL_MAP = {
    "neutral_hands": 0,
    "wrist_flexion": 1,
    "wrist_extension": 2,
    "collapsed_knuckles": 3,
    "flat_hands": 4
}


meta = pd.read_csv(META_FILE)

X, y, players = [], [], []

for _, row in meta.iterrows():
    video_name = os.path.splitext(row["video_id"])[0]
    kp_path = os.path.join(KEYPOINT_FOLDER, f"{video_name}.npy")

    if not os.path.exists(kp_path):
        print("Missing:", kp_path)
        continue

    keypoints = np.load(kp_path)   
    label = LABEL_MAP[row["posture_label"]]
    player = row["player_id"]

    F, J, C = keypoints.shape

    for start in range(0, F - WINDOW + 1, STRIDE):
        window = keypoints[start:start + WINDOW]  # (T,21,C)

        if window.shape == (WINDOW, J, C):
            X.append(window.astype(np.float32))
            y.append(label)
            players.append(player)
        

X = np.array(X)
y = np.array(y)
players = np.array(players)

print("X:", X.shape)
print("y:", y.shape)
print("players:", players.shape)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
np.save(OUTPUT_P, players)
