import numpy as np
import pandas as pd
import os
KEYPOINT_FOLDER = "keypoints_new"
META_FILE = "metadata.csv"

FPS = 30
WINDOW = FPS * 4        
STRIDE = int(WINDOW * 0.2)  # 80% overlap

OUTPUT_X = "X_mlp.npy"
OUTPUT_Y = "y_mlp.npy"
OUTPUT_P = "players_mlp.npy"

LABEL_MAP = {
    "neutral_hands": 0,
    "wrist_flexion": 1,
    "wrist_extension": 2,
    "collapsed_knuckles": 3,
    "flat_hands": 4
}

def extract_mlp_features(window):
    mean = np.nanmean(window, axis=0)
    std  = np.nanstd(window, axis=0)
    min_ = np.nanmin(window, axis=0)
    max_ = np.nanmax(window, axis=0)

    features = np.concatenate([
        mean, std, min_, max_
    ], axis=-1)
    return features.flatten()

def build_dataset():
    meta = pd.read_csv(META_FILE)

    X, y, players = [], [], []

    for _, row in meta.iterrows():
        kp_path = os.path.join(KEYPOINT_FOLDER, row["video_id"])
        if not os.path.exists(kp_path):
            print(" Missing:", kp_path)
            continue

        keypoints = np.load(kp_path)  
        label = LABEL_MAP[row["posture_label"]]
        player = row["player_id"]

        F, V, C = keypoints.shape

        for start in range(0, F - WINDOW + 1, STRIDE):
            window = keypoints[start:start + WINDOW]
            features = extract_mlp_features(window)
            X.append(features)
            y.append(label)
            players.append(player)
    
    return np.array(X) , np.array(y), np.array(players)

if __name__ == "__main__":
    X, y, players = build_dataset()

    print("X:", X.shape)
    print("y:", y.shape)
    print("players:", players.shape)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    np.save(OUTPUT_P, players)
