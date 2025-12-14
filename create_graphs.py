import numpy as np
import pandas as pd
import os
import json

# ---------------- CONFIG ----------------
KEYPOINT_FOLDER = "keypoints_new"
OUTPUT_FOLDER = "dataset_graphs"
META_FILE = "metadata.csv"

FPS = 30
WINDOW = FPS * 4       # 4 seconds
STRIDE = int(WINDOW * 0.2)    # 80% overlap

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# create split folders
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)

# create player folders
for player in ["p001", "p002", "p003", "p004", "p005", "p006"]:
    os.makedirs(os.path.join(OUTPUT_FOLDER, player), exist_ok=True)

# ---------------- LABEL MAP ----------------
LABEL_MAP = {
    "neutral_hands": 0,
    "wrist_flexion": 1,
    "wrist_extension": 2,
    "collapsed_knuckles": 3,
    "flat_hands": 4
}

TRAIN_PLAYERS = ["p001", "p002", "p003", "p004", "p006"]
# VAL_PLAYERS   = ["p004"]
TEST_PLAYERS  = ["p005"]

# with open("label_map.json", "w") as f:
#     json.dump(LABEL_MAP, f, indent=2)

# ---------------- LOAD METADATA ----------------
meta = pd.read_csv(META_FILE)

sample_id = 0

# for _, row in meta.iterrows():
#     kp_path = os.path.join(KEYPOINT_FOLDER, row["video_id"])
#     if not os.path.exists(kp_path):
#         print("⚠️  Missing keypoints:", kp_path)
#         continue

#     keypoints = np.load(kp_path)   # (F, V, 3)

#     label_text = row["posture_label"]
#     label = LABEL_MAP[label_text]

#     player = row["player_id"]
#     if player in TRAIN_PLAYERS:
#         split = "train"
#     # elif player in VAL_PLAYERS:
#     #     split = "val"
#     elif player in TEST_PLAYERS:
#         split = "test"

#     F, V, C = keypoints.shape

#     for start in range(0, F - WINDOW + 1, STRIDE):
#         clip = keypoints[start:start+WINDOW]      # (T,V,3)
#         X = clip.transpose(2, 0, 1)                # (C,T,V)

#         out_path = os.path.join(
#             OUTPUT_FOLDER,
#             f"{split}/sample_{sample_id:05d}.npz"
#         )

#         np.savez(
#             out_path,
#             X=X.astype(np.float32),
#             y=label,
#             player=player,
#             posture=label_text
#         )

#         sample_id += 1

for _, row in meta.iterrows():
    kp_path = os.path.join(KEYPOINT_FOLDER, row["video_id"])
    if not os.path.exists(kp_path):
        print("⚠️  Missing keypoints:", kp_path)
        continue

    keypoints = np.load(kp_path)   # (F, V, 3)

    label_text = row["posture_label"]
    label = LABEL_MAP[label_text]

    player = row["player_id"]

    F, V, C = keypoints.shape

    for start in range(0, F - WINDOW + 1, STRIDE):
        clip = keypoints[start:start+WINDOW]      # (T,V,3)
        X = clip.transpose(2, 0, 1)                # (C,T,V)

        out_path = os.path.join(
            OUTPUT_FOLDER,
            f"{player}/sample_{sample_id:05d}.npz"
        )

        np.savez(
            out_path,
            X=X.astype(np.float32),
            y=label,
            player=player,
            posture=label_text
        )

        sample_id += 1

print("✅ Saved graph samples:", sample_id)
