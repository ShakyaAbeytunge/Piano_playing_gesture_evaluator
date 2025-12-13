import os
import numpy as np
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------
KEYPOINT_DIR = "keypoints_new"             # folder containing *_processed.npy
LABEL_CSV = "Piano_Hand_Posture _Dataset.csv"

WINDOW_SECONDS = 1
FPS = 30
WINDOW_SIZE = WINDOW_SECONDS * FPS         # 30 frames


# ------------------------------
# LOAD LABELS
# ------------------------------
def load_labels():
    df = pd.read_csv(LABEL_CSV)

    # Create numeric labels
    label_map = {label: i for i, label in enumerate(df['posture_label'].unique())}
    df['label_id'] = df['posture_label'].map(label_map)

    return df, label_map


# ------------------------------
# EXTRACT SEQUENCES
# ------------------------------
def extract_sequences_from_video(npy_path):
    """
    Return full 1-second sequences (30 frames),
    shape per sequence: (30, 21, 3)
    """
    kp = np.load(npy_path)  # shape: (frames, 21, 3)

    sequences = []

    for start in range(0, len(kp), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        if end > len(kp):
            break

        window = kp[start:end]  # (30, 21, 3)
        if window.shape == (WINDOW_SIZE, 21, 3):
            sequences.append(window)

    return sequences


# ------------------------------
# BUILD DATASET
# ------------------------------
def build_dataset():
    df, label_map = load_labels()

    X = []
    y = []

    for _, row in df.iterrows():
        video_file = row['video_id']                  # e.g.: PXL_20251205.mp4
        video_name = os.path.splitext(video_file)[0]  # remove .mp4

        npy_path = os.path.join(KEYPOINT_DIR, f"{video_name}_processed.npy")

        if not os.path.exists(npy_path):
            print("⚠ Missing keypoints for:", video_name)
            continue

        sequences = extract_sequences_from_video(npy_path)

        for seq in sequences:
            X.append(seq)
            y.append(row['label_id'])

    return np.array(X), np.array(y), label_map


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    X, y, label_map = build_dataset()

    print("Dataset built!")
    print("X shape:", X.shape)   # (num_samples, 30, 21, 3)
    print("y shape:", y.shape)
    print("Label map:", label_map)

    np.save("X_tcnn.npy", X)
    np.save("y_tcnn.npy", y)

    print("\nSaved: X_tcnn.npy and y_tcnn.npy")
