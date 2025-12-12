import os
import numpy as np
import pandas as pd

KEYPOINT_DIR = "keypoints_new"
LABEL_CSV = "Piano_Hand_Posture _Dataset.csv"

WINDOW_SECONDS = 1
FPS = 30
WINDOW_SIZE = WINDOW_SECONDS * FPS


def load_labels():
    df = pd.read_csv(LABEL_CSV)

    # Use your CSV column name: posture_label
    label_map = {label: i for i, label in enumerate(df['posture_label'].unique())}
    df['label_id'] = df['posture_label'].map(label_map)

    return df, label_map


def extract_features_from_video(npy_path):
    """Load full video keypoints and extract 10-second averaged features."""
    kp = np.load(npy_path)   # shape: (frames, 63)
    samples = []

    for start in range(0, len(kp), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        if end > len(kp):
            break

        window = kp[start:end]
        avg = np.mean(window, axis=0)
        samples.append(avg)

    return samples


def build_dataset():
    df, label_map = load_labels()

    X = []
    y = []

    for _, row in df.iterrows():
        video_file = row['video_id']                  # includes .mp4
        video_name = os.path.splitext(video_file)[0]  # remove .mp4

        npy_path = os.path.join(KEYPOINT_DIR, f"{video_name}_processed.npy")
        if not os.path.exists(npy_path):
            print("⚠ Missing keypoint file:", npy_path)
            continue

        samples = extract_features_from_video(npy_path)

        for s in samples:
            X.append(s)
            y.append(row['label_id'])

    return np.array(X), np.array(y), label_map


if __name__ == "__main__":
    X, y, label_map = build_dataset()
    print("Dataset built!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label map:", label_map)

    np.save("X_mlp.npy", X)
    np.save("y_mlp.npy", y)
