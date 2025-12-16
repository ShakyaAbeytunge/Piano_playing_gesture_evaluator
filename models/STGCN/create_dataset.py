import numpy as np
import pandas as pd
import os
import argparse

def create_dataset(input_folder, output_folder, meta_file, fps=30, window_sec=4, stride_sec=1):

    window = window_sec * fps
    stride = stride_sec * fps

    os.makedirs(output_folder, exist_ok=True)

    # create player folders
    for player in ["p001", "p002", "p003", "p004", "p005"]:
        os.makedirs(os.path.join(output_folder, player), exist_ok=True)

    # ---------------- LOAD METADATA ----------------
    meta = pd.read_csv(meta_file)

    sample_id = 0

    for _, row in meta.iterrows():
        kp_path = os.path.join(input_folder, row["video_id"])
        if not os.path.exists(kp_path):
            print("⚠️  Missing keypoints:", kp_path)
            continue

        keypoints = np.load(kp_path)   # (F, V, 3)

        label_text = row["posture_label"]
        label = LABEL_MAP[label_text]

        player = row["player_id"]

        F, V, C = keypoints.shape

        for start in range(0, F - window + 1, stride):
            clip = keypoints[start:start+window]      # (T,V,3)
            X = clip.transpose(2, 0, 1)                # (C,T,V)

            out_path = os.path.join(
                output_folder,
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

    print(f"✅ Saved {sample_id} graph samples in {output_folder}.")

if __name__ == "__main__":
    # ---------------- LABEL MAP ----------------
    LABEL_MAP = {
        "neutral_hands": 0,
        "wrist_flexion": 1,
        "wrist_extension": 2,
        "collapsed_knuckles": 3,
        "flat_hands": 4
    }

    # --- DAULT CONFIG VALUES ---
    DEFAULT_FPS = 30
    DEFAULT_WINDOW_SEC = 4
    DEFAULT_STRIDE_SEC = 1

    # ---------------- PARSE ARGS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing keypoint .npy files")
    parser.add_argument("--output_folder", type=str, default="dataset", help="Path to output folder for graph .npz files")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second of the keypoint data")
    parser.add_argument("--window_sec", type=int, default=DEFAULT_WINDOW_SEC, help="Window size in seconds")
    parser.add_argument("--stride_sec", type=int, default=DEFAULT_STRIDE_SEC, help="Stride size in seconds")
    args = parser.parse_args()

    create_dataset(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        meta_file=args.meta_file,
        fps=args.fps,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec
    )