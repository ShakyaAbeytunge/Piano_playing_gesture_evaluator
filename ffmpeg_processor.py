import os
import subprocess

# -------- USER SETTINGS --------
INPUT_FOLDER = "piano_set_1_raw"
OUTPUT_FOLDER = "piano_set_1_processed"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
BITRATE = "10M"   # Change to "20M" if you want maximum quality

# --------------------------------

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_video(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT},fps={TARGET_FPS}",
        "-b:v", BITRATE,
        "-maxrate", BITRATE,
        "-bufsize", "20M",
        "-y",  # overwrite output
        output_path
    ]
    
    print(f"\nProcessing: {input_path}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Saved → {output_path}")

def process_all_videos():
    ensure_folder(OUTPUT_FOLDER)

    supported_ext = [".mp4", ".mov", ".mkv", ".avi", ".m4v"]

    for filename in os.listdir(INPUT_FOLDER):
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            in_path = os.path.join(INPUT_FOLDER, filename)
            out_name = os.path.splitext(filename)[0] + "_normalized.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)

            convert_video(in_path, out_path)

    print("\n✅ All videos processed!")

if __name__ == "__main__":
    process_all_videos()
