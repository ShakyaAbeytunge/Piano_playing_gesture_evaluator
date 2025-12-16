import os
import subprocess
import argparse

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_video(input_path, output_path, width, height, fps, bitrate):
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={width}:{height},fps={fps}",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", "20M",
        "-y",  # overwrite output
        output_path
    ]
    
    print(f"\nProcessing: {input_path}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Saved → {output_path}")

def process_all_videos(input_folder, output_folder, width, height, fps, bitrate):
    ensure_folder(output_folder)

    supported_ext = [".mp4", ".mov", ".mkv", ".avi", ".m4v"]

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            in_path = os.path.join(input_folder, filename)
            out_name = os.path.splitext(filename)[0] + "_normalized.mp4"
            out_path = os.path.join(output_folder, out_name)
            convert_video(in_path, out_path, width, height, fps, bitrate)

    print("\n✅ All videos processed!")

# ====================== VIDEO NORMALIZATION DEFAULT CONFIG ======================
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
BITRATE = "10M"  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input video folder")
    parser.add_argument("--output_folder", type=str, default="normalized_videos", help="Path to output video folder")
    parser.add_argument("--width", type=int, default=TARGET_WIDTH, help="Target video width")
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT, help="Target video height")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Target video frames per second")
    parser.add_argument("--bitrate", type=str, default=BITRATE, help="Target video bitrate (e.g., '10M')")
    args = parser.parse_args()

    process_all_videos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate
    )


    

