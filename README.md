# Piano Hand Posture Evaluation

This repository contains the complete implementation of all methods explored in this project for **piano hand posture evaluation**. We investigate a range of approaches, including:

- Rule-based methods  
- Multi-Layer Perceptrons (MLPs)  
- Temporal convolutional models  
- Spatial–temporal CNNs  
- Graph-based methods  

All implementations are preserved for reproducibility and comparison purposes.

While multiple models are included, this repository primarily provides a **ready-to-use pipeline for the best-performing model**, a **Spatial–Temporal Graph Convolutional Network (ST-GCN)**. This model leverages the natural graph structure of the human hand and temporal motion dynamics, enabling robust generalization across unseen players.

The guide below explains how to preprocess raw videos to generate inputs compatible with the **ST-GCN-based posture evaluation pipeline**.

---

## Preprocessing Pipeline (ST-GCN)

To ensure consistent and reliable performance, raw piano-playing videos must be processed through several preprocessing stages. All preprocessing scripts are located in:


The preprocessing pipeline consists of three main steps:

1. Video normalization  
2. Hand keypoint extraction  
3. Keypoint normalization  

---

### 1. Video Normalization

**Script:** `ffmpeg_processor.py`

Videos recorded using different devices may vary in resolution, frame rate, and encoding. This script standardizes all videos to a fixed resolution and frame rate to ensure consistent temporal sampling.

**Default configuration**
- Resolution: **1280 × 720**
- Frame rate: **30 FPS**
- Bitrate: **10 Mbps**

**Command**
```bash
python STGCN/preprocessing/ffmpeg_processor.py \
    --input_folder path/to/raw_videos \
    --output_folder normalized_videos

python STGCN/preprocessing/extract_keypoints.py \
    --input_folder normalized_videos \
    --output_folder keypoints

python STGCN/preprocessing/normalizing_keypoints.py \
    --input_folder keypoints \
    --output_folder_kps normalized_keypoints_npy
