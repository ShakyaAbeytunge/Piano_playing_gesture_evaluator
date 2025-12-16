# Piano Hand Posture Evaluation

This repository contains the complete implementation of all methods explored in this project for **piano hand posture evaluation**. We investigate a range of approaches, including:

- Rule-based methods  
- Multi-Layer Perceptrons (MLPs)  
- Temporal convolutional models  
- Spatial–temporal CNNs  
- Graph-based methods  

All implementations are preserved for reproducibility and comparison purposes.

While multiple models are included, this repository primarily provides a **ready-to-use pipeline for the best-performing model**, a **Spatial–Temporal Graph Convolutional Network (ST-GCN)**. This model leverages the natural graph structure of the human hand and temporal motion dynamics, enabling robust generalization across unseen players.

---

## Preprocessing Pipeline

The guide below explains how to preprocess raw videos to generate inputs compatible with our models.

First, download the dataset from [this drive folder](https://drive.google.com/drive/folders/1feeNVbfy7ozpBxeAeLsTV9Qh-aeMloA_?usp=sharing), and place the `videos` folder inside the `preprocessing` folder of this repository. You can also find the `metadata.csv` (which is needed for training dataset preparation) and `player_metadata.csv` in the same location

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
python preprocessing/ffmpeg_processor.py \
    --input_folder path/to/raw_videos \
    --output_folder normalized_videos
```

**Output**

Normalized .mp4 videos saved in the specified output directory.

---

### 2. Hand Keypoint Extraction

**Script:** `extract_keypoints.py`

This step extracts hand landmarks per frame using MediaPipe Hands. Each frame contains 21 keypoints with (x, y) image coordinates and a relative depth value, forming a 2.5D hand representation.

**Command**
```bash
python preprocessing/extract_keypoints.py \
    --input_folder normalized_videos \
    --output_folder keypoints
```

**Output**

One .npy file per video containing raw hand keypoints across all frames.

---

### 3. Keypoint Normalization

**Script:** `normalizing_keypoints.py`

Raw keypoints are normalized using a hand-based bounding box strategy. This step reduces sensitivity to:

- Camera distance
- Hand size
- Viewpoint variation

The output is directly compatible with ST-GCN training and inference.

**Command**
```bash
python preprocessing/normalizing_keypoints.py \
    --input_folder keypoints \
    --output_folder_kps normalized_keypoints_npy
```

**Output**

Normalized .npy keypoint files ready for ST-GCN input.

---

## Training ST-GCN

First, move your `normalized_keypoints_npy` folder to `models/STGCN/`. And change your current directory in bash to `models/STGCN/`.

```bash
cd models/STGCN/
```

Then run the `create_dataset.py`, which segments keypoint sequences into fixed-length temporal windows and assigns class labels for each sample. The dataset will be struture by player (creating folder for each player).

**Command**

```bash
python create_dataset.py \
  --input_folder path/to/keypoints \
  --output_folder path/to/output_dataset \
  --meta_file path/to/metadata.csv
```

Then you can train the ST-GCN for the dataset by running:

```bash
python train.py \
  --data_folder path/to/output_dataset
```

The default `--num_epochs` = 100, `--learning_rate` = 5-e4, and `--batch_size` = 8 (best for training the model). The best model will be saved into the `best_models` folder.

---

## Running the application

Now, you can run the Piano Hands Application with the trained model checkpoint by running the following command (your current location must be `models/STGCN/`). The default web camera of your device will be used by the application.

```bash
python piano_hands_app.py \
  --model_path path/to/model_checkpoint
```

Try with the provided checkpoint in the `best_models` folder.

Press `q` to quit the application at any time.
