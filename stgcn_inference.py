import cv2
import mediapipe as mp
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from STGCN_model import STGCN

def landmarks_to_frame(hand_landmarks):
    # Output: (C, V)
    frame = torch.zeros(3, 21)

    for i, lm in enumerate(hand_landmarks.landmark):
        frame[0, i] = lm.x
        frame[1, i] = lm.y
        frame[2, i] = lm.z

    return frame

def build_adjacency(num_nodes, edges):
    A = np.zeros((num_nodes, num_nodes))
    for i,j in edges:
        A[i,j] = 1
        A[j,i] = 1
    np.fill_diagonal(A, 1)
    return A

# ---------------- Graph ----------------
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Build adjacency matrix
A = build_adjacency(num_nodes=21, edges=HAND_EDGES)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# inference model
MODEL_FOLDER = "best_models_stgcn"
model_file = "best_model_0.7246_4sec.pth"
model_path = os.path.join(MODEL_FOLDER, model_file)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STGCN(num_classes=5, A=A).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Instances
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

frame_batch = [] # predict the pose for every 1 second
timestamp = time.time()

fps = cap.get(cv2.CAP_PROP_FPS)
print("Camera-reported FPS:", fps)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     hand_results = hands.process(rgb)

    

