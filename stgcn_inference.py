import cv2
import mediapipe as mp
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from STGCN_model import STGCN
from collections import deque

# ==============================
# GRAPH
# ==============================
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def build_adjacency(num_nodes, edges):
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 1)
    return A

def get_hand_bbox(hand_landmarks, img_w, img_h, margin=10):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    xmin = max(int(min(xs) * img_w) - margin, 0)
    ymin = max(int(min(ys) * img_h) - margin, 0)
    xmax = min(int(max(xs) * img_w) + margin, img_w)
    ymax = min(int(max(ys) * img_h) + margin, img_h)

    return xmin, ymin, xmax, ymax

# ==============================
# KEYPOINT PROCESSING
# ==============================
def normalize_landmarks_HBB(hand_landmarks):
    xs = np.array([lm.x for lm in hand_landmarks.landmark])
    ys = np.array([lm.y for lm in hand_landmarks.landmark])
    zs = np.array([lm.z for lm in hand_landmarks.landmark])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bounding_box = [x_max, x_min, y_max, y_min]

    w = max(x_max - x_min, 1e-6)
    h = max(y_max - y_min, 1e-6)

    xs = (xs - x_min) / w
    ys = (ys - y_min) / h

    return torch.tensor(np.stack([xs, ys, zs]), dtype=torch.float32), bounding_box

def fix_keypoints(curr, history):
    if len(history) == 0:
        return curr

    hist = torch.stack(list(history), dim=0)

    for c in range(2):  # x,y only
        mask = (curr[c] < 0) | (curr[c] > 1)
        if mask.any():
            curr[c, mask] = hist[:, c, mask].mean(dim=0)

    return curr

# ==============================
# CONFIG
# ==============================
CONF_THRESH = 0.6
ENTROPY_THRESH = 1.0
WINDOW_SECONDS = 4
UPDATE_SECONDS = 0.25
NUM_CLASSES = 5

# ---------------- LABEL MAP ----------------
LABEL_MAP = {
    "neutral_hands": 0,
    "wrist_flexion": 1,
    "wrist_extension": 2,
    "collapsed_knuckles": 3,
    "flat_hands": 4
}

POOR_POSTURES = [1, 2, 3, 4]  # all except neutral_hands

# ==============================
# LOAD MODEL
# ==============================
A = build_adjacency(21, HAND_EDGES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = "best_models_stgcn"
model_file = "best_model_0.7246_4sec.pth"
model_path = os.path.join(MODEL_FOLDER, model_file)

model = STGCN(NUM_CLASSES, A).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==============================
# MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if fps == 0 else fps

WINDOW_SIZE = int(WINDOW_SECONDS * fps)
UPDATE_FRAMES = int(UPDATE_SECONDS * fps)

frame_buffer = deque(maxlen=WINDOW_SIZE)
history = deque(maxlen=5)

frame_count = 0
label = "Initializing..."
color = (255, 255, 255)

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        frame_kp, bounding_box = normalize_landmarks_HBB(hand_landmarks)
        frame_kp = fix_keypoints(frame_kp, history)

        history.append(frame_kp)
        frame_buffer.append(frame_kp)

        frame_count += 1

        if len(frame_buffer) == WINDOW_SIZE and frame_count % UPDATE_FRAMES == 0:
            X = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(X)
                probs = F.softmax(logits, dim=1)[0]

            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            pred = probs.argmax().item()

            if max_prob < CONF_THRESH or entropy > ENTROPY_THRESH:
                label = "Posture not recognized"
                color = (0, 0, 255)
            elif pred in POOR_POSTURES:
                label = f"Poor Posture: {list(LABEL_MAP.keys())[pred]}"
                color = (0, 0, 255)
            else:
                label = "Good Posture"
                color = (0, 255, 0)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        xmin, ymin, xmax, ymax = [int(v) for v in bounding_box]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Piano Hand Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()