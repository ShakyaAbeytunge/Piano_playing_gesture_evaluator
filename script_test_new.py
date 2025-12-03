import cv2
import mediapipe as mp
import time

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose keypoints you want
POSE_POINTS = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW
]

# Custom pose CONNEC TIONS
POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
]

# Instances
hands = mp_hands.Hands(max_num_hands=1)
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    h, w, _ = frame.shape

    # -------- DRAW SELECTED POSE POINTS + LINES --------
    # if pose_results.pose_landmarks:
    #     lm = pose_results.pose_landmarks.landmark

    #     # Draw keypoints
    #     for idx in POSE_POINTS:
    #         x, y = int(lm[idx].x * w), int(lm[idx].y * h)
    #         cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

    #     # Draw custom connections (lines)
    #     for a, b in POSE_CONNECTIONS:
    #         ax, ay = int(lm[a].x * w), int(lm[a].y * h)
    #         bx, by = int(lm[b].x * w), int(lm[b].y * h)
    #         cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 3)

    # -------- DRAW HAND LANDMARKS + LINES (full hand) --------
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # draw full hand skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2)
            )

        # get points wrist, middle_finger_tip, middle_finger_mcp, middle_finger_pip; with a delay of 0.5 seconds
        

        lm = hand_landmarks.landmark
        wrist_x, wrist_y = int(lm[mp_hands.HandLandmark.WRIST].x * w), int(lm[mp_hands.HandLandmark.WRIST].y * h)
        
        mft_x, mft_y = int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w), int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)
        
        mfmcp_x, mfmcp_y = int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w), int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
        
        mfpip_x, mfpip_y = int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * w), int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * h)
        print(f"Wrist: ({wrist_x}, {wrist_y}), Middle Finger Tip: ({mft_x}, {mft_y}), Middle Finger MCP: ({mfmcp_x}, {mfmcp_y}), Middle Finger PIP: ({mfpip_x}, {mfpip_y})")

        # calculate distance between wrist and middle_finger_mcp
        distance_wrist_knuckles = ((wrist_x - mfmcp_x) ** 2 + (wrist_y - mfmcp_y) ** 2) ** 0.5
        # print(f"Distance Wrist to Middle Finger MCP: {distance_wrist_knuckles}")

        # testing a threshold for the different between the wrist and middle_finger_mcp heights
        treshold_height = 700  # adjust this value as needed
        norm_threshold = treshold_height / distance_wrist_knuckles

        horizontal_threshold = 5  # adjust this value as needed

        if wrist_x - horizontal_threshold  < mfmcp_x and mfmcp_x - horizontal_threshold  < mfpip_x and mfpip_x - horizontal_threshold  < mft_x:

            if mfmcp_y > mfpip_y and mft_y < wrist_y:
                cv2.putText(frame, "Collapsed Knuckles", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif wrist_y + norm_threshold < mfmcp_y:
                if wrist_y < mfpip_y:
                    cv2.putText(frame, "Wrist Flexion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No match 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 52, 0), 2)
            elif wrist_y - norm_threshold > mfmcp_y:
                if wrist_y > mfpip_y:
                    cv2.putText(frame, "Wrist Extension", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Neutral Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif mfmcp_y < mfpip_y and mfpip_y < mft_y:   
                cv2.putText(frame, "Neutral Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "you are not playing!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 63), 2)
        else:
            cv2.putText(frame, "Adjust Hand Position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        

    cv2.imshow("Custom Keypoints + Lines", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
