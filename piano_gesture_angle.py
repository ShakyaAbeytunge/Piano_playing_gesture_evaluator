import cv2
import mediapipe as mp
import time

import math

def clockwise_angle(a, b, c):
    """
    Returns the clockwise angle ABC in range 0–360°.
    A, B, C are (x, y) points.
    """

    # vectors BA and BC (from B to A and from B to C)
    BA = (a[0] - b[0], a[1] - b[1])
    BC = (c[0] - b[0], c[1] - b[1])

    # atan2 of each vector
    angleA = math.atan2(BA[1], BA[0])
    angleC = math.atan2(BC[1], BC[0])

    # clockwise angle = A - C (in radians)
    angle = math.degrees(angleA - angleC)

    # normalize to 0–360
    angle = angle % 360

    return angle

def compute_clockwise_angle(a, b, c):
    """
    Computes the clockwise angle ABC (angle at point B)
    where A, B, C are (x, y) tuples.
    Returns angle in degrees.
    """

    # vectors BA and BC
    BA = (a[0] - b[0], a[1] - b[1])
    BC = (c[0] - b[0], c[1] - b[1])

    # dot product and magnitude
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    mag_ba = math.sqrt(BA[0]**2 + BA[1]**2)
    mag_bc = math.sqrt(BC[0]**2 + BC[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    # angle using cosine rule
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp

    angle = math.degrees(math.acos(cos_angle))

    # Compute sign using cross product (to know clockwise/counterclockwise)
    cross = BA[0]*BC[1] - BA[1]*BC[0]
    if cross < 0:
        angle = -angle  # clockwise defined as negative (you can invert)

    return abs(angle)

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
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        # Draw keypoints
        for idx in POSE_POINTS:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

        # Draw custom connections (lines)
        for a, b in POSE_CONNECTIONS:
            ax, ay = int(lm[a].x * w), int(lm[a].y * h)
            bx, by = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 128, 0), 3)

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

        if pose_results.pose_landmarks:
            pm = pose_results.pose_landmarks.landmark
            elbow_x, elbow_y = int(pm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), int(pm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)


        # Points as tuples
        ELBOW = (elbow_x, elbow_y)
        WRIST = (wrist_x, wrist_y)
        KNUCKLE = (mfmcp_x, mfmcp_y)
        PIP = (mfpip_x, mfpip_y)

        # Angle 1: elbow → wrist → knuckle
        angle_wrist = clockwise_angle(ELBOW, WRIST, KNUCKLE)

        # Angle 2: wrist → knuckle → finger_tip
        angle_finger = clockwise_angle(WRIST, KNUCKLE, PIP)

        cv2.putText(frame, f"Wrist Angle: {int(angle_wrist)}°", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (162, 10, 105), 2)

        cv2.putText(frame, f"Finger Angle: {int(angle_finger)}°", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (162, 10, 105), 2)

        horizontal_threshold = 5  # adjust this value as needed

        label="Adjust Hand Position"
        color=(255, 0, 0)

        if wrist_x - horizontal_threshold  < mfmcp_x and mfmcp_x - horizontal_threshold  < mfpip_x and mfpip_x - horizontal_threshold  < mft_x:

            if angle_finger:
                if angle_finger > 180:
                    label = "Collapse knucles"
                    color = (0, 0, 255)
                    
                elif angle_finger > 150 :
                    if angle_wrist <180  and angle_wrist> 160:
                        label = "Flat hands  "
                        color = (0, 0, 255)
                    elif angle_wrist< 160:
                        label = "wrist flexion  "
                        color = (0, 0, 255)
                    else :   
                        label = "wrist extension"   # bad shape
                        color = (0, 0, 255)

                elif angle_wrist< 160:
                    label = "wrist flexion  "
                    color = (0, 0, 255)
                elif angle_wrist>180 :   
                    label = "wrist extension"   # bad shape
                    color = (0, 0, 255)   
                else : 
                    label = "Neutral Wrist"
                    color = (0, 255, 0)


        cv2.putText(frame, f"{label}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)



        # calculate distance between wrist and middle_finger_mcp
        # distance_wrist_knuckles = ((wrist_x - mfmcp_x) ** 2 + (wrist_y - mfmcp_y) ** 2) ** 0.5
        # # print(f"Distance Wrist to Middle Finger MCP: {distance_wrist_knuckles}")

        # # testing a threshold for the different between the wrist and middle_finger_mcp heights
        # treshold_height = 700  # adjust this value as needed
        # norm_threshold = treshold_height / distance_wrist_knuckles

        # horizontal_threshold = 5  # adjust this value as needed

        # if wrist_x - horizontal_threshold  < mfmcp_x and mfmcp_x - horizontal_threshold  < mfpip_x and mfpip_x - horizontal_threshold  < mft_x:

        #     if mfmcp_y > mfpip_y and mft_y < wrist_y:
        #         cv2.putText(frame, "Collapsed Knuckles", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     elif wrist_y + norm_threshold < mfmcp_y:
        #         if wrist_y < mfpip_y:
        #             cv2.putText(frame, "Wrist Flexion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #         else:
        #             cv2.putText(frame, "No match 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 52, 0), 2)
        #     elif wrist_y - norm_threshold > mfmcp_y:
        #         if wrist_y > mfpip_y:
        #             cv2.putText(frame, "Wrist Extension", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #         else:
        #             cv2.putText(frame, "Neutral Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     elif mfmcp_y < mfpip_y and mfpip_y < mft_y:   
        #         cv2.putText(frame, "Neutral Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     else:
        #         cv2.putText(frame, "you are not playing!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 63), 2)
        # else:
        #     cv2.putText(frame, "Adjust Hand Position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        

    cv2.imshow("Custom Keypoints + Lines", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
