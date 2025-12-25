import cv2
import mediapipe as mp
import numpy as np

# ---------------- MEDIAPIPE POSE ----------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# ---------------- THRESHOLDS ----------------
SHOULDER_TILT_THRESH = 0.04     # left/right leaning
SLOUCH_THRESH = 0.18            # forward slouching

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    posture = "NO PERSON"

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # -------- Key landmarks --------
        left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_hip  = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # -------- Normalized coordinates --------
        ls_y = left_shoulder.y
        rs_y = right_shoulder.y

        lh_y = left_hip.y
        rh_y = right_hip.y

        # -------- Shoulder tilt (left/right lean) --------
        shoulder_diff = ls_y - rs_y
        # positive → left shoulder lower → leaning right
        # negative → right shoulder lower → leaning left

        # -------- Slouch detection --------
        shoulder_avg_y = (ls_y + rs_y) / 2
        hip_avg_y = (lh_y + rh_y) / 2
        torso_length = hip_avg_y - shoulder_avg_y

        # -------- Decision logic --------
        if shoulder_diff > SHOULDER_TILT_THRESH:
            posture = "LEANING RIGHT"
        elif shoulder_diff < -SHOULDER_TILT_THRESH:
            posture = "LEANING LEFT"
        elif torso_length < SLOUCH_THRESH:
            posture = "SLOUCHING FORWARD"
        else:
            posture = "NORMAL POSTURE"

        # -------- Draw landmarks --------
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # -------- Debug values --------
        cv2.putText(frame, f"Tilt: {shoulder_diff:.3f}",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, f"Torso: {torso_length:.3f}",
                    (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # -------- Display result --------
    cv2.putText(frame, f"POSTURE: {posture}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow("LEVEL 8 - Shoulder & Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
