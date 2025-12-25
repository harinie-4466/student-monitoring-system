import cv2
import mediapipe as mp
import numpy as np

# ---------------- INITIALIZE MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# ---------------- CALIBRATION ----------------
baseline_pitch = None
baseline_chin_nose = None

calibration_frames = 30
pitch_samples = []
chin_nose_samples = []

# ---------------- HEAD POSE FUNCTION ----------------
def get_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),     # Nose
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[33].x * w, landmarks[33].y * h),   # Left eye
        (landmarks[263].x * w, landmarks[263].y * h), # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    _, rvec, _ = cv2.solvePnP(
        model_points,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles
    return pitch, yaw, roll

# ---------------- GEOMETRIC BACKUP ----------------
def chin_nose_distance(landmarks, w, h):
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    return np.linalg.norm(chin - nose)

# ---------------- DECISION THRESHOLDS ----------------
YAW_THRESHOLD = 15

PITCH_DOWN_THRESHOLD = 3    # EARLY detection
PITCH_UP_THRESHOLD = -12

CHIN_NOSE_DOWN_DELTA = 15   # backup geometry

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "NO FACE"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        chin_nose_dist = chin_nose_distance(landmarks, w, h)

        # -------- CALIBRATION --------
        if baseline_pitch is None:
            pitch_samples.append(pitch)
            chin_nose_samples.append(chin_nose_dist)

            status = "CALIBRATING... LOOK STRAIGHT"

            if len(pitch_samples) >= calibration_frames:
                baseline_pitch = np.mean(pitch_samples)
                baseline_chin_nose = np.mean(chin_nose_samples)

        else:
            normalized_pitch = pitch - baseline_pitch
            chin_nose_change = baseline_chin_nose - chin_nose_dist

            # -------- FINAL DECISION --------
            if yaw > YAW_THRESHOLD:
                status = "HEAD RIGHT"
            elif yaw < -YAW_THRESHOLD:
                status = "HEAD LEFT"
            elif normalized_pitch > PITCH_DOWN_THRESHOLD or chin_nose_change > CHIN_NOSE_DOWN_DELTA:
                status = "HEAD DOWN"
            elif normalized_pitch < PITCH_UP_THRESHOLD:
                status = "HEAD UP"
            else:
                status = "HEAD CENTER"

            # -------- DEBUG --------
            cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Pitch: {int(normalized_pitch)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Chin-Nose Δ: {int(chin_nose_change)}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, status, (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    cv2.imshow("LEVEL 7 - Head Pose Estimation (Fixed)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
