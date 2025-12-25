import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263

# ---------------- UTILS ----------------
def get_point(face, idx, w, h):
    lm = face.landmark[idx]
    return np.array([lm.x * w, lm.y * h])

def iris_center(face, ids, w, h):
    pts = [get_point(face, i, w, h) for i in ids]
    return np.mean(pts, axis=0)

def clamp(x):
    return max(0.0, min(1.0, x))

cap = cv2.VideoCapture(0)

# ---------------- SMOOTHING ----------------
H_hist = deque(maxlen=7)
irisY_hist = deque(maxlen=7)

# ---------------- CALIBRATION ----------------
center_H = None
center_irisY = None
calibrating = False
calib_frames = []
CALIBRATION_FRAMES = 30

# ---------------- THRESHOLDS ----------------
H_THRESH = 0.08       # horizontal sensitivity
V_PIX_THRESH = 6      # pixels for up/down

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    status = "LOOK CENTER & PRESS C"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        direction = "NO FACE"

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]

            # -------- Iris centers --------
            left_iris = iris_center(face, LEFT_IRIS, w, h)
            right_iris = iris_center(face, RIGHT_IRIS, w, h)

            iris = (left_iris + right_iris) / 2

            # -------- Horizontal ratio --------
            L_left = get_point(face, LEFT_EYE_LEFT, w, h)
            L_right = get_point(face, LEFT_EYE_RIGHT, w, h)
            R_left = get_point(face, RIGHT_EYE_LEFT, w, h)
            R_right = get_point(face, RIGHT_EYE_RIGHT, w, h)

            H = (
                clamp((left_iris[0] - L_left[0]) / (L_right[0] - L_left[0])) +
                clamp((right_iris[0] - R_left[0]) / (R_right[0] - R_left[0]))
            ) / 2

            H_hist.append(H)
            irisY_hist.append(iris[1])

            Hs = np.mean(H_hist)
            irisY = np.mean(irisY_hist)

            # -------- CALIBRATION --------
            if calibrating:
                calib_frames.append((Hs, irisY))
                status = f"CALIBRATING {len(calib_frames)}/{CALIBRATION_FRAMES}"

                if len(calib_frames) >= CALIBRATION_FRAMES:
                    center_H = np.mean([p[0] for p in calib_frames])
                    center_irisY = np.mean([p[1] for p in calib_frames])
                    calibrating = False
                    calib_frames.clear()
                    status = "CALIBRATION DONE"

            # -------- DIRECTION LOGIC --------
            if center_H is not None and not calibrating:
                dH = Hs - center_H
                dY = irisY - center_irisY  # IMAGE COORDINATE

                if abs(dH) > abs(dY / 10):  # normalize dominance
                    if dH > H_THRESH:
                        direction = "RIGHT"
                    elif dH < -H_THRESH:
                        direction = "LEFT"
                    else:
                        direction = "CENTER"
                else:
                    if dY > V_PIX_THRESH:
                        direction = "DOWN"
                    elif dY < -V_PIX_THRESH:
                        direction = "UP"
                    else:
                        direction = "CENTER"

            elif calibrating:
                direction = "CALIBRATING"

        # -------- UI --------
        cv2.putText(frame, f"LOOKING: {direction}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)

        cv2.putText(frame, status, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("EYE GAZE (FINAL CORRECT)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('c') and not calibrating and res.multi_face_landmarks:
            calibrating = True
            calib_frames.clear()
            H_hist.clear()
            irisY_hist.clear()
            status = "CALIBRATING..."

cap.release()
cv2.destroyAllWindows()
