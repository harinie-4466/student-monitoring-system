import cv2
import mediapipe as mp
import math
import time

# -------------------------------
# Utility: distance
# -------------------------------
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

# -------------------------------
# Thresholds
# -------------------------------
EAR_THRESHOLD = 0.20

BLINK_TIME_MAX = 0.4        # seconds → blink
SLEEP_TIME_MIN = 2.0        # seconds → sleeping

BLINK_RATE_WINDOW = 10      # seconds
SUSPICIOUS_BLINKS = 15

# -------------------------------
# State variables
# -------------------------------
eye_closed_start = None
blink_times = []

state = "NORMAL"

# -------------------------------
# FaceMesh model
# -------------------------------
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        state = "NORMAL"
        current_time = time.time()

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            def pt(i):
                lm = face.landmark[i]
                return int(lm.x * w), int(lm.y * h)

            # -------- EAR calculation --------
            L_up, L_down = pt(160), pt(144)
            L_left, L_right = pt(33), pt(133)

            R_up, R_down = pt(385), pt(380)
            R_left, R_right = pt(362), pt(263)

            left_EAR = distance(L_up, L_down) / distance(L_left, L_right)
            right_EAR = distance(R_up, R_down) / distance(R_left, R_right)

            EAR = (left_EAR + right_EAR) / 2

            # -------- Eye closed logic --------
            if EAR < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time

                closed_duration = current_time - eye_closed_start

                # 🔴 SLEEPING HAS HIGHEST PRIORITY
                if closed_duration >= SLEEP_TIME_MIN:
                    state = "SLEEPING"
                    blink_times.clear()   # STOP blink logic completely

            else:
                # Eyes opened again
                if eye_closed_start is not None:
                    closed_duration = current_time - eye_closed_start

                    # Count blink ONLY if not sleeping
                    if closed_duration <= BLINK_TIME_MAX:
                        blink_times.append(current_time)

                eye_closed_start = None

            # -------- Blink-based suspicious (ONLY IF NOT SLEEPING) --------
            if state != "SLEEPING":
                blink_times = [
                    t for t in blink_times
                    if current_time - t <= BLINK_RATE_WINDOW
                ]

                if len(blink_times) >= SUSPICIOUS_BLINKS:
                    state = "SUSPICIOUS (BLINKING)"

            # -------- Debug --------
            cv2.putText(frame, f"EAR: {EAR:.3f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Blinks(10s): {len(blink_times)}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        else:
            state = "NO FACE"

        # -------- Final display --------
        cv2.putText(frame, f"STATE: {state}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow("Eye State Detection (FINAL FIX)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
