import cv2
import mediapipe as mp
import math
import time
from collections import deque

# ---------------- Utility ----------------
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# ---------------- MediaPipe ----------------
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# ---------------- Mouth landmarks ----------------
TOP = 13
BOTTOM = 14
LEFT = 61
RIGHT = 291

# ---------------- Thresholds ----------------
HAND_MOUTH_DISTANCE = 60   # pixels
WARNING_TIME = 2.0
SUSPICIOUS_TIME = 5.0

TALK_MAR = 0.38
YAWN_MAR = 0.55

TALK_FRAMES = 10
YAWN_FRAMES = 25

SMOOTH_WINDOW = 5
mar_history = deque(maxlen=SMOOTH_WINDOW)

# ---------------- State ----------------
hand_near_start = None
talk_frames = 0
yawn_frames = 0
state = "Normal"

# ---------------- Main loop ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_mesh.process(rgb)
    hand_res = hands.process(rgb)

    hand_near_mouth = False
    mar_smooth = 0

    # ---------- Face ----------
    if face_res.multi_face_landmarks:
        face = face_res.multi_face_landmarks[0]

        def px(i):
            lm = face.landmark[i]
            return int(lm.x * w), int(lm.y * h)

        top = px(TOP)
        bottom = px(BOTTOM)
        left = px(LEFT)
        right = px(RIGHT)

        vertical = distance(top, bottom)
        horizontal = distance(left, right) + 1e-6
        mar = vertical / horizontal

        mar_history.append(mar)
        mar_smooth = sum(mar_history) / len(mar_history)

        mouth_center = top

        # ---------- Hand ----------
        if hand_res.multi_hand_landmarks:
            for hand in hand_res.multi_hand_landmarks:
                for lm in hand.landmark:
                    hp = (int(lm.x * w), int(lm.y * h))
                    if distance(hp, mouth_center) < HAND_MOUTH_DISTANCE:
                        hand_near_mouth = True
                        break

    current_time = time.time()

    # ---------- HAND LOGIC ----------
    if hand_near_mouth:
        talk_frames = 0
        yawn_frames = 0

        if hand_near_start is None:
            hand_near_start = current_time

        duration = current_time - hand_near_start

        if duration > SUSPICIOUS_TIME:
            state = "Mouth Hidden (Suspicious)"
        elif duration > WARNING_TIME:
            state = "Mouth Hidden (Warning)"
        else:
            state = "Normal"

    else:
        hand_near_start = None

        # ---------- YAWNING ----------
        if mar_smooth > YAWN_MAR:
            yawn_frames += 1
        else:
            yawn_frames = 0

        # ---------- TALKING ----------
        if mar_smooth > TALK_MAR:
            talk_frames += 1
        else:
            talk_frames = 0

        if yawn_frames > YAWN_FRAMES:
            state = "Yawning"
        elif talk_frames > TALK_FRAMES:
            state = "Talking"
        else:
            state = "Normal"

    # ---------- Display ----------
    cv2.putText(frame, f"STATE: {state}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("FINAL Mouth + Hand + Yawn", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
