import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque

# -------------------------
# Load MoveNet Quantized (UINT8) Model - Thunder version
# -------------------------
interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_thunder.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# Preprocess frame for UINT8 model (256x256 for Thunder)
# -------------------------
def preprocess(frame):
    img = cv2.resize(frame, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(img.astype(np.uint8), axis=0)  # uint8 for quantized model
    return input_image

# -------------------------
# Get keypoints
# -------------------------
def get_keypoints(frame):
    input_image = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints[0][0]

# -------------------------
# Improved Action detection functions
# -------------------------
def detect_turn(keypoints):
    nose_x = keypoints[0][1]      # x coordinate
    nose_conf = keypoints[0][2]
    left_shoulder_x = keypoints[5][1]
    ls_conf = keypoints[5][2]
    right_shoulder_x = keypoints[6][1]
    rs_conf = keypoints[6][2]

    if nose_conf < 0.3 or ls_conf < 0.3 or rs_conf < 0.3:
        return "No Detection"

    center_x = (left_shoulder_x + right_shoulder_x) / 2
    diff = nose_x - center_x

    if diff < -0.06:
        return "Turning Right"
    elif diff > 0.06:
        return "Turning Left"
    else:
        return "Facing Forward"

def detect_look(keypoints):
    nose_y = keypoints[0][0]      # smaller y = higher in image
    eye_y_avg = (keypoints[1][0] + keypoints[2][0]) / 2  # average of left and right eye y
    nose_conf = keypoints[0][2]
    eyes_conf = (keypoints[1][2] + keypoints[2][2]) / 2

    if nose_conf < 0.3 or eyes_conf < 0.3:
        return "No Detection"

    if nose_y < eye_y_avg - 0.04:
        return "Looking Up"
    elif nose_y > eye_y_avg + 0.04:
        return "Looking Down"
    else:
        return "Looking Forward"

def detect_posture(keypoints):
    shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
    hip_y = (keypoints[11][0] + keypoints[12][0]) / 2
    shoulder_conf = (keypoints[5][2] + keypoints[6][2]) / 2
    hip_conf = (keypoints[11][2] + keypoints[12][2]) / 2

    if shoulder_conf < 0.3 or hip_conf < 0.3:
        return "No Detection"

    if shoulder_y > hip_y - 0.05:
        return "Slouching"
    else:
        return "Straight Posture"

# -------------------------
# Phone detection with smoothing
# -------------------------
phone_queue = deque(maxlen=5)  # Increased to 5 for more stable detection

def detect_phone(keypoints):
    nose_x, nose_y, n_conf = keypoints[0]
    left_wrist_x, left_wrist_y, lw_conf = keypoints[9]
    right_wrist_x, right_wrist_y, rw_conf = keypoints[10]

    detected = "No Phone"
    threshold = 0.22  # Relaxed to catch phone near face

    if n_conf > 0.3:
        if (lw_conf > 0.3 and 
            abs(left_wrist_x - nose_x) < threshold and 
            abs(left_wrist_y - nose_y) < threshold):
            detected = "Phone Left Hand"
        elif (rw_conf > 0.3 and 
              abs(right_wrist_x - nose_x) < threshold and 
              abs(right_wrist_y - nose_y) < threshold):
            detected = "Phone Right Hand"

    phone_queue.append(detected)
    if phone_queue.count(detected) >= 3 and detected != "No Phone":
        return detected
    return "No Phone"

# -------------------------
# Draw skeleton
# -------------------------
EDGES = {
    (0,1):(255,0,0), (0,2):(255,0,0), (1,3):(255,0,0), (2,4):(255,0,0),
    (0,5):(0,255,0), (0,6):(0,255,0), (5,7):(0,255,0), (7,9):(0,255,0),
    (6,8):(0,255,0), (8,10):(0,255,0), (5,6):(0,255,0), (5,11):(0,255,0),
    (6,12):(0,255,0), (11,12):(0,255,0), (11,13):(0,255,0), (13,15):(0,255,0),
    (12,14):(0,255,0), (14,16):(0,255,0)
}

def draw_skeleton(frame, keypoints):
    h, w, _ = frame.shape
    for edge, color in EDGES.items():
        p1 = keypoints[edge[0]]
        p2 = keypoints[edge[1]]
        if p1[2] > 0.3 and p2[2] > 0.3:
            x1, y1 = int(p1[1]*w), int(p1[0]*h)
            x2, y2 = int(p2[1]*w), int(p2[0]*h)
            cv2.line(frame, (x1,y1), (x2,y2), color, 2)
    for kp in keypoints:
        if kp[2] > 0.3:
            x, y = int(kp[1]*w), int(kp[0]*h)
            cv2.circle(frame, (x,y), 4, (0,0,255), -1)

# -------------------------
# Main Loop
# -------------------------
cap = cv2.VideoCapture(0)
alert_cooldown = 2  # seconds (kept but no sound)
last_alert_time = time.time()
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = get_keypoints(frame)

    turn = detect_turn(keypoints)
    look = detect_look(keypoints)
    phone = detect_phone(keypoints)
    posture = detect_posture(keypoints)

    draw_skeleton(frame, keypoints)

    # -------------------------
    # Trigger Alerts (visual only - no sound)
    # -------------------------
    alert_text = ""
    if phone != "No Phone":
        alert_text += f"{phone} "
    if posture == "Slouching":
        alert_text += "Slouching "

    current_time = time.time()
    if alert_text and current_time - last_alert_time > alert_cooldown:
        last_alert_time = current_time
        print(f"ALERT: {alert_text.strip()}")  # Console alert only

    # Display info on frame
    cv2.putText(frame, f"Turn: {turn} | Look: {look}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"{phone} | {posture}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if alert_text:
        cv2.putText(frame, f"ALERT: {alert_text.strip()}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Cheating Detection with Alerts (Thunder)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()