import cv2
import math
import joblib
import pandas as pd
from hand_tracker import HandTracker
import vgamepad as vg

# -----------------------------
# Initialize webcam and tracker
# -----------------------------
cap = cv2.VideoCapture(0)
tracker = HandTracker()
left_model = joblib.load("left_model.pkl")
right_model = joblib.load("right_model.pkl")
gamepad = vg.VX360Gamepad()

# Steering smoothing
previous_steering = 0.0
SMOOTHING = 0.5
MAX_ANGLE = 35.0

left_stable = "UNKNOWN"
right_stable = "UNKNOWN"

left_candidate = None
right_candidate = None

left_count = 0
right_count = 0

STABLE_FRAMES = 1

def extract_features(rawLm):

    wrist = rawLm[0]

    features = []

    for lm in rawLm:

        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    return features

left_prediction = "UNKNOWN"
right_prediction = "UNKNOWN"

while True:
    
    left_prediction = "UNKNOWN"
    right_prediction = "UNKNOWN"

    success, frame = cap.read()

    if not success:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Detect hands
    frame, hands = tracker.findHands(frame)
    
    left = None
    right = None
    

    # -----------------------------
    # Detect left and right hands
    # -----------------------------
    for label, lmList, rawLm in hands:

        # Palm center (Landmark 9)
        x, y = lmList[9]

        cv2.putText(
            frame,
            label,
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if label == "Left":
            left = (x, y)

        elif label == "Right":
            right = (x, y)
            
                # -----------------------------
        features = extract_features(rawLm)

        if label == "Left":

            features = pd.DataFrame(
                [extract_features(rawLm)],
                columns=left_model.feature_names_in_
            )

            left_prediction = left_model.predict(features)[0]

            cv2.putText(
                frame,
                f"LEFT : {left_stable}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

        elif label == "Right":

            features = pd.DataFrame(
                [extract_features(rawLm)],
                columns=right_model.feature_names_in_
            )

            right_prediction = right_model.predict(features)[0]

            cv2.putText(
                frame,
                f"RIGHT : {right_stable}",
                (20,170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )
            
    # Left hand smoothing
    if left_prediction != "UNKNOWN":

        if left_prediction == left_candidate:
            left_count += 1
        else:
            left_candidate = left_prediction
            left_count = 1

        if left_count >= STABLE_FRAMES:
            left_stable = left_candidate

    # Right hand smoothing
    if right_prediction != "UNKNOWN":

        if right_prediction == right_candidate:
            right_count += 1
        else:
            right_candidate = right_prediction
            right_count = 1

        if right_count >= STABLE_FRAMES:
            right_stable = right_candidate
            
    # -----------------------------
    # Steering calculation
    # -----------------------------
    if left is not None and right is not None:

        # Draw steering line
        cv2.line(frame, left, right, (255, 0, 0), 4)

        cv2.circle(frame, left, 10, (0, 255, 255), -1)
        cv2.circle(frame, right, 10, (0, 255, 255), -1)

        # Calculate steering angle
        dx = right[0] - left[0]
        dy = right[1] - left[1]

        angle = math.degrees(math.atan2(dy, dx))

        # Convert to steering value
        steering = angle / MAX_ANGLE

        # Clamp to [-1, 1]
        steering = max(-1.0, min(1.0, steering))

        # Smooth steering
        steering = (
            previous_steering * SMOOTHING
            + steering * (1 - SMOOTHING)
        )

        previous_steering = steering
        
        # Steering -> Left analog stick
        x_value = int(steering * 32767)

        gamepad.left_joystick(
            x_value=x_value,
            y_value=0
        )

        gamepad.update()
        
        # Throttle
        if right_stable == "THROTTLE":
            gamepad.right_trigger(value=255)
        else:
            gamepad.right_trigger(value=0)

        # Brake
        if left_stable == "BRAKE":
            gamepad.left_trigger(value=255)
        else:
            gamepad.left_trigger(value=0)

        gamepad.update()

        # Display angle
        cv2.putText(
            frame,
            f"Angle: {angle:.1f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display steering
        cv2.putText(
            frame,
            f"Steering: {steering:.2f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # Show webcam
    cv2.imshow("Virtual Steering Wheel", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()