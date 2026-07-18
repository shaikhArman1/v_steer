import cv2
import math
from hand_tracker import HandTracker

# -----------------------------
# Initialize webcam and tracker
# -----------------------------
cap = cv2.VideoCapture(0)
tracker = HandTracker()

# Steering smoothing
previous_steering = 0.0
SMOOTHING = 0.8
MAX_ANGLE = 35.0

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

while True:

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
        # Thumb Distance Experiment
        # -----------------------------

        thumb_tip = lmList[4]
        thumb_mcp = lmList[2]

        index_mcp = lmList[5]
        middle_mcp = lmList[9]
        pinky_mcp = lmList[17]
        wrist = lmList[0]

        # Raw distances
        thumb_to_palm = distance(thumb_tip, middle_mcp)
        thumb_to_index = distance(thumb_tip, index_mcp)
        thumb_to_pinky = distance(thumb_tip, pinky_mcp)
        thumb_to_wrist = distance(thumb_tip, wrist)
        thumb_length = distance(thumb_tip, thumb_mcp)

        # Normalize by palm width
        palm_width = distance(index_mcp, pinky_mcp)

        thumb_to_palm /= palm_width
        thumb_to_index /= palm_width
        thumb_to_pinky /= palm_width
        thumb_to_wrist /= palm_width
        thumb_length /= palm_width
        
        if label == "Right":

            color = (0,255,0)
            startY = 120

        else:

            color = (0,0,255)
            startY = 250

        cv2.putText(
            frame,
            f"{label} Palm : {thumb_to_palm:.2f}",
            (20,startY),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{label} Index: {thumb_to_index:.2f}",
            (20,startY+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{label} Pinky: {thumb_to_pinky:.2f}",
            (20,startY+50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{label} Wrist: {thumb_to_wrist:.2f}",
            (20,startY+75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{label} Thumb: {thumb_length:.2f}",
            (20,startY+100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

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