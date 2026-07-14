import cv2
from hand_tracker import HandTracker

cap = cv2.VideoCapture(0)

tracker = HandTracker()

while True:

    success, frame = cap.read()

    frame = cv2.flip(frame, 1)

    frame, hands = tracker.findHands(frame)

    print(hands)

    cv2.imshow("Virtual Steering", frame)

    if cv2.waitKey(1) == ord("q"):
        break