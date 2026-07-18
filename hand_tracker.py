import cv2
import mediapipe as mp

class HandTracker:

    def __init__(self):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.drawer = mp.solutions.drawing_utils

    def findHands(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        handData = []

        if result.multi_hand_landmarks:

            for handLandmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):

                lmList = []
                rawLandmarks = []

                h, w, _ = frame.shape

                for lm in handLandmarks.landmark:

                    lmList.append(
                        (
                            int(lm.x * w),
                            int(lm.y * h)
                        )
                    )

                    rawLandmarks.append(lm)

                label = handedness.classification[0].label

                handData.append((label, lmList, rawLandmarks))

                self.drawer.draw_landmarks(
                    frame,
                    handLandmarks,
                    self.mpHands.HAND_CONNECTIONS
                )

        return frame, handData