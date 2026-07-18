import cv2
import csv
from hand_tracker import HandTracker

tracker = HandTracker()
cap = cv2.VideoCapture(0)

current_label = "NONE"
sample_count = 0

# ---------- Duplicate frame filtering ----------
last_saved_features = None
MOVEMENT_THRESHOLD = 0.01
status = "WAITING"


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


csv_file = open("left_dataset.csv", "a", newline="")
writer = csv.writer(csv_file)

if csv_file.tell() == 0:

    header = ["label"]

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    writer.writerow(header)

print("LEFT HAND DATASET")
print("-----------------")
print("N -> Neutral")
print("B -> Brake")
print("S -> Stop")
print("Q -> Quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame, hands = tracker.findHands(frame)

    status = "WAITING"

    for handLabel, lmList, rawLm in hands:

        if handLabel != "Left":
            continue

        cv2.putText(
            frame,
            "LEFT HAND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Recording : {current_label}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples : {sample_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Status : {status}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        if current_label != "NONE":

            features = extract_features(rawLm)

            save = False

            if last_saved_features is None:
                save = True

            else:

                diff = 0

                for a, b in zip(features, last_saved_features):
                    diff += abs(a - b)

                diff /= len(features)

                if diff > MOVEMENT_THRESHOLD:
                    save = True

            if save:

                row = [current_label] + features

                writer.writerow(row)

                sample_count += 1

                last_saved_features = features

                status = "SAVED"

            else:

                status = "SKIPPED"

    cv2.imshow("Collect LEFT Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        current_label = "NEUTRAL"

    elif key == ord("b"):
        current_label = "BRAKE"

    elif key == ord("s"):
        current_label = "NONE"

    elif key == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()