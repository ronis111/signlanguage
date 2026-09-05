"""Capture webcam landmark samples for the missing 0-9 sign classes."""

import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
SAMPLES_PER_CLASS = 1000


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in set("0123456789"):
        raise SystemExit("Usage: python collect_number_data.py <0-9>")

    label = sys.argv[1]
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open the camera")

    captured = 0
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands, DATASET_PATH.open("a", newline="", encoding="utf-8") as dataset_file:
        writer = csv.writer(dataset_file)
        while captured < SAMPLES_PER_CLASS:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a camera frame")

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0].landmark
                writer.writerow([
                    value
                    for landmark in landmarks
                    for value in (landmark.x, landmark.y, landmark.z)
                ] + [label])
                captured += 1

            cv2.putText(
                frame,
                f"Sign {label}: {captured}/{SAMPLES_PER_CLASS} - press Q to stop",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture number sign", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured} samples for number {label}.")


if __name__ == "__main__":
    main()
