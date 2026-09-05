"""Add augmented landmark samples from the available alphabet reference photos."""

import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "static" / "assets" / "a-z"
DATASET_PATH = BASE_DIR / "dataset.csv"
MISSING_LETTERS = tuple("JKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LETTER = 1000


def extract_largest_hand(image, hands):
    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None

    hand = max(
        result.multi_hand_landmarks,
        key=lambda candidate: (
            max(point.x for point in candidate.landmark)
            - min(point.x for point in candidate.landmark)
        ) * (
            max(point.y for point in candidate.landmark)
            - min(point.y for point in candidate.landmark)
        ),
    )
    return np.array(
        [[point.x, point.y, point.z] for point in hand.landmark],
        dtype=np.float32,
    ).reshape(63)


def augment_landmarks(reference, rng):
    points = reference.reshape(21, 3).copy()
    points[:, :2] += rng.normal(0, 0.008, size=(21, 2))
    points[:, 2] += rng.normal(0, 0.004, size=21)
    points[:, :2] = np.clip(points[:, :2], 0.001, 0.999)
    return points.reshape(63)


def main():
    rng = np.random.default_rng(42)
    rows = []
    imported = []

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2,
    ) as hands:
        for letter in MISSING_LETTERS:
            reference_path = REFERENCE_DIR / f"{letter.lower()}.jpeg"
            image = cv2.imread(str(reference_path))
            if image is None:
                raise FileNotFoundError(f"Reference image not found: {reference_path}")

            reference = extract_largest_hand(image, hands)
            if reference is None:
                raise RuntimeError(f"No hand detected in reference image: {reference_path}")

            rows.extend(
                (*augment_landmarks(reference, rng), letter)
                for _ in range(SAMPLES_PER_LETTER)
            )
            imported.append(letter)

    with DATASET_PATH.open("a", newline="", encoding="utf-8") as dataset_file:
        csv.writer(dataset_file).writerows(rows)

    print(f"Added {len(rows)} samples for: {', '.join(imported)}")


if __name__ == "__main__":
    main()
