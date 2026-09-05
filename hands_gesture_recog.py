import cv2
import json
import mediapipe as mp
import numpy as np
from pathlib import Path
from keras.models import load_model


# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
model = load_model(str(BASE_DIR / "cnn_model_final2.h5"))

print("Model loaded successfully")


# ==========================================
# CLASS LABELS
# ==========================================

with (BASE_DIR / "label_classes.json").open(encoding="utf-8") as labels_file:
    classLabels = {
        int(class_id): label
        for class_id, label in json.load(labels_file).items()
    }


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def normalize_landmarks(landmarks):
    """Apply the same wrist-relative, scale-invariant preprocessing as training."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    relative_points = points - points[0]
    scale = np.linalg.norm(relative_points[9])

    if scale == 0:
        scale = np.max(np.abs(relative_points))

    if scale > 0:
        relative_points = relative_points / scale

    return relative_points.reshape(63)


# ==========================================
# PROCESS HAND
# ==========================================

def image_processed(hand_img):

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(
        hand_img,
        cv2.COLOR_BGR2RGB
    )

    # Flip image horizontally
    img_flip = cv2.flip(img_rgb, 1)

    # Process image
    output = hands.process(img_flip)

    # No hand detected
    if not output.multi_hand_landmarks:

        return None

    # Get first hand
    hand_landmarks = output.multi_hand_landmarks[0]

    data = []

    # Extract 21 landmarks
    for landmark in hand_landmarks.landmark:

        data.append(landmark.x)
        data.append(landmark.y)
        data.append(landmark.z)

    return np.array(data, dtype=np.float32)


# ==========================================
# GET LETTER
# ==========================================

def getLetter(result):

    try:

        result = int(result)

        return classLabels[result]

    except:

        return "Error"


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot open camera")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:

        print("Can't receive frame")
        break


    # ======================================
    # PROCESS IMAGE
    # ======================================

    data = image_processed(frame)


    if data is not None:

        # Make shape:
        # (63,)
        # ->
        # (1,63,1)

        data = normalize_landmarks(data).reshape(1, 63, 1)


        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            data,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]

        confidence = np.max(prediction)


        letter = getLetter(predicted_class)


        # ==================================
        # DISPLAY
        # ==================================

        text = f"{letter} ({confidence * 100:.1f}%)"


        cv2.putText(
            frame,
            text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            4,
            cv2.LINE_AA
        )


        # Draw landmarks

        img_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(img_rgb)


    else:

        cv2.putText(
            frame,
            "No hand detected",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Sign Language Detection",
        frame
    )


    # Press Q to exit

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()