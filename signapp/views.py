from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from keras.models import load_model

from signapp.models import SignLanguages

import cv2
import mediapipe as mp
import numpy as np

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from rest_framework.decorators import api_view

import base64
import json
from pathlib import Path
from PIL import Image
from io import BytesIO


BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / 'static' / 'hdf' / 'label_classes.json'
with LABELS_PATH.open(encoding='utf-8') as labels_file:
    CLASS_LABELS = {
        int(class_id): label
        for class_id, label in json.load(labels_file).items()
    }


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


# =========================================================
# CREATE YOUR VIEWS HERE
# =========================================================


@login_required
def index(request):
    return render(request, "index.html")


@login_required
def home(request):
    return render(request, "home.html")


@login_required
def TextToSign(request):

    if request.method == 'POST':

        name = request.POST['text']

        value = SignLanguages.objects.filter(comment=name)

        if value:
            print("success")

            det = SignLanguages.objects.get(comment=name)

            return render(
                request,
                "texttosign.html",
                {'data': det}
            )

    return render(request, "texttosign.html")


@login_required
def tutorials(request):
    return render(request, "tutorials.html")


@login_required
def alphabets(request):
    return render(request, "alphabets.html")


@login_required
def numbers(request):
    return render(request, "numbers.html")


@login_required
def upload_image(request):
    return render(request, "test.html")


# =========================================================
# LOAD SIGN LANGUAGE MODEL
# =========================================================

model = load_model(
    str(BASE_DIR / 'static' / 'hdf' / 'cnn_model_final1.h5')
)


# =========================================================
# SIGN TO TEXT / IMAGE DETECTION
# =========================================================

@api_view(['POST'])
def detect(request):

    try:

        # -------------------------------------------------
        # GET IMAGE DATA FROM FRONTEND
        # -------------------------------------------------

        image_data = request.data.get('image_data')

        if not image_data:

            return JsonResponse(
                {
                    'error': 'No image_data received'
                },
                status=400
            )


        # -------------------------------------------------
        # CHECK BASE64 IMAGE FORMAT
        # -------------------------------------------------

        if ',' not in image_data:

            return JsonResponse(
                {
                    'error': 'Invalid image data format'
                },
                status=400
            )


        # -------------------------------------------------
        # DECODE BASE64 IMAGE
        # -------------------------------------------------

        decoded_image_data = base64.b64decode(
            image_data.split(',', 1)[1]
        )


        # -------------------------------------------------
        # CONVERT BASE64 TO PIL IMAGE
        # -------------------------------------------------

        pil_image = Image.open(
            BytesIO(decoded_image_data)
        ).convert('RGB')


        # -------------------------------------------------
        # CONVERT PIL IMAGE TO OPENCV IMAGE
        # -------------------------------------------------

        open_cv_image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )


        # -------------------------------------------------
        # MEDIAPIPE HAND DETECTION
        # -------------------------------------------------

        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        ) as hands:

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(
                open_cv_image,
                cv2.COLOR_BGR2RGB
            )

            # Flip image horizontally
            img_flip = cv2.flip(
                img_rgb,
                1
            )

            # Process image
            output = hands.process(
                img_flip
            )


        # -------------------------------------------------
        # CHECK WHETHER HAND WAS DETECTED
        # -------------------------------------------------

        if not output.multi_hand_landmarks:

            print("No hand detected")

            return JsonResponse(
                {
                    'detected_gesture': 'Error',
                    'message': 'No hand detected'
                }
            )


        # -------------------------------------------------
        # GET HAND LANDMARKS
        # -------------------------------------------------

        hand_landmarks = output.multi_hand_landmarks[0]


        # -------------------------------------------------
        # EXTRACT X, Y AND Z VALUES
        # -------------------------------------------------

        landmarks = []


        for landmark in hand_landmarks.landmark:

            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)


        # Convert landmarks to the same normalized feature space used for training.
        data = np.array(
            landmarks,
            dtype=np.float32
        )


        print(
            "Landmark shape:",
            data.shape
        )


        # -------------------------------------------------
        # CHECK LANDMARK SHAPE
        # -------------------------------------------------

        if data.shape != (63,):

            return JsonResponse(
                {
                    'error': f'Invalid landmark shape: {data.shape}'
                },
                status=400
            )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            normalize_landmarks(data).reshape(1, 63, 1),
            verbose=0
        )


        # Get the class and confidence from the model output.
        predicted_class = int(
            np.argmax(prediction, axis=1)[0]
        )
        confidence = float(np.max(prediction))
        detected_gesture = CLASS_LABELS.get(predicted_class)

        if detected_gesture is None or confidence < 0.60:
            return JsonResponse({
                'detected_gesture': 'Error',
                'message': 'Sign is not recognized confidently',
                'confidence': confidence,
                'supported_signs': list(CLASS_LABELS.values()),
            })


        # -------------------------------------------------
        # PRINT RESULT IN TERMINAL
        # -------------------------------------------------

        print(
            "Predicted class:",
            predicted_class
        )

        print(
            "Detected gesture:",
            detected_gesture
        )


        # -------------------------------------------------
        # SEND RESULT BACK TO JAVASCRIPT
        # -------------------------------------------------

        result = {
            'detected_gesture': detected_gesture,
            'confidence': confidence,
        }


        return JsonResponse(result)


    except Exception as e:

        print(
            "DETECT ERROR:",
            str(e)
        )

        return JsonResponse(
            {
                'error': str(e)
            },
            status=400
        )


# =========================================================
# LOGIN
# =========================================================

def loginr(request):

    if request.method == "GET":

        return render(
            request,
            "login.html"
        )


    elif request.method == "POST":

        username = request.POST.get(
            'username'
        )

        password = request.POST.get(
            'password'
        )


        user = authenticate(
            request,
            username=username,
            password=password
        )


        if user is not None:

            login(
                request,
                user
            )

            return redirect("index")


        else:

            return HttpResponse(
                "Invalid credentials"
            )


# =========================================================
# REGISTER
# =========================================================

def registerr(request):

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        email = request.POST.get(
            'email'
        )

        password = request.POST.get(
            'password'
        )


        # Check empty fields
        if not (
            username and
            email and
            password
        ):

            return HttpResponse(
                "All fields are required"
            )


        # Check existing username/email
        if (
            User.objects.filter(
                username=username
            ).exists()
            or
            User.objects.filter(
                email=email
            ).exists()
        ):

            return HttpResponse(
                "Username or email already exists"
            )


        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )


        if user:

            return render(
                request,
                "login.html"
            )


        else:

            return HttpResponse(
                "Error creating user"
            )


    else:

        return HttpResponse(
            "Method not allowed"
        )
