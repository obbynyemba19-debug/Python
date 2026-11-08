from contextlib import asynccontextmanager
from pathlib import Path
import threading
import urllib.request

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image


PROJECTS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECTS_DIR.parent
FACE_CASCADE_PATH = (
    REPOSITORY_DIR
    / "Machine Learning"
    / "Face Detection"
    / "haarcascade_frontalface_default.xml"
)
MODEL_PATH = PROJECTS_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_TIP_IDS = [4, 8, 12, 16, 20]
DETECTION_LOCK = threading.Lock()


class ImageSize(BaseModel):
    width: int
    height: int


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Point(BaseModel):
    x: float
    y: float
    z: float


class HandDetection(BaseModel):
    handedness: str
    confidence: float
    fingers_up: dict[str, bool]
    raised_count: int
    landmarks: list[Point]


class DetectionResponse(BaseModel):
    filename: str | None
    image_size: ImageSize
    faces: list[FaceBox]
    hands: list[HandDetection]


def _fingers_up(landmarks, handedness: str) -> list[bool]:
    if handedness == "Right":
        thumb_up = landmarks[4].x < landmarks[3].x
    else:
        thumb_up = landmarks[4].x > landmarks[3].x

    return [thumb_up] + [
        landmarks[tip].y < landmarks[tip - 2].y
        for tip in FINGER_TIP_IDS[1:]
    ]


def _analyze_image(frame, face_cascade, hand_landmarker) -> dict:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangles = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
    )
    faces = [
        {"x": int(x), "y": int(y), "width": int(box_width), "height": int(box_height)}
        for x, y, box_width, box_height in face_rectangles
    ]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
    result = hand_landmarker.detect(image)
    hands = []

    for index, landmarks in enumerate(result.hand_landmarks):
        handedness_items = result.handedness[index]
        category = handedness_items[0] if handedness_items else None
        label = category.category_name if category else "Right"
        confidence = float(category.score) if category else 0.0
        states = _fingers_up(landmarks, label)
        hands.append(
            {
                "handedness": label,
                "confidence": confidence,
                "fingers_up": dict(zip(FINGER_NAMES, states)),
                "raised_count": sum(states),
                "landmarks": [
                    {"x": point.x, "y": point.y, "z": point.z}
                    for point in landmarks
                ],
            }
        )

    return {
        "image_size": {"width": width, "height": height},
        "faces": faces,
        "hands": hands,
    }


def _analyze_safely(frame, face_cascade, hand_landmarker) -> dict:
    with DETECTION_LOCK:
        return _analyze_image(frame, face_cascade, hand_landmarker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not FACE_CASCADE_PATH.is_file():
        raise RuntimeError(f"Face cascade file not found: {FACE_CASCADE_PATH}")

    face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {FACE_CASCADE_PATH}")

    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    app.state.face_cascade = face_cascade
    app.state.hand_landmarker = HandLandmarker.create_from_options(options)
    try:
        yield
    finally:
        app.state.hand_landmarker.close()


app = FastAPI(
    title="Face and Hand Detection API",
    version="1.0.0",
    description=(
        "Analyze an uploaded image for face boxes and hand landmarks. "
        "Uploaded images are limited to 10 MiB."
    ),
    lifespan=lifespan,
)


@app.get("/health", summary="Check service health", tags=["status"])
async def health():
    return {"status": "ok"}


@app.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect faces and hands in an image",
    description=(
        "Upload one supported image as multipart form field `image`. "
        "The response contains pixel-coordinate face boxes and normalized hand landmarks."
    ),
    tags=["detection"],
)
async def detect_image(request: Request, image: UploadFile = File(...)):
    image_bytes = await image.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=415, detail="Upload must contain a supported image")

    result = await run_in_threadpool(
        _analyze_safely,
        frame,
        request.app.state.face_cascade,
        request.app.state.hand_landmarker,
    )
    result["filename"] = image.filename
    return result