from pathlib import Path
import urllib.request

import cv2

try:
    import mediapipe.tasks.python.vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, drawing_utils
    from mediapipe.tasks.python.vision.core import image as mp_image
except ImportError:
    mp_vision = None
    HandLandmarker = None
    HandLandmarkerOptions = None
    drawing_utils = None
    mp_image = None
    BaseOptions = None

# ---------- Face detection setup ----------
face_cascade_path = (
    Path(__file__).resolve().parents[1]
    / "Machine Learning"
    / "Face Detection"
    / "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
if face_cascade.empty():
    raise FileNotFoundError(f"Could not load face cascade: {face_cascade_path}")

# ---------- Hand / finger detection setup ----------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")
hand_landmarker = None


def load_hand_landmarker():
    global hand_landmarker
    if mp_vision is None or HandLandmarker is None or BaseOptions is None or mp_image is None:
        print("MediaPipe tasks API is not available in this environment; hand tracking is disabled.")
        return None

    if not MODEL_PATH.exists():
        try:
            print(f"Downloading hand model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:  # pragma: no cover - runtime/network issue only
            print(f"Could not download hand model: {exc}")
            return None

    try:
        hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.6,
            )
        )
        return hand_landmarker
    except Exception as exc:  # pragma: no cover - runtime/model issue only
        print(f"Could not initialize hand landmarker: {exc}")
        return None


hand_landmarker = load_hand_landmarker()

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_IDS = [4, 8, 12, 16, 20]  # fingertip landmarks


def fingers_up(hand_landmarks, hand_label):
    """Return a list of 5 booleans: [thumb, index, middle, ring, pinky]."""
    landmark_list = getattr(hand_landmarks, "landmark", hand_landmarks)
    if not landmark_list:
        return [False] * 5

    up = []

    # Thumb: compare x instead of y (depends on which hand it is)
    if hand_label == "Right":
        up.append(landmark_list[4].x < landmark_list[3].x)
    else:
        up.append(landmark_list[4].x > landmark_list[3].x)

    # Other four fingers: tip is above the middle joint (smaller y = higher)
    for tip in TIP_IDS[1:]:
        up.append(landmark_list[tip].y < landmark_list[tip - 2].y)

    return up


# ---------- Camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_state = [False] * 5
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view feels natural

        # Faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hands
        current_state = [False] * 5
        if hand_landmarker is not None and drawing_utils is not None:
            mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=frame)
            result = hand_landmarker.detect_for_video(
                mp_frame,
                int(frame_counter * 1000 / 30),
            )

            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name if handedness else "Right"
                    current_state = fingers_up(hand_landmarks, label)
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        connections=mp_vision.HandLandmarksConnections,
                    )
        else:
            cv2.putText(
                frame,
                "Hand tracking disabled: MediaPipe hand model unavailable",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Detect the moment a finger goes up or down
        for i, name in enumerate(FINGER_NAMES):
            if current_state[i] and not previous_state[i]:
                print(f"{name} finger lifted")
            elif not current_state[i] and previous_state[i]:
                print(f"{name} finger lowered")
        previous_state = current_state

        # On-screen text
        raised = [n for n, u in zip(FINGER_NAMES, current_state) if u]
        text = f"Fingers up: {', '.join(raised) if raised else 'none'}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {sum(current_state)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
            break

        frame_counter += 1
finally:
    if hand_landmarker is not None:
        hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()