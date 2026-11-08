# Face and Hand Detection API

A local HTTP API for detecting faces and hand landmarks in uploaded images. The API is separate from `Projects/Face_Recognisation.py`; it does not open or require a webcam.

## Requirements

- Python available through the repository's `.venv`
- Dependencies from [`requirements.txt`](requirements.txt)
- The Haar cascade at `Machine Learning/Face Detection/haarcascade_frontalface_default.xml`
- Network access on first startup if `Projects/hand_landmarker.task` is not already present

MediaPipe is pinned to version `0.10.35`. This version was tested with the hand-tracking model on the target machine. Do not replace it with an unpinned latest version without checking CPU compatibility.

## Install

Run these commands from the repository root:

```bash
.venv/bin/python -m pip install -r Projects/face_recognition_api/requirements.txt
```

If the repository virtual environment does not exist yet, create it with a Python version supported by the installed packages, then install the requirements above.

## Start the server

Run from the repository root:

```bash
.venv/bin/python -m uvicorn app:app --app-dir Projects/face_recognition_api --host 127.0.0.1 --port 8000
```

The first startup downloads the MediaPipe hand-landmarker model to `Projects/hand_landmarker.task` if it is missing. Startup fails if the face cascade, model, or detector cannot be initialized.

The server is available at `http://127.0.0.1:8000`.

## Interactive documentation

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Endpoints

### `GET /health`

Returns HTTP 200 after application startup:

```json
{"status":"ok"}
```

### `POST /detect`

Send one image using `multipart/form-data` with the field name `image`.

Example using cURL:

```bash
curl -X POST \
  -F "image=@photo.jpg" \
  http://127.0.0.1:8000/detect
```

The request body is limited to 10 MiB. The server decodes the uploaded bytes as an image; the filename and browser-provided content type are not used to determine the image format.

Successful response fields:

| Field | Description |
| --- | --- |
| `filename` | Name supplied for the uploaded file, or `null`. |
| `image_size` | Decoded image width and height in pixels. |
| `faces` | Face bounding boxes, each with top-left `x`, `y`, `width`, and `height` in pixels. Empty when no face is found. |
| `hands` | Detected hands. Empty when no hand is found. The current model configuration detects at most one hand. |

Each hand contains:

| Field | Description |
| --- | --- |
| `handedness` | MediaPipe hand label, normally `Left` or `Right`. |
| `confidence` | MediaPipe handedness score from 0 to 1. |
| `fingers_up` | Boolean values for `thumb`, `index`, `middle`, `ring`, and `pinky`. |
| `raised_count` | Number of fingers marked as raised by the simple landmark heuristic. |
| `landmarks` | The 21 hand landmarks. `x` and `y` are normalized image coordinates; `z` is MediaPipe's relative depth value. |

Example response with no detections:

```json
{
  "filename": "photo.jpg",
  "image_size": {"width": 1280, "height": 720},
  "faces": [],
  "hands": []
}
```

### Error responses

| Status | Meaning |
| --- | --- |
| `413` | Uploaded image is larger than 10 MiB. |
| `415` | Uploaded bytes could not be decoded as a supported image. |
| `422` | Required multipart field `image` is missing or malformed. |
| `500` | Unexpected detection or server error. |

Errors use FastAPI's JSON `detail` response format.

## Detection behavior and limitations

- Faces are detected with the repository's OpenCV Haar cascade.
- Hands are detected with MediaPipe Tasks in image mode. Inference is serialized so the shared detector is not called concurrently.
- Finger-up values use a geometric heuristic over the hand landmarks. They are estimates, not gesture-recognition classifications.
- MediaPipe handedness assumes a mirrored selfie-style input. For unmirrored photos, handedness and the thumb-up estimate may be reversed.
- Face boxes use image pixels. Hand `x` and `y` landmarks are normalized to the image dimensions.
- The API returns JSON only; it does not return an annotated image.
- Uploaded image data is not saved as a permanent application file. The hand model is saved locally for subsequent startups.

## Local security

The documented launch command binds to `127.0.0.1`, making the service available only on the local machine. The API has no authentication or authorization. Before binding to a public or shared network interface, add appropriate authentication, transport security, request controls, and deployment configuration.

## Troubleshooting

- **Cannot import dependencies:** install `requirements.txt` using the repository `.venv`, and run the server with that same environment.
- **MediaPipe crashes with an AVX/illegal-instruction error:** reinstall the pinned `mediapipe==0.10.35` dependency; a newer wheel may not support the processor.
- **Startup cannot find the face cascade:** run the server from this repository checkout and check the cascade file path listed under Requirements.
- **Hand model download fails:** ensure network access to `storage.googleapis.com`, or place a valid hand-landmarker task model at `Projects/hand_landmarker.task` before startup.
- **The webcam script behaves differently:** it is a separate application. This API analyzes uploaded images and never opens a camera.