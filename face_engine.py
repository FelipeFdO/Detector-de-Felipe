import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
CLASSIFIER_PATH = MODELS_DIR / "felipe_classifier.joblib"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_bgr(image_path: Path, max_side: int = 1280) -> np.ndarray:
    """Load an image safely from Unicode paths, fix EXIF orientation, and resize if huge."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_detector(score_threshold: float = 0.7) -> cv2.FaceDetectorYN:
    if not YUNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo YuNet não encontrado em: {YUNET_MODEL_PATH}")
    # OpenCV C++ ONNX loader on Windows fails on non-ASCII absolute paths; load via relative path from BASE_DIR
    prev_cwd = Path.cwd()
    try:
        os.chdir(BASE_DIR)
        return cv2.FaceDetectorYN.create(
            model="models/face_detection_yunet_2023mar.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
    finally:
        os.chdir(prev_cwd)


def create_recognizer() -> cv2.FaceRecognizerSF:
    if not SFACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo SFace não encontrado em: {SFACE_MODEL_PATH}")
    prev_cwd = Path.cwd()
    try:
        os.chdir(BASE_DIR)
        return cv2.FaceRecognizerSF.create(
            model="models/face_recognition_sface_2021dec.onnx",
            config="",
        )
    finally:
        os.chdir(prev_cwd)


def detect_faces(image_bgr: np.ndarray, detector: cv2.FaceDetectorYN) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def extract_embedding(
    image_bgr: np.ndarray,
    face_row: np.ndarray,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray:
    aligned = recognizer.alignCrop(image_bgr, face_row)
    feat = recognizer.feature(aligned).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def extract_largest_face_embedding(
    image_bgr: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray | None:
    faces = detect_faces(image_bgr, detector)
    if len(faces) == 0:
        return None
    largest_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    return extract_embedding(image_bgr, largest_face, recognizer)
