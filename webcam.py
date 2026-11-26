import argparse
import time
from pathlib import Path
import cv2
import joblib
import numpy as np

from face_engine import (
    CLASSIFIER_PATH,
    create_detector,
    create_recognizer,
    detect_faces,
    extract_embedding,
    load_image_bgr,
)


def classify_face(
    image_bgr: np.ndarray,
    face_row: np.ndarray,
    recognizer: cv2.FaceRecognizerSF,
    bundle: dict,
) -> tuple[bool, float, float]:
    emb = extract_embedding(image_bgr, face_row, recognizer)
    clf = bundle["classifier"]
    centroid = bundle["felipe_centroid"]
    prob_threshold = bundle.get("prob_threshold", 0.65)
    min_cosine = bundle.get("min_cosine", 0.20)

    prob_felipe = float(clf.predict_proba([emb])[0, 1])
    cos_sim = float(np.dot(emb, centroid))
    is_felipe = (prob_felipe >= prob_threshold) and (cos_sim >= min_cosine)
    return is_felipe, prob_felipe, cos_sim


def draw_prediction(
    frame: np.ndarray,
    face_row: np.ndarray,
    is_felipe: bool,
    prob_felipe: float,
    cos_sim: float,
) -> None:
    x, y, w, h = face_row[:4].astype(int)
    x = max(0, x)
    y = max(0, y)

    if is_felipe:
        color = (40, 200, 60)  # Verde
        label = f"FELIPE ({prob_felipe * 100:.0f}%)"
    else:
        color = (50, 60, 230)  # Vermelho
        label = f"NAO E O FELIPE ({prob_felipe * 100:.0f}%)"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(y - 10, th + 10)
    cv2.rectangle(
        frame,
        (x, label_y - th - 8),
        (x + tw + 10, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 5, label_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    sim_text = f"sim: {cos_sim:.2f}"
    cv2.putText(
        frame,
        sim_text,
        (x, y + h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def run_on_image(image_path: Path, bundle: dict, detector, recognizer) -> None:
    img_bgr = load_image_bgr(image_path)
    faces = detect_faces(img_bgr, detector)
    if len(faces) == 0:
        print("Nenhum rosto detectado na imagem.")
    else:
        for i, face in enumerate(faces, start=1):
            is_felipe, prob, cos_sim = classify_face(img_bgr, face, recognizer, bundle)
            status = "FELIPE" if is_felipe else "NÃO É O FELIPE"
            print(f"Rosto {i}: {status} (prob={prob*100:.1f}%, sim={cos_sim:.3f})")
            draw_prediction(img_bgr, face, is_felipe, prob, cos_sim)

    cv2.imshow("Identificador de Felipe (Imagem)", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_faces_scaled(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
    max_det_width: int = 480,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_det_width:
        return detect_faces(frame, detector)

    scale = max_det_width / float(w)
    small = cv2.resize(frame, (max_det_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    faces = detect_faces(small, detector)
    if len(faces) > 0:
        faces = faces.copy()
        # Escala bbox (0:4) e landmarks (4:14) de volta ao frame original; mantém score (14)
        faces[:, :14] /= scale
    return faces


def run_webcam(
    camera_index: int,
    bundle: dict,
    detector,
    recognizer,
    width: int = 640,
    height: int = 480,
) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera (índice {camera_index}). "
            "Verifique se a webcam está conectada e não está em uso por outro app."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Câmera iniciada! Pressione 'q' ou 'ESC' na janela para sair.")
    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Falha ao capturar frame da câmera.")
                break

            # Espelha horizontalmente para parecer um espelho natural
            frame = cv2.flip(frame, 1)

            faces = detect_faces_scaled(frame, detector)
            for face in faces:
                is_felipe, prob, cos_sim = classify_face(frame, face, recognizer, bundle)
                draw_prediction(frame, face, is_felipe, prob, cos_sim)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)

            hud = f"FPS: {fps:.1f} | Rostos: {len(faces)} | Pressione Q ou ESC para sair"
            cv2.putText(
                frame,
                hud,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Identificador de Felipe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Identificador de Felipe via webcam ou imagem.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão: 0)")
    parser.add_argument("--image", type=Path, default=None, help="Caminho de uma imagem para testar")
    parser.add_argument("--width", type=int, default=640, help="Largura da câmera (padrão: 640)")
    parser.add_argument("--height", type=int, default=480, help="Altura da câmera (padrão: 480)")
    args = parser.parse_args()

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Modelo treinado não encontrado em {CLASSIFIER_PATH}. Execute 'python train.py' primeiro."
        )

    bundle = joblib.load(CLASSIFIER_PATH)
    detector = create_detector(score_threshold=0.70)
    recognizer = create_recognizer()

    if args.image is not None:
        run_on_image(args.image, bundle, detector, recognizer)
    else:
        run_webcam(args.camera, bundle, detector, recognizer, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
