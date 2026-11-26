from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm import tqdm

from face_engine import (
    BASE_DIR,
    CLASSIFIER_PATH,
    IMAGE_EXTENSIONS,
    create_detector,
    create_recognizer,
    extract_largest_face_embedding,
    load_image_bgr,
)

DATA_DIR = BASE_DIR / "Dados"
FELIPE_DIR = DATA_DIR / "Felipe"
NOT_FELIPE_DIR = DATA_DIR / "Não Felipe"


def collect_embeddings(
    folder: Path,
    label: int,
    detector,
    recognizer,
) -> tuple[list[np.ndarray], list[int], int, int]:
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for img_path in tqdm(files, desc=f"Processando '{folder.name}'", unit="img"):
        try:
            img_bgr = load_image_bgr(img_path)
            emb = extract_largest_face_embedding(img_bgr, detector, recognizer)
        except Exception as exc:
            print(f"[AVISO] Erro ao ler {img_path.name}: {exc}")
            skipped += 1
            continue

        if emb is None:
            skipped += 1
            continue

        embeddings.append(emb)
        labels.append(label)

    return embeddings, labels, len(files), skipped


def main() -> None:
    if not FELIPE_DIR.exists() or not NOT_FELIPE_DIR.exists():
        raise FileNotFoundError(
            f"Pastas esperadas não encontradas em {DATA_DIR} ('Felipe' e 'Não Felipe')."
        )

    detector = create_detector(score_threshold=0.65)
    recognizer = create_recognizer()

    emb_pos, y_pos, total_pos, skip_pos = collect_embeddings(
        FELIPE_DIR, 1, detector, recognizer
    )
    emb_neg, y_neg, total_neg, skip_neg = collect_embeddings(
        NOT_FELIPE_DIR, 0, detector, recognizer
    )

    print("\n=== Resumo da Extração de Rostos ===")
    print(f"Felipe:     {len(emb_pos)}/{total_pos} imagens com rosto detectado ({skip_pos} sem rosto/erro)")
    print(f"Não Felipe: {len(emb_neg)}/{total_neg} imagens com rosto detectado ({skip_neg} sem rosto/erro)")

    if len(emb_pos) < 5 or len(emb_neg) < 5:
        raise RuntimeError("Poucos rostos detectados para treinar o classificador com segurança.")

    X = np.vstack(emb_pos + emb_neg)
    y = np.array(y_pos + y_neg, dtype=np.int32)

    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred_cv)
    cm = confusion_matrix(y, y_pred_cv)

    print("\n=== Avaliação (Validação Cruzada 5-Fold) ===")
    print(f"Acurácia: {acc * 100:.2f}%")
    print("Matriz de Confusão [linhas=real (0=Não Felipe, 1=Felipe), colunas=previsto]:")
    print(cm)
    print("\nRelatório detalhado:")
    print(classification_report(y, y_pred_cv, target_names=["Não Felipe", "Felipe"]))

    # Treina modelo final em todos os dados
    clf.fit(X, y)

    felipe_centroid = np.mean(np.vstack(emb_pos), axis=0)
    felipe_centroid = felipe_centroid / np.linalg.norm(felipe_centroid)

    pos_sims = np.dot(np.vstack(emb_pos), felipe_centroid)
    neg_sims = np.dot(np.vstack(emb_neg), felipe_centroid)
    print(
        f"Similaridade cosseno com centróide do Felipe -> "
        f"Felipe média: {pos_sims.mean():.3f} (mín: {pos_sims.min():.3f}) | "
        f"Não Felipe média: {neg_sims.mean():.3f} (máx: {neg_sims.max():.3f})"
    )

    bundle = {
        "classifier": clf,
        "felipe_centroid": felipe_centroid.astype(np.float32),
        "prob_threshold": 0.65,
        "cv_accuracy": float(acc),
        "n_felipe_faces": len(emb_pos),
        "n_not_felipe_faces": len(emb_neg),
    }
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CLASSIFIER_PATH)
    print(f"\nModelo salvo com sucesso em: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
