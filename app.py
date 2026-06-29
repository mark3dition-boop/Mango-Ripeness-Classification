from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "Dataset"
TRAIN_DIR = DATASET_DIR / "Train"
TEST_DIR = DATASET_DIR / "Test"
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "mango_svm.joblib"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_image(image):
    augmented = [
        image,
        cv2.flip(image, 1),
        cv2.flip(image, 0),
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.convertScaleAbs(image, alpha=1.2, beta=30),
        cv2.convertScaleAbs(image, alpha=0.8, beta=-30),
    ]
    return augmented


def read_image_rgb(path):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_dataset(split_dir, augment=False):
    images = []
    labels = []

    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = read_image_rgb(image_path)
            if augment:
                for augmented_image in augment_image(image):
                    images.append(augmented_image)
                    labels.append(label_dir.name)
            else:
                images.append(image)
                labels.append(label_dir.name)

    return images, np.array(labels)


def segment_mango(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([95, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([10, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


def extract_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_pixels = mask > 0

    if not np.any(mask_pixels):
        raise ValueError("No mango-colored region was detected. Try a clearer image with a simpler background.")

    return np.array(
        [
            np.mean(hsv[:, :, 0][mask_pixels]),
            np.mean(hsv[:, :, 1][mask_pixels]),
            np.mean(hsv[:, :, 2][mask_pixels]),
        ],
        dtype=np.float64,
    )


def build_feature_matrix(images):
    features = []
    for image in images:
        mask = segment_mango(image)
        features.append(extract_features(image, mask))
    return np.array(features)


def train_model(train_dir, test_dir):
    train_images, y_train = load_dataset(train_dir, augment=True)
    test_images, y_test = load_dataset(test_dir, augment=False)

    if len(train_images) == 0 or len(test_images) == 0:
        raise ValueError("Dataset folders are empty or image files were not found.")

    X_train = build_feature_matrix(train_images)
    X_test = build_feature_matrix(test_images)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_samples_after_augmentation": len(train_images),
        "test_samples": len(test_images),
    }

    return {"model": model, "scaler": scaler, "metrics": metrics, "labels": sorted(set(y_train))}


def save_bundle(bundle):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_image(bundle, image):
    mask = segment_mango(image)
    features = extract_features(image, mask).reshape(1, -1)
    scaled_features = bundle["scaler"].transform(features)
    prediction = bundle["model"].predict(scaled_features)[0]

    decision_scores = None
    if hasattr(bundle["model"], "decision_function"):
        scores = bundle["model"].decision_function(scaled_features)
        decision_scores = np.ravel(scores)

    return prediction, mask, features[0], decision_scores


def show_metrics(metrics):
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    cols[1].metric("Recall", f"{metrics['recall_weighted']:.2%}")
    cols[2].metric("Precision", f"{metrics['precision_weighted']:.2%}")
    cols[3].metric("F1-score", f"{metrics['f1_weighted']:.2%}")

    st.caption(
        f"Train samples after augmentation: {metrics['train_samples_after_augmentation']} | "
        f"Test samples: {metrics['test_samples']}"
    )


def main():
    st.set_page_config(page_title="Mango Ripeness Classifier", page_icon="🥭", layout="centered")
    st.title("Mango Ripeness Classifier")
    st.write("Classifies mango images into ripeness stages using HSV feature extraction and an SVM model.")

    bundle = load_bundle()
    dataset_available = TRAIN_DIR.exists() and TEST_DIR.exists()

    with st.sidebar:
        st.header("Model")
        st.write(f"Model artifact: `{MODEL_PATH.relative_to(APP_DIR)}`")

        if bundle is not None:
            st.success("Saved model loaded.")
        elif dataset_available:
            st.info("Dataset found. Train the model before prediction.")
        else:
            st.warning("No saved model or Dataset folder found.")

        if dataset_available and st.button("Train and save model"):
            with st.spinner("Training SVM model..."):
                trained_bundle = train_model(TRAIN_DIR, TEST_DIR)
                save_bundle(trained_bundle)
                st.cache_resource.clear()
                st.success("Model trained and saved. Refreshing app state...")
                st.rerun()

    if bundle is None:
        st.subheader("Setup required")
        st.write(
            "For local use, put the dataset in `Dataset/Train` and `Dataset/Test`, then click "
            "`Train and save model`. For Streamlit Cloud, commit `model/mango_svm.joblib` after "
            "training locally, or include the dataset folder in the repository."
        )
        st.stop()

    if "metrics" in bundle:
        st.subheader("Model evaluation")
        show_metrics(bundle["metrics"])

    st.subheader("Predict mango ripeness")
    uploaded_file = st.file_uploader("Upload a mango image", type=sorted(IMAGE_EXTENSIONS))

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to start prediction.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    try:
        prediction, mask, features, decision_scores = predict_image(bundle, image_np)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Predicted ripeness: {prediction}")

    col1, col2 = st.columns(2)
    col1.image(image_np, caption="Uploaded image", use_container_width=True)
    col2.image(mask, caption="HSV segmentation mask", use_container_width=True, clamp=True)

    st.subheader("Extracted HSV features")
    st.dataframe(
        {
            "Feature": ["Hue mean", "Saturation mean", "Value mean"],
            "Value": [round(float(value), 4) for value in features],
        },
        hide_index=True,
        use_container_width=True,
    )

    if decision_scores is not None:
        st.caption("Decision scores are SVM margin values, not probabilities.")
        st.write(np.round(decision_scores, 4))


if __name__ == "__main__":
    main()
