import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle

model_path_aug = 'aug.pkl'
model_path_default = 'default.pkl'


def augment_image(image):
    """Menghasilkan variasi augmentasi dari satu gambar."""
    augmented = []

    # 1. Gambar asli
    augmented.append(image)

    # 2. Flip horizontal
    augmented.append(cv2.flip(image, 1))

    # 3. Flip vertikal
    augmented.append(cv2.flip(image, 0))

    # 4. Rotasi 90°
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    # 5. Rotasi 180°
    augmented.append(cv2.rotate(image, cv2.ROTATE_180))

    # 6. Rotasi 270°
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 7. Brightness lebih terang
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented.append(bright)

    # 8. Brightness lebih gelap
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
    augmented.append(dark)

    return augmented


def import_data(path, X_container, y_container, augment=False):

    for label in os.listdir(path):

        label_path = os.path.join(path, label)

        if os.path.isdir(label_path):

            for file in os.listdir(label_path):

                img_path = os.path.join(label_path, file)

                # Read image
                img = cv2.imread(img_path)

                # Convert color from BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if augment:
                    # Augmentasi hanya untuk data train
                    augmented_images = augment_image(img)
                    for aug_img in augmented_images:
                        X_container.append(aug_img)
                        y_container.append(label)
                else:
                    X_container.append(img)
                    y_container.append(label)


def segment_mango(image):
    blurred = cv2.GaussianBlur(image, (5,5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([10,30,30]),
        np.array([95,255,255])
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([0,30,30]),
        np.array([10,255,255])
    )

    mask = cv2.bitwise_or(mask1, mask2)

    return mask


def extract_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mask_pixels = mask > 0

    # Mean
    h_mean = np.mean(hsv[:,:,0][mask_pixels])
    s_mean = np.mean(hsv[:,:,1][mask_pixels])
    v_mean = np.mean(hsv[:,:,2][mask_pixels])

    return [
        h_mean,
        s_mean,
        v_mean,
    ]

def assign(X, container):
    for image in X:
        mask = segment_mango(image)
        features = extract_features(image, mask)
        container.append(features)


train_dir = "Dataset/Train"
test_dir = "Dataset/Test"

def load_or_train(isAug):

    model_path = "aug.pkl" if isAug else "default.pkl"

    # Jika model sudah ada, langsung load
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as file:
            return pickle.load(file)

    print("Model not found. Training new model...")

    # =============================
    # Load Dataset
    # =============================
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    import_data(train_dir, X_train, y_train, augment=isAug)
    import_data(test_dir, X_test, y_test, augment=False)

    # =============================
    # Feature Extraction
    # =============================
    X_train_features = []
    X_test_features = []

    assign(X_train, X_train_features)
    assign(X_test, X_test_features)

    X_train = np.array(X_train_features)
    y_train = np.array(y_train)

    X_test = np.array(X_test_features)
    y_test = np.array(y_test)

    # =============================
    # Scaling
    # =============================
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =============================
    # Train
    # =============================
    svm_model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    svm_model.fit(X_train, y_train)

    # =============================
    # Evaluation
    # =============================
    y_pred = svm_model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("F1-Score :", f1_score(y_test, y_pred, average="weighted"))


    # =============================
    # Save Model
    # =============================
    with open(model_path, "wb") as file:
        pickle.dump((svm_model, scaler), file)

    return svm_model, scaler



# Test outside data
print("--- SVC Model with Augmentation ---")
svc_aug, scaler_aug = load_or_train(True)
print("done")
print("--------------------------------------\n")

print("--- SVC Model without Augmentation ---")
svc_default, scaler_default = load_or_train(False)
print("done")
print("--------------------------------------\n")

path = 'rott.png'

test = cv2.imread(path)

test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

mask = segment_mango(test)

features = extract_features(test, mask)

features = np.array(features).reshape(1, -1)

features = scaler_aug.transform(features)

pred = svc_aug.predict(features)

print(f"Prediksi tingkat kematangan", path, ": ", pred)
