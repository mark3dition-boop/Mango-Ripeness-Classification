# Mango Ripeness Classification

Mango Ripeness Classification is a computer vision project that classifies mango images into ripeness categories using color based feature extraction and a Support Vector Machine (SVM) classifier.

The system uses OpenCV to segment mango regions from an image, extracts mean HSV color features, scales the features with `StandardScaler`, and predicts the ripeness class with an SVM model. A Streamlit web app is included so users can upload a mango image and view the predicted ripeness class together with the segmentation mask.

## Features

- Classifies mango ripeness from uploaded images.
- Uses HSV color thresholding for mango segmentation.
- Extracts three color features: mean Hue, mean Saturation, and mean Value.
- Uses an SVM classifier with an RBF kernel.
- Includes image augmentation during training.
- Provides a Streamlit interface for simple deployment and demonstration.
- Shows both the uploaded image and the generated segmentation mask.

## Project Structure

```text
Mango-Ripeness-Classification/
|-- app.py              # Streamlit web app for prediction
|-- main.py             # Training and evaluation script
|-- aug.pkl             # Saved SVM model and scaler trained with augmentation
|-- requirements.txt    # Python dependencies for local/cloud deployment
|-- README.md           # Project documentation
|-- unripe.jpg          # Example test image
|-- part-ripe.jpg       # Example test image
|-- ripe.png            # Example test image
`-- rott.png            # Example test image
```

If training from the dataset, the expected dataset folder structure is:

```text
Dataset/
|-- Train/
|   |-- Partially_ripe/
|   |-- Ripe/
|   |-- Rotten/
|   `-- Unripe/
`-- Test/
    |-- Partially_ripe/
    |-- Ripe/
    |-- Rotten/
    `-- Unripe/
```

## Dataset

Dataset source:

```text
https://drive.google.com/file/d/1WvyiAMiT9P_-DWqnM0JV52thGWM0_RX6/view?usp=sharing
```

The project uses four ripeness classes:

- `Unripe`
- `Partially_ripe`
- `Ripe`
- `Rotten`

The Streamlit app can run directly with the included `aug.pkl` model file. The dataset is only required if you want to retrain the model using `main.py`.

## Method Overview

The classification pipeline follows these steps:

1. Read mango image data with OpenCV.
2. Convert images from BGR to RGB.
3. Apply augmentation to training images.
4. Segment mango-colored regions using HSV thresholding.
5. Extract mean HSV values from the segmented region.
6. Scale the extracted features using `StandardScaler`.
7. Train an SVM classifier with an RBF kernel.
8. Predict the ripeness class of a new uploaded image.

## Image Preprocessing

The preprocessing stage applies:

- Horizontal flip
- Vertical flip
- 90 degree rotation
- 180 degree rotation
- 270 degree rotation
- Brightness increase
- Brightness decrease
- Gaussian blur before HSV segmentation

Augmentation is only applied to training data. Test data is not augmented.

## Model

The model used in this project is:

```python
SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)
```

The saved model file `aug.pkl` contains:

```python
(svm_model, scaler)
```

where:

- `svm_model` is the trained SVM classifier.
- `scaler` is the fitted `StandardScaler` used to normalize HSV features.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- Streamlit
- OpenCV
- NumPy
- scikit-learn
- Pillow

## Run the Streamlit App Locally

From the project folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Upload a mango image in `.jpg`, `.jpeg`, or `.png` format. The app will show:

- Uploaded input image
- Segmentation mask
- Predicted ripeness class

## Train or Retrain the Model

To retrain the model, make sure the dataset is placed in the expected `Dataset/Train` and `Dataset/Test` folders.

Then run:

```bash
python main.py
```

`main.py` will:

- Load the dataset.
- Train the augmented SVM model if `aug.pkl` is missing.
- Train the non-augmented SVM model if `default.pkl` is missing.
- Print evaluation metrics.
- Save trained models as `.pkl` files.

Important note: if `default.pkl` is not available, `main.py` will try to train it from the dataset. That means running `main.py` requires the dataset folder unless all needed model files already exist.

## Deploy to Streamlit Cloud

This repository is ready for Streamlit Cloud deployment because it includes:

- `app.py`
- `requirements.txt`
- `aug.pkl`

Deployment steps:

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app.
4. Select this GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

Because `aug.pkl` is included in the repository, the cloud app can load the trained model directly without needing the dataset folder.

## Usage Example

1. Start the Streamlit app.
2. Upload an image of a mango.
3. The app segments the mango area using HSV color thresholding.
4. The mean HSV values are extracted from the segmented pixels.
5. The trained SVM model predicts the ripeness class.
6. The prediction result is shown in the app.

## Evaluation Metrics

The training script supports these evaluation metrics:

- Accuracy
- Weighted recall
- Weighted precision
- Weighted F1-score

These metrics are calculated using scikit-learn after predicting the test set labels.

## Limitations

The current approach is lightweight and easy to explain, but it has several limitations:

- Prediction depends heavily on image lighting.
- Complex backgrounds can affect HSV segmentation.
- The model uses only average HSV values, so it may miss texture or local defects.
- The model may perform worse on mango varieties or image conditions not represented in the training data.
- The app does not currently show prediction probabilities because the SVM model is not configured with probability output.

## Future Improvements

Possible improvements:

- Add confusion matrix and per-class evaluation output.
- Add better background removal.
- Add color constancy or lighting normalization.
- Use color histograms or texture features instead of only mean HSV values.
- Compare SVM performance with CNN-based models.
- Add model confidence output by enabling SVM probability estimates.
- Add clearer error handling when no mango region is detected.

## Team

Group 7 - Computer Vision Final Project

## License

This repository is intended for academic project use. Check the dataset source for dataset-specific license or usage restrictions.
