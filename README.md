# EEG-Based Alcoholism Detection using Machine Learning and Deep Learning

This project focuses on classifying alcoholic and control subjects using EEG signals from the SMNI_CMI dataset. The system implements a complete machine learning pipeline, including EEG preprocessing, feature extraction, model training, evaluation, model export, and a web-based dashboard for result visualization and inference.

## Project Overview

Electroencephalography (EEG) signals provide valuable information about brain activity and can be used to support the analysis of neurological and behavioral patterns. In this project, EEG recordings are processed and transformed into machine learning features for binary classification between alcoholic and non-alcoholic subjects.

The main objective of this project is to build an end-to-end EEG classification system that combines traditional machine learning models, deep learning approaches, and an interactive dashboard for model comparison and demonstration.

## Key Features

* End-to-end EEG classification pipeline
* EEG signal preprocessing and feature extraction
* Bandpower-based feature representation
* Spectrogram-based representation for deep learning models
* Train/test leakage checking using file hashing
* Outlier handling using quantile clipping
* Feature scaling using RobustScaler
* Multiple machine learning and deep learning models
* Evaluation using classification metrics and ROC-AUC
* Model export for later inference
* FastAPI backend for model serving
* React-based web dashboard for visualization and prediction

## Dataset

The project uses the SMNI_CMI EEG dataset, organized into separate training and testing folders:

```text
SMNI_CMI_TRAIN/
SMNI_CMI_TEST/
```

The pipeline uses the training set for model fitting and the testing set for evaluation. Duplicate content between train and test data can be detected and removed using MD5-based leakage diagnostics.

## Project Structure

```text
.
├── SMNI_CMI_TRAIN/          # Training EEG data
├── SMNI_CMI_TEST/           # Testing EEG data
├── data/                    # Additional data files
├── saved_models/            # Exported models, scalers, metrics, and EDA outputs
├── src/                     # Main Python source code
│   ├── app.py               # FastAPI backend
│   ├── config.py            # Project configuration
│   ├── diagnostics.py       # Leakage, data quality, and distribution diagnostics
│   ├── eda.py               # Exploratory data analysis utilities
│   ├── evaluation.py        # Model evaluation and export functions
│   ├── feature_engineering.py
│   ├── main.py              # End-to-end training pipeline
│   ├── preprocessing.py     # EEG preprocessing and feature extraction
│   └── model_*.py           # Model training modules
├── web/                     # React dashboard
├── requirements.txt         # Python dependencies
└── README.md
```

## Methodology

The system follows a structured EEG classification workflow:

1. Load EEG files from the training and testing directories.
2. Analyze class distribution.
3. Check for train-test leakage using MD5 hashing.
4. Extract EEG features from raw signal files.
5. Build bandpower feature vectors and spectrogram representations.
6. Apply outlier clipping based on training-set quantiles.
7. Normalize bandpower features using RobustScaler.
8. Train multiple machine learning and deep learning models.
9. Evaluate model performance on the test set.
10. Export trained models, scalers, metrics, and EDA summaries.
11. Serve results through a FastAPI backend and React dashboard.

## Models

The project compares several model families:

### Traditional Machine Learning

* Logistic Regression
* K-Nearest Neighbors
* Random Forest
* XGBoost
* Multi-Layer Perceptron

### Deep Learning

* CNN on spectrogram features
* EfficientNetB0 Transfer Learning on spectrogram images

## Evaluation

The models are evaluated using binary classification metrics, including:

* Accuracy
* ROC-AUC
* Predicted labels
* Prediction probabilities
* Model comparison based on test-set performance

The pipeline also generates additional diagnostic information, including:

* Train/test class distribution
* Data leakage report
* Distribution shift analysis
* Data quality checks
* Feature importance for interpretable models
* Bandpower summaries for EEG channels

## Installation

Clone the repository:

```bash
git clone https://github.com/zudoan/EEG-Alcoholism-Classification.git
cd EEG-Alcoholism-Classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Training Pipeline

Run the full EEG classification pipeline:

```bash
python -m src.main
```

After execution, the trained models, scalers, metrics, and EDA outputs will be saved in:

```text
saved_models/
```

## Running the API

Start the FastAPI backend:

```bash
python -m uvicorn src.app:app --reload --port 8000
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Running the Web Dashboard

Open a second terminal and run:

```bash
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Dashboard Functions

The web dashboard provides:

* Model performance visualization
* Metrics comparison
* Exploratory data analysis summaries
* EEG feature insights
* Available model listing
* File upload for inference
* Prediction result display

## Technologies Used

### Machine Learning and Data Processing

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Joblib

### Backend

* FastAPI
* Uvicorn

### Frontend

* React
* TypeScript
* Vite

## Notes

* The project uses `SMNI_CMI_TRAIN` as the training set and `SMNI_CMI_TEST` as the testing set.
* MD5-based duplicate checking is included to reduce the risk of train-test leakage.
* Traditional machine learning models use bandpower features.
* Deep learning models use spectrogram-based representations.
* TensorFlow-based models can be enabled or disabled depending on the runtime environment.

## Author

**Doan Anh Vu**
AI and Data Science Student
Thuy Loi University - Southern Campus

## Disclaimer

This project is developed for educational and research purposes. It is not intended for clinical diagnosis or medical decision-making.
