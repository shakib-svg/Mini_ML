# Project Overview

This repository contains a complete pipeline for processing underwater acoustic recordings and classifying spectrogram images using machine learning models. The workflow includes:

1. **Temporal Splitting**: Extract annotated segments from long WAV recordings (split.py)
2. **Time-Frequency Cropping**: Apply frequency-band cropping via STFT and ISTFT (split_freq.py)
3. **Spectrogram Generation**: Convert cropped WAV segments to 100×100 PNG spectrogram images (wav_to_image.py)
4. **Aggregation by Class**: Consolidate spectrograms into class-based folders (rassemblage_class.py)
5. **Machine Learning Classification**: Train and evaluate various classifiers (XGBoost, Random Forest, LightGBM, Linear SVC)

---

## Repository Structure

```
├── annotations/            # CSV files with labels and event times
├── audio/                  # Raw WAV recordings by dataset and class
├── train_1/                # Split WAV segments (by time)
├── train_2/                # Time-frequency cropped WAV segments
├── spectrograms_100x100/   # Generated spectrogram images
├── by_class/               # Spectrograms aggregated by class
├── final_train/
│   ├── train/              # Class subfolders with PNGs for training
│   └── test/               # Class subfolders with PNGs for testing
├── split.py                # Script: extract temporal segments
├── split_freq.py           # Script: STFT/ISTFT frequency cropping
├── wav_to_image.py         # Script: PNG spectrogram generation
├── rassemblage_class.py    # Script: aggregate images by class
├── train_xgboost.py        # XGBoost classification script
├── train_random_forest.py  # Random Forest classification script
├── train_lightgbm.py       # LightGBM classification script
├── train_linear_svc.py     # Linear SVC classification script
└── README.md               # This documentation
```

---

## Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install numpy pandas scipy matplotlib pillow librosa soundfile scikit-learn xgboost lightgbm
  ```
  - For GPU support in XGBoost / LightGBM, install the GPU-enabled versions.

---

## 1. Temporal Splitting (`split.py`)

Extract annotated events from raw WAV files:
```bash
python split.py \
  --csv path/to/annotations.csv \
  --audio_root path/to/audio/ \
  --output_root path/to/train_1/
```
- **Input**: CSV with `dataset,filename,annotation,low_freq,high_freq,start_datetime,end_datetime` columns
- **Output**: WAV segments saved under `train_1/<dataset>/<annotation>/<timestamp>.wav`

---

## 2. Time-Frequency Cropping (`split_freq.py`)

Perform STFT, zero-out outside [low_freq,high_freq], invert via ISTFT:
```bash
python split_freq.py \
  --csv path/to/annotations.csv \
  --trimmed_root path/to/train_1/ \
  --output_root path/to/train_2/ \
  --updated_csv path/to/train_2/annotations_with_tf.csv
```
- **Output**: Cropped WAVs under `train_2/<annotation>/` and updated CSV with `tf_cropped_wav` column

---

## 3. Spectrogram Generation (`wav_to_image.py`)

Convert each cropped WAV to a 100×100 colored spectrogram PNG:
```bash
python wav_to_image.py
```
- **Reads**: `train_2/audio/<dataset>/<annotation>/<wav_fn>`
- **Writes**: `spectrograms_100x100/<dataset>/<annotation>/<wav_fn>.png`

---

## 4. Aggregate by Class (`rassemblage_class.py`)

Group all spectrograms into class-specific folders:
```bash
python rassemblage_class.py
```
- **Reads**: `spectrograms_100x100/<dataset>/<annotation>/*.png`
- **Writes**: `by_class/<annotation>/*.png`

---

## 5. Classification Scripts

### XGBoost (`train_xgboost.py`)
```bash
python train_xgboost.py
```
- GPU-enabled: sets `tree_method='gpu_hist'` & `predictor='gpu_predictor'`

### Random Forest (`train_random_forest.py`)
```bash
python train_random_forest.py
```
- Uses `RandomForestClassifier(verbose=1, n_jobs=-1)`

### LightGBM (`train_lightgbm.py`)
```bash
python train_lightgbm.py
```
- Logging with `verbose` in constructor and callbacks

### Linear SVC (`train_linear_svc.py`)
```bash
python train_linear_svc.py
```
- Fast linear model: `LinearSVC(dual=False)` for high-dimensional data

Each script:
- Loads `final_train/train` and `final_train/test`
- Prints F2-score (macro), classification report, and confusion matrix

---

## Evaluation Metrics

- **F2 Score** (macro-average)
- **Classification Report**: Precision, Recall, F1 per class
- **Confusion Matrix**: Visual error analysis

---

## Notes

- Adjust paths at top of each script to your environment.
- For GPU training, ensure proper installation of GPU-enabled libraries.
- Experiment with model hyperparameters for best performance.

---

*End of README*

