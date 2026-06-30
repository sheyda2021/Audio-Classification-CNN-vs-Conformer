<div align="center">

# 🎙️ Audio Classification: CNN vs. Conformer

### A comparative study of CNN and Conformer architectures for short audio intent classification

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio-1DB954?style=flat-square)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#)

</div>

---

## 📌 Overview

This repository presents a comparative study of two deep learning architectures — **CNN** and **Conformer** — applied to short speech segment classification into two intent categories: **`identity`** and **`request`**.

> Audio classification models often behave differently depending on dataset size and noise conditions. While transformer-based models such as Conformer excel in large-scale speech recognition, they don't always outperform simpler architectures when training data is limited.

This project investigates that exact scenario through a controlled experiment using **Mel-spectrogram features**, comparing the two architectures across:

- 🎯 **Accuracy** — raw classification performance
- 🛡️ **Robustness** — behavior under noisy conditions
- 📊 **Data efficiency** — performance on small datasets
- 🏗️ **Architectural suitability** — fit for small-scale audio tasks

---

## 🗂️ Dataset

The dataset consists of short (~1 second) audio recordings labeled into two classes:

| Label | Description |
|---|---|
| `identity` | Speaker identification utterances |
| `request` | Request-type utterances |

Each sample is converted into a **Mel-spectrogram** representation before being fed into the models.

---

## 🔊 Audio Processing Pipeline

### 1️⃣ Waveform Normalization

The raw waveform is normalized to stabilize amplitude variations across samples.

<p align="center">
  <img src="Fig1_Normalized_Waveform.png" width="800">
</p>

### 2️⃣ Spectrogram Feature Extraction

The audio signal is converted from the time domain to the frequency domain using Mel-Spectrograms.

<p align="center">
  <img src="Fig2_Waveform_Spectrogram.png" width="900">
</p>

### 3️⃣ Class Spectrogram Comparison

A visual comparison of spectrogram patterns between the two classes — these patterns help explain why convolutional filters capture discriminative features effectively.

<p align="center">
  <img src="Fig4_MelSpec_Comparison.png" width="850">
</p>

---

## 🧠 Model Architectures

### 🔹 CNN Baseline

A convolutional neural network processing Mel-spectrogram images to learn local spectral patterns associated with each class.

**Advantages:**
- ⚡ Efficient training
- 🎯 Strong local feature extraction
- 📉 Good performance with smaller datasets

### 🔹 Conformer

Combines convolutional layers (local feature extraction) with self-attention layers (global context modeling). Conformers are widely used in large-scale speech recognition, but their effectiveness depends heavily on dataset size.

---

## 📈 Experimental Results

### K-Fold Accuracy

Cross-validation results show the **CNN model consistently outperformed the Conformer** on this dataset, maintaining accuracy above **98%** across folds.

<p align="center">
  <img src="Fig3_Classification_Results.png" width="700">
</p>

### Confusion Matrix Comparison

The CNN achieved near-perfect classification, while the Conformer showed higher confusion between the two classes.

<p align="center">
  <img src="Fig6_CNN_Confusion_Matrix.png" width="45%">
  <img src="Fig6_Conformer_Confusion_Matrix.png" width="45%">
</p>

### 🔉 Noise Robustness (SNR Test)

To simulate real-world environments, both models were tested under different Signal-to-Noise Ratio (SNR) conditions. Both models degrade under heavy noise, but the **CNN maintains stronger stability**.

<p align="center">
  <img src="Fig5_Accuracy_vs_SNR.png" width="700">
</p>

---

## 💡 Technical Insights

- ✅ CNNs can outperform attention-based models on small datasets
- ✅ Local spectral patterns were sufficient to distinguish the two classes
- ⚠️ Attention mechanisms require larger datasets to generalize effectively
- ⚠️ Noise robustness remains a challenge for both models at low SNR levels

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge)

</div>

- **Python** — core language
- **TensorFlow / Keras** — model building and training
- **Librosa** — audio processing and feature extraction
- **NumPy** — numerical computation
- **Scikit-learn** — evaluation metrics, cross-validation
- **Matplotlib** — visualization

---

## 🚀 Reproducibility

**1. Clone the repository**
```bash
git clone https://github.com/sheyda2021/Audio-Classification-CNN-vs-Conformer.git
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the training notebook or script** to reproduce the results.

---

## 👩‍💻 Author

<div align="center">

**Sheyda Asadi**
Data Science & Machine Learning

[![GitHub](https://img.shields.io/badge/-sheyda2021-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sheyda2021)

</div>

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
