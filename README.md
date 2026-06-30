Audio Classification: CNN vs. Conformer
This repository presents a comparative study of two deep learning architectures for short audio intent classification.

The task is to classify short speech segments into two categories: “identity” and “request”.

The project explores how different model architectures behave when trained on relatively small datasets and evaluated under noisy conditions.

The comparison focuses on:

Convolutional Neural Networks (CNN)
Conformer architecture (Convolution + Self‑Attention)
The goal is not only to compare accuracy, but also to analyze robustness, data efficiency, and architectural suitability for small‑scale audio classification tasks.

Project Overview
Audio classification models often behave differently depending on dataset size and noise conditions.

While transformer‑based models such as Conformer perform well in large‑scale speech recognition, they may not always outperform simpler architectures when training data is limited.

This project investigates that scenario through a controlled experiment using Mel‑spectrogram features and two model architectures.

Dataset
The dataset contains short audio recordings labeled as:

identity – speaker identification utterances
request – request‑type utterances
Each audio sample is approximately 1 second long and processed into Mel‑spectrogram representations for model input.

Audio Processing Pipeline
Before model training, raw audio signals are transformed into Mel‑spectrogram features.

1. Waveform Normalization
The raw waveform is normalized to stabilize amplitude variations across samples.

<p align=“center”>

<img src=“Fig1_Normalized_Waveform.png” width=“800”>

</p>

2. Spectrogram Feature Extraction
The audio signal is converted from the time domain to the frequency domain using Mel‑Spectrograms.

<p align=“center”>

<img src=“Fig2_Waveform_Spectrogram.png” width=“900”>

</p>

3. Class Spectrogram Comparison
Below is a visual comparison of spectrogram patterns between the two classes.

<p align=“center”>

<img src=“Fig4_MelSpec_Comparison.png” width=“850”>

</p>

These visual patterns help explain why convolutional filters can capture discriminative features effectively.

Model Architectures
CNN Baseline
A convolutional neural network was used as the baseline model.

The CNN processes Mel‑spectrogram images and learns local spectral patterns associated with each class.

Advantages:

Efficient training
Strong local feature extraction
Good performance with smaller datasets
Conformer
The Conformer architecture combines:

Convolution layers (local feature extraction)
Self‑attention layers (global context modeling)
Conformers are widely used in large‑scale speech recognition tasks, but their effectiveness depends heavily on dataset size.

Experimental Results
K‑Fold Accuracy
Cross‑validation results show that the CNN model consistently outperformed the Conformer on this dataset.

<p align=“center”>

<img src=“Fig3_Classification_Results.png” width=“700”>

</p>

The CNN maintained accuracy above 98% across folds.

Confusion Matrix Comparison
<p align=“center”>

<img src=“Fig6_CNN_Confusion_Matrix.png” width=“45%”>

<img src=“Fig6_Conformer_Confusion_Matrix.png” width=“45%”>

</p>

The CNN achieved near‑perfect classification, while the Conformer showed higher confusion between the two classes.

Noise Robustness (SNR Test)
To simulate real‑world environments, both models were tested under different Signal‑to‑Noise Ratio (SNR) conditions.

<p align=“center”>

<img src=“Fig5_Accuracy_vs_SNR.png” width=“700”>

</p>

Both models degrade under heavy noise, but the CNN maintains stronger stability.

Technical Insights
Key observations from the experiments:

CNNs can outperform attention‑based models on small datasets
Local spectral patterns were sufficient to distinguish the two classes
Attention mechanisms require larger datasets to generalize effectively
Noise robustness remains a challenge for both models at low SNR levels
Tech Stack
Python
TensorFlow / Keras
Librosa
NumPy
Scikit‑learn
Matplotlib
Reproducibility
Clone the repository:

text
git clone https://github.com/sheyda2021/Audio-Classification-CNN-vs-Conformer.git
Install dependencies:

text
pip install -r requirements.txt
Run the training notebook or script to reproduce the results.

Author
Sheyda Asadi

Data Science & Machine Learning

GitHub:

https://github.com/sheyda2021
