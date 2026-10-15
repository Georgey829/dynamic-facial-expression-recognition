# Dynamic Facial Expression Recognition
### From Static Classification to Temporal Emotion Analysis

An undergraduate research project investigating facial expression
recognition using deep learning, with a particular focus on
temporal dynamics, prediction stability, and recognition timing.

## 1. Overview

Facial Expression Recognition (FER) is an important research area
in computer vision and human-computer interaction.

Traditional FER methods primarily focus on classifying facial
expressions from individual images. However, facial expressions
are inherently dynamic processes that evolve over time.

This research investigates the transition from static facial
expression classification to dynamic facial expression recognition.

In addition to classification performance, the project explores
how model predictions evolve throughout facial expression sequences.

The research consists of three main components:

1. Static Facial Expression Recognition using ResNet18
2. Experiment A: Dynamic FER using CNN-LSTM without pretraining
3. Experiment B: Dynamic FER using pretrained visual features

A further objective is to analyze temporal prediction behavior,
including recognition confidence, prediction stability,
Trigger Time, and Stable Recognition Time.

---

## 2. Research Objectives

The primary objectives of this research are:

- Establish a baseline for static facial expression recognition.
- Investigate dynamic facial expression recognition using CNN-LSTM.
- Examine the influence of pretrained visual representations.
- Analyze changes in prediction confidence over time.
- Investigate when facial expressions can be reliably recognized.
- Analyze temporal prediction behavior using Trigger Time and Stable Recognition Time.

Rather than focusing exclusively on final classification accuracy,
this research aims to provide a more detailed understanding of
the temporal recognition process.

---

## 3. Research Framework

The project consists of three experimental components.

### 3.1 Static Facial Expression Recognition

**Model:** ResNet18

The static FER component investigates facial expression
classification using individual facial images.

A ResNet18-based convolutional neural network is employed
to extract visual features and classify facial expressions.

This experiment serves as a reference point for understanding
conventional image-based FER.

### 3.2 Experiment A: CNN-LSTM without Pretraining

**Model:** CNN-LSTM

Experiment A investigates dynamic facial expression recognition
using sequential facial images.

A CNN extracts spatial features from individual frames,
while an LSTM models temporal dependencies across the sequence.

The model is trained without pretrained visual representations.

This experiment establishes a baseline for evaluating
temporal facial expression recognition.

### 3.3 Experiment B: CNN-LSTM with Pretrained Features

**Model:** Pretrained CNN + LSTM

Experiment B investigates whether pretrained visual features
improve dynamic facial expression recognition.

Pretrained visual representations are incorporated into
the CNN-LSTM framework.

The experiment examines their influence on classification
performance and temporal prediction behavior.

---

## 4. Temporal Recognition Analysis

An important component of this research is the investigation
of how facial expression predictions evolve over time.

Traditional FER evaluation often focuses on the final
classification result.

However, a correct prediction at the final frame does not
necessarily indicate that the model recognized the expression
reliably throughout the sequence.

This research therefore investigates temporal prediction behavior.

### 4.1 Prediction Confidence

The evolution of class probabilities is analyzed across
facial expression sequences.

This allows the observation of how model confidence changes
as facial expressions develop.

### 4.2 Prediction Stability

Prediction stability is examined to distinguish between
temporary predictions and sustained recognition.

The analysis considers how predicted emotion labels change
throughout a sequence.

### 4.3 Trigger Time and Stable Recognition Time

Trigger Time is the time point at which the predicted probability
of the target emotion (the ground-truth class) shows the largest
increase between consecutive frames. It describes the strongest
increase in confidence in the target emotion and does not necessarily
indicate correct or stable recognition.

Stable Recognition Time is the earliest frame from which the target
emotion remains the highest-probability class through the end of the
sequence. If no such frame exists, Stable Recognition Time is recorded as NaN.

Together, these measures distinguish the timing of the largest probability
increase from the onset of sustained correct recognition.

---

## 5. Datasets

The research uses facial expression datasets for static
and dynamic recognition experiments.

### FER2013

FER2013 contains grayscale facial expression images
commonly used for static facial expression classification.

### Extended Cohn-Kanade Dataset (CK+)

CK+ contains facial expression sequences that capture
the development of facial expressions over time.

The dataset is used to investigate temporal recognition
behavior in dynamic FER experiments.

**Note:** Original datasets are not included in this repository.
Users must obtain the datasets separately and comply with
their respective licensing and usage conditions.

---

## 6. Repository Structure

```text
dynamic-facial-expression-recognition/
│
├── README.md
├── .gitignore
│
├── static_fer/
│   ├── train_resnet18.py
│   └── evaluate_resnet18.py
│
├── experiment_A/
│   ├── 01_prepare_ckplus.py
│   ├── 02_train_full_sequence.py
│   └── 03_evaluate_full_sequence.py
│
└── experiment_B/
    ├── 04_train_pretrained.py
    ├── 05_evaluate_pretrained.py
    └── 06_temporal_certainty_analysis.py
```

---

## 7. Experimental Evaluation

The experimental framework considers both conventional
classification performance and temporal recognition behavior.

Classification evaluation includes:

- Classification Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Temporal analysis includes:

- Frame-level prediction probabilities
- Temporal confidence evolution
- Prediction stability
- Trigger Time
- Stable Recognition Time

These analyses provide complementary perspectives on
the performance of dynamic facial expression recognition.

---

## 8. Implementation

The research is implemented primarily in Python.

Main libraries include:

- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- Pillow
- tqdm

The implementation includes data preprocessing,
model training, evaluation, and temporal prediction analysis.

---

## 9. Reproducibility

The repository provides research code for the three
experimental components.

The original datasets and trained model weights are
not distributed in this repository. The Experiment A
and Experiment B scripts currently use~/Desktop/
FER_project as their working directory. Before
running these scripts, check and update the dataset,
checkpoint, and output paths to match your local setup.

Detailed environment configuration, dataset preparation,
and execution instructions will be added as the repository
documentation is finalized.

After obtaining CK+ and configuring the local paths:

    1. Run experiment_A/01_prepare_ckplus.py to prepare the sequence
    splits shared by Experiments A and B.

    2. For Experiment A, run experiment_A/02_train_full_sequence.py,
    followed by experiment_A/03_evaluate_full_sequence.py.

    3. For Experiment B, run experiment_B/04_train_pretrained.py,
    followed by experiment_B/05_evaluate_pretrained.py.

    4. Run experiment_B/06_temporal_certainty_analysis.py to analyze
    the frame-level probabilities generated by Experiment B evaluation.

Experiments A and B share the prepared data splits, but each uses its
own training and evaluation scripts.

---

## 10. Research Context

This project was developed as an undergraduate graduation
research project in the Department of Information and
Communication Sciences, Faculty of Science and Technology,
Sophia University, Tokyo, Japan.

The research explores deep learning approaches to facial
expression recognition, with particular attention to
temporal modeling and recognition stability.

---

## 11. Author

Undergraduate Research Project

Sophia University  
Faculty of Science and Technology  
Department of Information and Communication Sciences

---

## 12. License

No open-source license has been assigned to this repository.

All rights reserved unless otherwise specified.