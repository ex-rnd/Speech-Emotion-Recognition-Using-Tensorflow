# Speech-Emotion-Recognition-Using-Tensorflow
A jupyter notebook for processing emotions from speech

🔆 Speech Emotion Recognition
A deep‐learning project that classifies speech into one of seven emotions using an LSTM‐based model.

✳️ Visual Overview 


https://github.com/user-attachments/assets/75b68583-e15e-4e30-9a8f-c42a8230bc3a



📹 Demo
First, see it in action:
- 1. Open a terminal.
- 2. Run the prediction script on a WAV file
```
python scripts/predict_ser.py -f data/audio/sad_01.wav
```
- 3. Observe live confidence scores and predicted labels
(Replace the above with a short screen recording or animated GIF once you upload to GitHub.)

🔍 Project Overview

- 1. Problem
Automatically recognize emotion (angry, disgust, fear, happy, neutral, sad, surprise) from 3–4 s speech clips.

- 2. Key Components
- Feature extraction: 40-dim MFCCs
- Model: Single‐layer LSTM → Dropout → Dense → Dropout → Dense → Dropout → Dense(softmax)
- Training: 80/20 train/validation split, ModelCheckpoint callback
- Evaluation: Accuracy, loss curves

🛠️ Getting Started
1. Clone
```
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Environment
```
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

3. Data
- Place your .wav files in " data/audio/ " with filenames like " speakerID_emotion.wav ".
- Emotion labels must match EMOTIONS = ['angry','disgust','fear','happy','neutral','ps','sad'].

▶️ Usage
Run inference on any WAV:
```
python scripts/predict_ser.py \
  -f data/audio/happy_10.wav \
  -m models/best_ser.h5
```
- -f/--file: path to input WAV
- -m/--model: path to saved .h5 model (defaults to models/best_ser.h5)

📓 Interactive Notebook
All EDA, feature engineering, model building, training and plotting live in the Jupyter notebook:
1. Dataset samples
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/3fc307d0-a9d1-48c9-82f2-10bc3f416bf7" />

📐 Model Architecture
```
Input (40 × 1 MFCC feature vector)
    │
    ▼
  LSTM (256 units)
    │
    ▼
  Dropout (0.2)
    │
    ▼
  Dense (128, ReLU)
    │
    ▼
  Dropout (0.2)
    │
    ▼
  Dense (64, ReLU)
    │
    ▼
  Dropout (0.2)
    │
    ▼
  Dense (7, Softmax)
    │
    ▼
  Output: 7 emotion probabilities
```

💾 Data & Preprocessing
- Duration: Load or pad/truncate to 3 s (or 4 s)
- Sampling: librosa.load(sr=None)
- Feature: librosa.feature.mfcc(n_mfcc=40) → mean over time → shape (40,)
- Labels: extracted from filename suffix (e.g. happy, sad

📊 Training & Results
- Checkpoint: ModelCheckpoint('../models/best_ser.h5', monitor='val_accuracy')
- Curves:
Train/Val Accuracy
Train/Val Loss
- Final performance
- Train accuracy: ~99%
- Validation accuracy: ~99%

🤝 Contributing
- Fork the repo
- Branch naming: feature/xyz-description or fix/xyz-description
- Install pre-commit hooks:
```
pip install pre-commit
pre-commit install
```

- Code style: Follow PEP 8, lint with flake8
- Tests: Add unit tests under tests/ directory
- Pull request:
- Describe your changes
- Link any related issues
- Ensure CI passes
Thank you for your contributions! 🎉













































































































