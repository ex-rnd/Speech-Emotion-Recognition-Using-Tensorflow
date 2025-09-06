###

#!/usr/bin/env python3
import os
import argparse

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# List of target emotions (must match your data filenames)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_mfcc(path, n_mfcc=40, duration=3, offset=0.5):
    """
    Load audio file, compute MFCCs, then return the time-averaged feature vector.
    """
    y, sr = librosa.load(path, duration=duration, offset=offset)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def load_data(data_dir):
    """
    Scan data_dir for .wav files, extract MFCCs, map labels to one-hot floats.
    Returns:
      X: np.ndarray, shape (n_samples, 40, 1), dtype float32
      y: np.ndarray, shape (n_samples, n_emotions), dtype float32
    """
    files = []
    labels = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.wav'):
            continue
        files.append(os.path.join(data_dir, fname))
        # Expect filenames like: anything_emotion.wav
        lbl = fname.rsplit('_', 1)[-1].replace('.wav', '').lower()
        if lbl not in EMOTIONS:
            raise ValueError(f"Unknown emotion label '{lbl}' in file '{fname}'")
        labels.append(lbl)

    # Extract MFCCs
    mfccs = [extract_mfcc(fp) for fp in files]

    # Build X: (n_samples, 40) → (n_samples, 40, 1)
    X = np.stack(mfccs).astype('float32')
    X = np.expand_dims(X, axis=-1)

    # Map labels to integer IDs
    label_to_index = {emo: idx for idx, emo in enumerate(EMOTIONS)}
    ids = np.array([label_to_index[lbl] for lbl in labels], dtype=np.int32)

    # One-hot encode to float32
    y = tf.keras.utils.to_categorical(ids, num_classes=len(EMOTIONS))
    y = y.astype('float32')

    # Sanity check
    print(f"Loaded {len(X)} samples.")
    print(f"X.shape={X.shape}, X.dtype={X.dtype}")
    print(f"y.shape={y.shape}, y.dtype={y.dtype}")

    return X, y

def build_model(input_shape, n_classes):
    """
    Sequential LSTM-based classifier.
    """
    model = Sequential([
        Input(shape=input_shape, name='mfcc_input'),
        LSTM(256),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main(args):
    X, y = load_data(args.data_dir)

    model = build_model(input_shape=(40, 1), n_classes=y.shape[1])
    model.summary()

    ckpt = ModelCheckpoint(
        filepath=args.output,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt]
    )
    print(f"\nTraining complete. Best model saved at: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition model")
    parser.add_argument('--data_dir',   required=True, help="Path to folder of .wav files")
    parser.add_argument('--output',     default='models/ser.h5', help="Where to save best model")
    parser.add_argument('--epochs',     type=int, default=50,        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64,        help="Batch size")
    args = parser.parse_args()
    main(args)


# ###
# cd speech-emotion-recognition
# pip install -r requirements.txt
# python scripts/train_ser.py --data_dir data/audio --output models/ser.h5
# ###
