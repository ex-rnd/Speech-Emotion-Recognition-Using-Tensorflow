#!/usr/bin/env python3
import argparse

import numpy as np
import librosa
import tensorflow as tf

# Must match the same list/order from train_ser.py
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_mfcc(path, n_mfcc=40, duration=3, offset=0.5):
    """
    Load audio file, compute MFCCs, then return the time-averaged feature vector.
    """
    y, sr = librosa.load(path, duration=duration, offset=offset)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).astype('float32')

def main(args):
    # Load the best model from disk
    model = tf.keras.models.load_model(args.model)

    # Extract features from your audio file
    mfcc = extract_mfcc(args.audio)
    x = mfcc.reshape((1, mfcc.shape[0], 1))  # (1, 40, 1)

    # Predict
    probs = model.predict(x)[0]
    idx   = np.argmax(probs)
    emotion = EMOTIONS[idx]
    confidence = probs[idx] * 100

    print(f"Predicted emotion: {emotion} ({confidence:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict emotion from a WAV file")
    parser.add_argument('--model', required=True, help="Path to trained .h5 model")
    parser.add_argument('--audio', required=True, help="Path to input .wav file")
    args = parser.parse_args()
    main(args)




# ###
# python scripts/predict_ser.py --model models/ser.h5 --audio data/audio/OAF_back_fear.wav
# ###