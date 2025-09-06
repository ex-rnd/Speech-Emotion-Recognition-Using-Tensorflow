# production/emotion_recognizer.py

import os
import shutil
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from pydub.utils import which

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


class EmotionRecognizer:
    """
    Loads a pre-trained speech-emotion-recognition model, predicts emotion for
    incoming WAV files, and converts them to MP3 (or copies WAV if FFmpeg missing).
    """

    def __init__(
        self,
        model_path: str,
        incoming_dir: str = 'incoming/audio',
        outgoing_dir: str = 'outgoing/audio'
    ):
        # Load your Keras model once
        self.model = tf.keras.models.load_model(model_path)

        # Warm up the model (optional but reduces latency on first call)
        dummy = np.zeros((1, 40, 1), dtype=np.float32)
        _ = self.model.predict(dummy)

        # Ensure folders exist
        self.incoming_dir = Path(incoming_dir)
        self.outgoing_dir = Path(outgoing_dir)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.outgoing_dir.mkdir(parents=True, exist_ok=True)

        # Configure pydub’s FFmpeg path if available
        AudioSegment.converter = which('ffmpeg') or which('ffmpeg.exe')
        if AudioSegment.converter:
            self.ffmpeg_available = True
        else:
            self.ffmpeg_available = False
            print(
                "Warning: ffmpeg not found on PATH. "
                "MP3 conversion will be skipped, WAV will be copied instead."
            )

    def extract_mfcc(
        self,
        file_path: Path,
        n_mfcc: int = 40,
        duration: float = 3.0,
        offset: float = 0.5
    ) -> np.ndarray:
        """
        Load a slice of the audio file and compute the time-averaged MFCC vector.
        Returns a (40,) float32 array.
        """
        y, sr = librosa.load(str(file_path), duration=duration, offset=offset)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0).astype('float32')

    @tf.function
    def _predict(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Wrap model call in tf.function for performance.
        """
        return self.model(inputs, training=False)

    def predict_emotion(self, file_path: Path) -> (str, float):
        """
        Extract features from file_path and return (label, confidence).
        """
        mfcc = self.extract_mfcc(file_path)
        x = mfcc.reshape((1, mfcc.shape[0], 1))    # (1, 40, 1)
        probs = self._predict(tf.constant(x)).numpy()[0]
        idx = int(np.argmax(probs))
        return EMOTIONS[idx], float(probs[idx])

    def convert_to_mp3(self, file_path: Path) -> Path:
        """
        If FFmpeg is available, export as MP3; otherwise copy the WAV as-is.
        Returns the path to the new file in outgoing_dir.
        """
        if self.ffmpeg_available:
            audio = AudioSegment.from_file(str(file_path))
            out_name = file_path.stem + '.mp3'
            out_path = self.outgoing_dir / out_name
            audio.export(str(out_path), format='mp3')
        else:
            # Fallback: just copy the WAV
            out_path = self.outgoing_dir / file_path.name
            shutil.copy2(str(file_path), str(out_path))
        return out_path

    def process_file(self, file_arg: str):
        """
        1. Normalize the filename argument
        2. Predict emotion
        3. Print result
        4. Convert/copy to outgoing folder
        """
        fname = Path(file_arg).name
        input_path = self.incoming_dir / fname

        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found.")

        # 1) Emotion prediction
        label, confidence = self.predict_emotion(input_path)
        print(f"Predicted emotion: {label} ({confidence * 100:.2f}%)")

        # 2) Conversion or copy
        out_path = self.convert_to_mp3(input_path)
        print(f"Output → {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition + WAV→MP3 (or copy) converter"
    )
    parser.add_argument(
        '--model',
        required=True,
        help="Path to trained .h5 model file"
    )
    parser.add_argument(
        '--file',
        required=True,
        help="WAV filename in incoming/audio to process (e.g. wontbelongtest.wav)"
    )
    args = parser.parse_args()

    recognizer = EmotionRecognizer(model_path=args.model)
    recognizer.process_file(args.file)