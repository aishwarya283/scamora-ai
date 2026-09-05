from pathlib import Path

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_NAME = "SpeechAntiSpoofingBenchmarks/AASIST"

REAL_DIR = Path("data/audio/real")
FAKE_DIR = Path("data/audio/fake")


print("Loading model...")

extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

print("Model loaded successfully!\n")


def predict_audio(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    inputs = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(probabilities, dim=-1).item()

    label = model.config.id2label[predicted_id]
    confidence = probabilities[0][predicted_id].item()

    return label, confidence


# Test one real file
real_file = next(REAL_DIR.glob("*.flac"))

print("Testing REAL audio:")
print(real_file.name)

label, confidence = predict_audio(real_file)

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")


# Test one fake file
fake_file = next(FAKE_DIR.glob("*.flac"))

print("\nTesting FAKE audio:")
print(fake_file.name)

label, confidence = predict_audio(fake_file)

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")