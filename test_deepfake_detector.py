from pathlib import Path

from deepfake_detector import detect_deepfake


audio_file = Path(
    r"C:\Users\dyapa shruthi\Downloads\Scamora_AI\ML\data\audio\real\LA_E_1027220.flac"
)

label, confidence = detect_deepfake(audio_file)

print("=" * 50)
print("SCAMORA AI - DEEPFAKE AUDIO TEST")
print("=" * 50)

print(f"Audio      : {audio_file.name}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")

print("=" * 50)