from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch


# Allow Scamora AI to use the official AASIST repository
AASIST_REPO = Path(
    r"C:\Users\dyapa shruthi\Downloads\aasist-main\aasist-main"
)

sys.path.insert(0, str(AASIST_REPO))

from models.AASIST import Model


MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

MODEL_PATH = AASIST_REPO / "models" / "weights" / "AASIST.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load model once
model = Model(MODEL_CONFIG)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()


def pad_audio(audio, max_len=64600):
    """Pad or crop audio to AASIST input size."""

    if len(audio) >= max_len:
        return audio[:max_len]

    repeats = int(max_len / len(audio)) + 1
    padded = np.tile(audio, repeats)

    return padded[:max_len]


def detect_deepfake(audio_path):
    """
    Detect whether an audio file is real or spoof/deepfake.

    Returns:
        label: REAL or FAKE
        confidence: probability between 0 and 1
    """

    audio, sample_rate = sf.read(str(audio_path))

    # Convert stereo audio to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, got {sample_rate} Hz"
        )

    audio = pad_audio(audio)

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        _, output = model(audio_tensor)

    probabilities = torch.softmax(output, dim=1)

    spoof_probability = probabilities[0][0].item()
    bonafide_probability = probabilities[0][1].item()

    if bonafide_probability >= spoof_probability:
        return "REAL", bonafide_probability

    return "FAKE", spoof_probability