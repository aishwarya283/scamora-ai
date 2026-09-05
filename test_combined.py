from pathlib import Path

from ml_engine import analyze_call


audio_file = Path(
    r"C:\Users\dyapa shruthi\Downloads\Scamora_AI\ML\data\audio\real\LA_E_1027220.flac"
)

transcript = (
    "This is an urgent call from the bank. "
    "Your account will be blocked. "
    "Please provide your OTP immediately."
    "Hello, I am calling to confirm your delivery. "
    "Your order will arrive tomorrow. "
    "Thank you."
)

result = analyze_call(
    audio_file,
    transcript
)

print("=" * 50)
print("SCAMORA AI - COMBINED ANALYSIS")
print("=" * 50)

for key, value in result.items():

    if key == "text_risk_score":
        print(f"{key:<20}: {value:.2f}%")

    elif key == "ml_scam_probability":
        print(f"{key:<20}: {value:.2%}")

    elif key == "audio_confidence":
        print(f"{key:<20}: {value:.2%}")

    else:
        print(f"{key:<20}: {value}")

print("=" * 50)