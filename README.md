# Scamora AI

## Project Overview
Scamora AI is a real-time, multilingual (Telugu/Hindi/English) mobile application that protects users from social engineering scams and AI-driven voice fraud — including "Digital Arrest" scams — by analyzing live call audio, alerting a trusted guardian, and generating police-ready evidence reports.

## Problem Statement
Scam calls, including sophisticated AI deepfake voice impersonations, are rising rapidly and disproportionately target people unfamiliar with legal/technical processes. Existing call-blocking apps only flag known spam numbers — they cannot detect a new scammer or a cloned voice in real time, leaving victims unprotected during the most critical moments of a call.

## Proposed Solution
Scamora AI activates automatically on calls from unknown numbers, analyzes the caller's speech using both keyword/ML-based scam detection and AI deepfake voice detection, and classifies the call as **Normal, Promotional, Scam, or AI Deepfake Voice**. On a high-risk result, it immediately alerts a designated guardian and generates a downloadable evidence PDF for reporting to authorities.

## Features
- Real-time call classification: Normal / Promotional / Scam / AI Deepfake Voice
- Dual-model analysis: text-based scam detection + AI voice deepfake detection
- Multilingual support (English, Hindi, Telugu)
- Guardian alert system for vulnerable users
- Auto-generated, police-ready evidence PDF reports
- Manual "Analyze Call" mode for testing with recorded audio + transcript

## Technology Stack
- **Mobile App:** Flutter (Android)
- **Backend:** FastAPI (Python)
- **Text Classification:** TF-IDF + Logistic Regression (92.68% accuracy on real call-transcript data)
- **Deepfake Voice Detection:** AASIST model trained on ASVspoof 2019 LA dataset (97% accuracy)
- **Speech Processing:** Whisper / Groq for speech-to-text
- **Database:** PostgreSQL
- **PDF Generation:** ReportLab
- **Notifications:** Firebase Cloud Messaging

## Setup & Usage Instructions

### Backend
```bash
cd Scamora_AI/backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Backend runs at `http://127.0.0.1:8000` (use `http://10.0.2.2:8000` when calling from an Android emulator).

### Mobile App
```bash
cd scam_voice_detection/test_flutter_app
flutter pub get
flutter run
```

### Testing the Analysis Engine
Use the in-app **Analyze Call** screen to upload a sample audio file and enter a transcript. The app will return a risk score, matched keywords, and final classification.

## Team Details
_[Add your team name and member names/roles here]_

## Current Limitations / Roadmap
Real caller-number detection for unknown numbers is functional. Full live audio capture and analysis during an active call is still being connected to the backend, due to Android's restrictions on third-party call control without being the default dialer. Planned fix: notification-based audio capture instead of a custom in-call UI.
