# 🤖 Offline Edge Voice Assistant (Raspberry Pi 5)

A fully offline, privacy-focused voice assistant running entirely on a Raspberry Pi 5. This project demonstrates how to deploy a modular AI pipeline (Wake Word -> ASR -> LLM -> TTS) on embedded hardware without any cloud dependencies.

📖 Overview
This project implements a "Listen-Think-Speak" loop using state-of-the-art quantized models. It is designed to be lightweight, running within the 8GB RAM limit of the Raspberry Pi while maintaining real-time responsiveness (~15 tokens/sec).

🛠️ Hardware Setup
The system is built around the Raspberry Pi 5 (8GB). 

Input&output: USB Audio Adapter with Microphone (3.5mm jack) and speakers. 

🏗️ Software Architecture
The application runs as a synchronous state machine to prevent resource contention .
-   Wait State: Listens for "Hey Jarvis" (Low CPU usage).
-   Record State: VAD (Voice Activity Detection) records command until silence is detected.
-   Transcribe: Whisper converts audio to text.
-   Think: TinyLlama generates a response (token-by-token).
-   Speak: Piper reads the response; input is blocked to prevent self-triggering.

Installation
1. System Requirements
-   Raspberry Pi OS (Bookworm) 64-bit
-   Python 3.12 or 3.13
2. Install Dependencies
```
sudo apt-get update && sudo apt-get install -y libasound2-dev portaudio19-dev
pip install -r requirements.txt 
```

3. Download Models
Place your GGUF models in the models/ directory.
### Example: Download TinyLlama
```
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```
4. Run the Assistant
```
python3 main.py
```

## NOTE
# Recommended installation method for Pi 5
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

📂 Project Structure
├── scripts/main.py              # Main loop & state machine
├── scripts/benchmark.py         # Script to test model performance
├── requirements.txt     # Python dependencies
├── models.json          # Config for benchmark models
├── prompts.json         # Test prompts for benchmark
├── models/              # Directory for GGUF / ONNX models
└── utils/               # Helper scripts (audio, VAD)