import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import brain_phi2 as brain
import brain_tinyllama as brain
import voice
import whisper

# --- CONFIG ---
WAKE_WORD = "hey_jarvis_v0.1"
SCORE_THRESHOLD = 0.5     # Sensitivity (0.0 to 1.0)

# --- INIT MODELS ---
print("Initializing Systems...")

# 1. Load Wake Word Model
print(f"Loading Wake Word Model ({WAKE_WORD})...")
ww_model = Model(wakeword_model_paths=["../models/hey_jarvis_v0.1.onnx"],)

# 2. Audio Config (for OpenWakeWord)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()

print("Systems Ready.")

def start_assistant_cycle():
    """Run the Listen -> Think -> Speak cycle once."""
    
    # 1. Listen
    print("\nListening for command...")
    user_text = whisper.listen()
    
    # 2. Think
    if user_text:
        print(f"User said: {user_text}")
        print("Thinking...")
        bot_reply = brain.think(user_text)
        print(f"Bot: {bot_reply}")
        
        # 3. Speak
        print("Speaking...")
        voice.speak(bot_reply)
    else:
        print("No speech detected.")

# --- MAIN LOOP ---
try:
    # Start the Microphone Stream for Wake Word
    mic_stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

    print(f"\nWaiting for activation ('hey Jarvis!')'...")

    while True:
        # 1. Get Audio Chunk
        audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

        # 2. Feed to Wake Word Model
        prediction = ww_model.predict(audio_data)

        # 3. Check for Wake Word
        if prediction[WAKE_WORD] > SCORE_THRESHOLD:
            print(f"Wake Word Detected!")

            ww_model.reset()
            # --- RELEASE THE MIC ---
            mic_stream.stop_stream()
            mic_stream.close()

            # --- RUN THE ASSISTANT ---
            start_assistant_cycle()

            # --- RECLAIM THE MIC ---
            mic_stream = audio.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)
            
            print(f"\nWaiting for '{WAKE_WORD}'...")

except KeyboardInterrupt:
    print("\nStopping...")
    if mic_stream.is_active():
        mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
    print("Bye!")