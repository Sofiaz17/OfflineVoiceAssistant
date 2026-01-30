from faster_whisper import WhisperModel
import pyaudio
import numpy as np

model_size = "tiny.en"
print("Loading Whisper model...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def listen():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening... (Speak now)")
    

    frames = []
    # Record for fixed 5 seconds
    for i in range(0, int(RATE / CHUNK * 5)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert raw audio to numpy array
    audio_data = np.frombuffer(b''.join(frames), np.int16).flatten().astype(np.float32) / 32768.0

    segments, _ = model.transcribe(audio_data, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    print(f"You said: {text}")
    return text

if __name__ == "__main__":
    listen()