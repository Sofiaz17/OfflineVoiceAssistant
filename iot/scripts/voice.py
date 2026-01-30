import os
import subprocess

def speak(text):
    print("Inside voice.py")

    model_file = "../models/en_US-ryan-medium.onnx" 
    
    clean_text = text.replace('"', '').replace("'", "")
    
    cmd = f'echo "{clean_text}" | piper --model {model_file} --output_file response.wav'
    
    subprocess.run(cmd, shell=True)
    os.system("aplay -D plughw:2,0 response.wav")

if __name__ == "__main__":
    speak("Hello, I am ready to convert your text to speech.")
