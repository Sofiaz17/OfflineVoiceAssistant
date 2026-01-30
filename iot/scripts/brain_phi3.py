from llama_cpp import Llama
import os

# 1. LOAD THE MODEL
model_file = "../models/slm_models/Phi-3-mini-4k-instruct-q4.gguf" 

print(f"Loading Phi-3 from {model_file}...")

# Initialize Llama
llm = Llama(
    model_path=model_file,
    n_ctx=4096,  
    n_threads=4,
    verbose=False
)

def think(prompt):
    print("Thinking inside brain_phi3.py")
    
    # 2. FORMAT PROMPT
    formatted_prompt = f"<|user|>\n{prompt} <|end|>\n<|assistant|>"

    # 3. GENERATE RESPONSE
    output = llm(
        formatted_prompt,
        max_tokens=500, 
        stop=["<|end|>"],
        echo=False
    )

    return output['choices'][0]['text'].strip()

if __name__ == "__main__":
    print("AI is thinking...")
    response = think("Hello, how are you today?")
    print(f"Answer: {response}")