from llama_cpp import Llama
import os

# 1. LOAD THE MODEL
model_file = "../models/slm_models/stablelm-1.6b-code_instructions_alpaca_style.Q4_K_M.gguf"

print(f"Loading StableLM...")

# Initialize Llama
llm = Llama(
    model_path=model_file,
    n_ctx=4096,  
    n_threads=4,
    verbose=False
)

def think(prompt):
    print("Thinking inside brain_stablelm.py")

    # 2. FORMAT PROMPT
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"

    # 3. GENERATE RESPONSE
    output = llm(
        formatted_prompt,
        max_tokens=500,
        stop=["### Instruction:", "### Response:"], 
        echo=False
    )

    return output['choices'][0]['text'].strip()

if __name__ == "__main__":
    print("AI is thinking...")
    response = think("Hello, how are you today?")
    print(f"Answer:\n{response}")