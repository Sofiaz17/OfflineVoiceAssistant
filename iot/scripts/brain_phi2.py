from llama_cpp import Llama

# 1. LOAD THE MODEL
print("Loading phi-2...")
llm = Llama(
    model_path="../models/slm_models/phi-2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

def think(prompt):
    print("Thinking inside brain_phi2.py")
    
    # 2. FORMAT PROMPT
    formatted_prompt = f"Instruct: {prompt}\nOutput:"

    # 3. GENERATE RESPONSE
    output = llm(
        formatted_prompt,
        max_tokens=500, 
        stop=["Instruct:", "Output:"],
        echo=False
    )

    return output['choices'][0]['text'].strip()

if __name__ == "__main__":
    print("AI is thinking...")
    response = think("Hello, how are you today?")
    print(f"Answer: {response}")