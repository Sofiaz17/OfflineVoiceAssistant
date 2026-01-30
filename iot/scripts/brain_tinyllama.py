from llama_cpp import Llama

# 1. Initialize the Model
print("Loading tinyllama...")
llm = Llama(
    model_path="../models/slm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    #model_path="../models/slm_models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

def think(prompt):

    print("Thinking inside brain_tinyllama.py")

    system_message = "You are a helpful AI assistant. Answer strictly with no more than 250 words."
    
    input_text = f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

    # 4. Generate Response
    output = llm(
        input_text,
        max_tokens=500,
        stop=["</s>"],
        echo=False
    )

    return output['choices'][0]['text'].strip()


if __name__ == "__main__":
    print("AI is thinking...")
    response = think("Hello, how are you today?")
    print(f"Answer: {response}")