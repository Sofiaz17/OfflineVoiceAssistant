import time
import json
import csv
import psutil
import os
import gc
from llama_cpp import Llama
from datetime import datetime

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def mem_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 ** 2)

def format_prompt(model_format, system, user):
    """
    Formats the input string based on the specific template required by the model.
    """

    if model_format == "tinyllama":
        return f"<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>"

    elif model_format == "phi2":
        return f"Instruct: {user}\nOutput:"

    elif model_format == "phi3":
        return f"<|user|>\n{user} <|end|>\n<|assistant|>"
    
    elif model_format == "stablelm":
        return f"### Instruction:\n{user}\n\n### Response:"

    # Fallback
    return f"{system}\n\n{user}"

def get_stop_tokens(model_format):
    """
    Returns the specific stop tokens for each model architecture.
    """
    if model_format == "tinyllama":
        return ["</s>"]
    elif model_format == "phi2":
        return ["Instruct:", "Output:"]
    elif model_format == "phi3":
        return ["<|end|>"]
    elif model_format == "stablelm":
        return ["### Instruction:", "### Response:"]
    return []

# --- LOAD CONFIGS ---
try:
    with open("../prompts.json") as f:
        PROMPTS = json.load(f)["prompts"]
    with open("../models.json") as f:
        MODELS = json.load(f)["models"]
except FileNotFoundError:
    print("Error: models.json or prompts.json not found.")
    exit()

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = f"{RESULTS_DIR}/benchmark_{timestamp}.csv"

print(f"Starting Benchmark... Output: {csv_path}")

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "model", "prompt", "load_time_s", "ram_idle_mb", 
        "ram_peak_mb", "latency_s", "tokens_gen", "tps", "output"
    ])

    for model_cfg in MODELS:
        print(f"\n------------------------------------------")
        print(f"=== Loading {model_cfg['name']} ===")
        
        # 1. Clean previous memory before loading new model
        gc.collect() 
        ram_start = mem_mb()
        
        load_start = time.perf_counter()
        try:
            llm = Llama(
                model_path=model_cfg["path"],
                n_ctx=model_cfg["ctx"],
                n_threads=model_cfg["threads"],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load {model_cfg['name']}: {e}")
            continue
            
        load_time = time.perf_counter() - load_start
        ram_idle = mem_mb()
        print(f"Loaded in {load_time:.2f}s | RAM overhead: {ram_idle - ram_start:.1f} MB")

        # 2. Warm-up
        print("Warm-up run...")
        llm("Hello", max_tokens=1)

        for i, prompt in enumerate(PROMPTS):
            system = "You are a helpful assistant. Answer very concisely."
            formatted = format_prompt(model_cfg["format"], system, prompt)
            stops = get_stop_tokens(model_cfg["format"])

            ram_before = mem_mb()
            start = time.perf_counter()

            # Run Inference
            out = llm(
                formatted,
                max_tokens=256,
                stop=stops,
                echo=False
            )

            end = time.perf_counter()
            ram_after = mem_mb()

            # 3. METRICS
            text = out["choices"][0]["text"].strip()
            
            tokens = out["usage"]["completion_tokens"] 
            
            latency = end - start
            tps = tokens / latency if latency > 0 else 0

            writer.writerow([
                model_cfg["name"],
                f"Prompt {i+1}",
                f"{load_time:.2f}",
                f"{ram_idle:.1f}",
                f"{max(ram_before, ram_after):.1f}",
                f"{latency:.2f}",
                tokens,
                f"{tps:.2f}",
                text
            ])

            print(f"> Prompt {i+1}: {latency:.2f}s | {tps:.2f} t/s | {tokens} tokens")

        # 4. Explicit cleanup to free RAM for next model
        del llm
        gc.collect()

print(f"\nBenchmark complete.")