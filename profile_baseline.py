import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What is the capital of France and why is it famous?",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto"
)
model.eval()

with torch.no_grad():
    inp = tokenizer("Warmup", return_tensors="pt").to("cuda")
    model.generate(**inp, max_new_tokens=5)
torch.cuda.synchronize()
print("Warmup done. Starting profiled run...")

total_tokens = 0
total_time = 0.0
MAX_NEW = 100

with torch.no_grad():
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_tok = out.shape[1] - inputs.input_ids.shape[1]
        total_tokens += n_tok
        total_time += elapsed
        print(f"Prompt done: {n_tok} tokens in {elapsed:.2f}s ({n_tok/elapsed:.1f} tok/s)")

print(f"\n=== BASELINE ===")
print(f"Avg throughput: {total_tokens/total_time:.2f} tokens/sec")
print(f"Avg latency/token: {total_time/total_tokens*1000:.1f} ms")
