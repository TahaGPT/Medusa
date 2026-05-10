import torch, time, os
from transformers import AutoTokenizer, BitsAndBytesConfig
from medusa.model.medusa_model import MedusaModel

MEDUSA_HEAD_PATH = os.path.expanduser(
    "~/Z4RAmode/PDC/Project/Medusa/results/medusa-heads_medusa_mlp_TinyLlama-1.1B-Chat-v1.0_medusa_3_lr_0.001_layers_1"
)

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

print("Loading Medusa model...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = MedusaModel.from_pretrained(
    MEDUSA_HEAD_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print("Warming up...")
for chunk in model.medusa_generate(
    **tokenizer("Warmup", return_tensors="pt").to("cuda"),
    max_steps=5,
    temperature=0.0,
    posterior_threshold=0.09,
    posterior_alpha=0.3,
):
    pass
torch.cuda.synchronize()
print("Warmup done. Starting profiled run...")

total_tokens = 0
total_time = 0.0
MAX_STEPS = 100

for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    output_text = ""
    for chunk in model.medusa_generate(
        **inputs,
        max_steps=MAX_STEPS,
        temperature=0.0,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
    ):
        output_text = chunk["text"]
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(output_text))
    total_tokens += n_tok
    total_time += elapsed
    print(f"Prompt done: {n_tok} tokens in {elapsed:.2f}s ({n_tok/elapsed:.1f} tok/s)")

print(f"\n=== STANDARD MEDUSA ===")
print(f"Avg throughput: {total_tokens/total_time:.2f} tokens/sec")
print(f"Avg latency/token: {total_time/total_tokens*1000:.1f} ms")
