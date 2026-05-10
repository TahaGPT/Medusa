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
    bnb_4bit_use_double_quant=True,  # OPTIMIZATION: double quant reduces memory further
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

# ── OPTIMIZATION 1: Explicit 3-head medusa_choices tuned for TinyLlama (Gap 5) ──
# Standard Medusa uses default 5-head tree — wrong for our 3-head model
# This tree is sized for exactly 3 heads, reducing wasted candidate evaluation
medusa_choices_3head = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0],
    [0, 2], [3], [0, 3], [4], [0, 0, 1], [0, 1, 0],
    [0, 4], [1, 1], [0, 0, 0, 0]
]

# ── OPTIMIZATION 2: Adaptive posterior threshold (Gap 5) ──
# Instead of fixed 0.09, adapt based on recent acceptance rate
class AdaptiveThreshold:
    def __init__(self, base=0.09, window=5):
        self.base = base
        self.history = []
        self.window = window

    def update(self, tokens_generated, max_steps):
        acceptance_rate = tokens_generated / max(max_steps, 1)
        self.history.append(acceptance_rate)
        if len(self.history) > self.window:
            self.history.pop(0)

    def get(self):
        if not self.history:
            return self.base
        avg = sum(self.history) / len(self.history)
        # Lower threshold = more aggressive (accept more candidates)
        # Higher threshold = more selective (accept fewer but safer candidates)
        return max(0.05, min(0.3, avg * 0.8))

adaptive = AdaptiveThreshold()

# ── OPTIMIZATION 3: inference_mode + explicit cache clearing (Gap 3) ──
# Clear KV cache between prompts to avoid stale state overhead
if hasattr(model, 'past_key_values'):
    del model.past_key_values

print("Warming up...")
with torch.inference_mode():
    for chunk in model.medusa_generate(
        **tokenizer("Warmup", return_tensors="pt").to("cuda"),
        max_steps=5,
        temperature=0.0,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        medusa_choices=medusa_choices_3head,
    ):
        pass
torch.cuda.synchronize()

# Clear KV cache after warmup
if hasattr(model, 'past_key_values'):
    del model.past_key_values

print("Warmup done. Starting profiled run...")

total_tokens = 0
total_time = 0.0
MAX_STEPS = 100

for prompt in PROMPTS:
    # Clear KV cache between prompts
    if hasattr(model, 'past_key_values'):
        del model.past_key_values

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    threshold = adaptive.get()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output_text = ""
    with torch.inference_mode():
        for chunk in model.medusa_generate(
            **inputs,
            max_steps=MAX_STEPS,
            temperature=0.0,
            posterior_threshold=threshold,
            posterior_alpha=0.3,
            medusa_choices=medusa_choices_3head,
        ):
            output_text = chunk["text"]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(output_text))
    total_tokens += n_tok
    total_time += elapsed
    adaptive.update(n_tok, MAX_STEPS)
    print(f"Prompt done: {n_tok} tokens in {elapsed:.2f}s ({n_tok/elapsed:.1f} tok/s) [threshold={threshold:.3f}]")

print(f"\n=== OPTIMIZED MEDUSA ===")
print(f"Avg throughput: {total_tokens/total_time:.2f} tokens/sec")
print(f"Avg latency/token: {total_time/total_tokens*1000:.1f} ms")
