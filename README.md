# Medusa + TinyLlama — Setup & Profiling Guide

> WSL2 Ubuntu 24.04 · RTX 4050 · Python 3.8.10 · May 2026  
> Living document — update as new findings are added.

---

## Table of Contents

1. [System Specifications](#1-system-specifications)
2. [Installation Steps](#2-installation-steps)
3. [Environment Status](#3-environment-status)
4. [Baseline Profile Results](#4-baseline-profile-results)
5. [Next Steps](#5-next-steps)
6. [Known Issues & Fixes](#6-known-issues--fixes)

---

## 1. System Specifications

| Component | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble) on WSL2 |
| Windows Host | Windows 11 + WSL2 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6,141 MB (~6 GB) |
| NVIDIA Driver | 591.91 (Windows) / 590.60 (WSL2) |
| CUDA Version | 13.1 (driver) / 12.4 (PyTorch wheels) |
| Python | 3.8.10 (via pyenv) |
| PyTorch | 2.4.1+cu124 |
| Transformers | 4.44.2 (pinned — version sensitive) |
| Nsight Systems | 2025.6.3 |

---

## 2. Installation Steps

> **Why Python 3.8.10?** Medusa's dependency chain (transformers + torch version triangle) only works cleanly on 3.8.10. System Python 3.12 will fail with version conflicts. pyenv lets both coexist — your system Python is untouched.

### 2.1 Install pyenv

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git
```

Install pyenv:

```bash
curl https://pyenv.run | bash
```

Add to `~/.bashrc`:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

Install Python 3.8.10 (takes ~10 min — compiling from source, don't panic if it looks frozen):

```bash
pyenv install 3.8.10
```

### 2.2 Create virtual environment

```bash
~/.pyenv/versions/3.8.10/bin/python3 -m venv ~/medusa-env
source ~/medusa-env/bin/activate
python3 --version   # must print: Python 3.8.10
```

Always activate before doing anything:

```bash
source ~/medusa-env/bin/activate
```

### 2.3 Install PyTorch

> `torch==2.3.1` does not exist for cu124. Use `2.4.1` — everything downstream works fine with it.

```bash
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 2.4 Install Medusa

Clone your team's fork (not the original repo — the fork has fixes already applied):

```bash
cd ~/Z4RAmode/PDC/Project
git clone <your-fork-url> Medusa
cd Medusa
```

Fix build tooling first, then install:

```bash
pip3 install wheel setuptools
pip3 install --no-build-isolation -e ".[train]"
```

Install remaining dependencies:

```bash
pip3 install "transformers==4.44.2" accelerate bitsandbytes
pip3 install fschat sentencepiece protobuf datasets
```

### 2.5 Patch stale Flash Attention imports

Medusa was written for `transformers <=4.34` which used `is_flash_attn_available`. This name was changed to `is_flash_attn_2_available` in transformers 4.35. Medusa's repo was never updated. Patch all affected files at once:

```bash
grep -rl "is_flash_attn_available" ~/Z4RAmode/PDC/Project/Medusa/medusa/model/ \
  | xargs sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g'
```

Confirm zero old occurrences remain:

```bash
grep -r "is_flash_attn_available" ~/Z4RAmode/PDC/Project/Medusa/medusa/model/ \
  | grep -v "is_flash_attn_2_available" \
  && echo "STILL BROKEN" || echo "ALL CLEAN"
```

### 2.6 Verify full installation

```bash
python3 --version
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "from medusa.model.medusa_model import MedusaModel; print('Medusa: OK')"
```

Expected output:

```
Python 3.8.10
2.4.1+cu124
True
Medusa: OK
```

### 2.7 Install Nsight Systems

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install -y nsight-systems-2025.6.3
nsys --version   # NVIDIA Nsight Systems version 2025.6.3
```

---

## 3. Environment Status

| Status | Step | Notes |
|---|---|---|
| ✅ DONE | pyenv installed + Python 3.8.10 compiled | |
| ✅ DONE | medusa-env virtual environment created | |
| ✅ DONE | PyTorch 2.4.1+cu124 installed | CUDA confirmed working |
| ✅ DONE | Medusa repo cloned + installed (editable) | |
| ✅ DONE | Flash attention import patch applied | ALL CLEAN |
| ✅ DONE | transformers 4.44.2 pinned | Version sensitive — do not upgrade |
| ✅ DONE | Nsight Systems 2025.6.3 installed | |
| ✅ DONE | Baseline profile generated | See Section 4 |
| ⏳ PENDING | Train Medusa heads | Next step |
| ⏳ PENDING | Post-Medusa Nsight profile | For comparison |
| ⏳ PENDING | Results comparison + graphs | Final analysis |

---

## 4. Baseline Profile Results

Profile collected using Nsight Systems on `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with 4-bit NF4 quantization. Three prompts, 100 max new tokens each.

**Profile files:**
```
~/Z4RAmode/PDC/Project/Medusa/results/baseline_profile.nsys-rep
~/Z4RAmode/PDC/Project/Medusa/results/baseline_profile.sqlite
```

### 4.1 Summary metrics

| Metric | Value | Notes |
|---|---|---|
| Total kernel launches | 80,219 | For 3 prompts (~76 tokens total) |
| GPU compute time | 0.695 s | Pure kernel execution |
| Memory transfer time | 0.454 s | 994 memcpy ops |
| Data transferred | 2,210.5 MB | Host ↔ device |
| Dominant kernel share | 51.8% | 4-bit GEMM (see below) |

### 4.2 Top CUDA kernels

| Kernel | GPU % | Calls | Avg ms | Role |
|---|---|---|---|---|
| `kgemm_4bit_inference_naive` | **51.8%** | 11,858 | 0.029 | Core 4-bit GEMM — dominant bottleneck |
| `kDequantizeBlockwise` | 10.8% | 616 | 0.114 | Weight dequantization |
| `gemvx::kernel` | 8.3% | 77 | 0.703 | Matrix-vector multiply |
| `kQuantizeBlockwise` | 7.5% | 154 | 0.320 | Activation quantization |
| `flash_fwd_kernel` | 3.8% | 1,694 | 0.014 | Flash attention forward pass |
| `cutlass WMMA GEMM` | 3.6% | 444 | 0.053 | Tensor core GEMM |
| `elementwise_kernel (misc)` | 2.5% | 7,308 | 0.002 | Elementwise ops |
| `reduce_kernel` | 1.7% | 3,645 | 0.003 | Reduction ops |

### 4.3 Key insight

**51.8% of GPU time is one kernel called 11,858 times** — once per autoregressive token step. This is exactly the bottleneck Medusa targets.

Standard autoregressive decoding flow:
```
1 forward pass → 1 token accepted → repeat → 11,858 GEMM calls for ~76 tokens
```

Medusa flow (expected after training):
```
1 forward pass → multiple candidate tokens proposed → tree verification → several tokens accepted
→ dramatically fewer GEMM calls per accepted token
```

Expected improvement on RTX 4050 (6 GB VRAM): **1.5x–2.5x throughput** over baseline.

---

## 5. Next Steps

| # | Action |
|---|---|
| 1 | `pip3 install datasets` — required for Medusa training data |
| 2 | Prepare training data from ShareGPT dataset |
| 3 | Train Medusa heads (3–4 heads recommended for RTX 4050) |
| 4 | Run post-Medusa `nsys profile` with identical prompts |
| 5 | Compare kernel stats: launch count, throughput, latency |
| 6 | Generate comparison charts and final report |

---

## 6. Known Issues & Fixes

### `torch==2.3.1` not found for cu124

`torch 2.3.1` was never packaged for CUDA 12.4. Use `2.4.1` — all downstream components work fine.

---

### `ERROR: Package requires Python >=3.9`

Medusa's `pyproject.toml` restricts to `>=3.9` but the actual code runs on 3.8. Fix:

```bash
pip3 install wheel setuptools
pip3 install --no-build-isolation -e ".[train]"
```

---

### `ImportError: cannot import name 'is_flash_attn_available'`

Renamed to `is_flash_attn_2_available` in transformers 4.35. Medusa was never updated. Fix:

```bash
grep -rl "is_flash_attn_available" medusa/model/ \
  | xargs sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g'
```

---

### `error: invalid command 'bdist_wheel'`

pip 21 is missing the `wheel` package. Fix:

```bash
pip3 install wheel setuptools
```

---

### `OSError: Incorrect path '~/models/...'`

Python does not expand `~` in strings. Use the HuggingFace Hub ID directly:

```python
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# OR use os.path.expanduser if loading locally
model_path = os.path.expanduser("~/models/tinyllama")
```

---

### `E: Unable to locate package nsight-systems-2025.x.x`

NVIDIA CUDA apt repo not added yet. Fix:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
apt-cache search nsight-systems   # see available versions
sudo apt install -y nsight-systems-2025.6.3
```

---

*Update this file as new profiling results and findings are added.*
