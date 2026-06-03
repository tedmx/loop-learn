# loop-learn

A minimalist, experimental playground designed to explore running local Large Language Models (LLMs) with a custom Retrieval-Augmented Generation (RAG) pipeline. Built entirely in Rust, **loop-learn** runs fully on local hardware, combining text generation and vector-based knowledge retrieval into a lightweight conversational terminal interface.

## Core Architecture

The system consists of isolated components working together in a clean, synchronous pipeline:

* **`src/loader.rs`**: Handles automated acquisition and caching of model files (both standard safetensors configurations and quantized GGUF blocks) directly from the Hugging Face Hub.
* **`src/inference.rs`**: Implements a low-level, token-by-token autoregressive generation engine with robust context tracking, dynamic repetition penalties, and explicit special token sanitization.
* **`src/presets.rs`**: Manages pre-configured model profiles (Presets), abstracting parameters like temperature, top_p, tokenizers, and hardware requirements.
* **`src/storage.rs`**: Manages persistent conversation history (chat sessions) and a memory-backed vector database for local knowledge lookup (RAG).

## Advanced Hardware Resilience (BF16 to F32 Fallback)

To accommodate varying hardware setups without manual configuration, the engine incorporates a **Dynamic Hardware Verification** step during initialization:
* When a non-quantized `BF16` (bfloat16) profile is selected on a CUDA device, the engine executes a lightweight mathematical probe directly on the GPU.
* If the underlying GPU architecture lacks native hardware execution units for `BF16` (such as the NVIDIA Turing architecture, e.g., RTX 2060), the engine catches the driver error and **safely promotes the computation context to `F32`**.
* This guarantees plug-and-play execution across both legacy and modern GPU architectures while preventing hard CUDA runtime driver crashes (`CUDA_ERROR_NOT_FOUND`).

## Model Presets

The application features built-in execution profiles optimized for specific VRAM footprints and token granularities:

| Preset Name | Model Target | VRAM Class | Format / Quantization | Template Layout |
| :--- | :--- | :--- | :--- | :--- |
| `qwen-1.5b-bf16` | Qwen2.5-1.5B-Instruct | 6 GB | Safetensors (BF16) | `chatml` |
| `qwen-3b-q4` | Qwen2.5-3B-Instruct-GGUF | 6 GB | Q4_K_M (Quantized) | `chatml` |
| `phi3-3.8b-q4` | Phi-3-mini-4k-instruct-GGUF | 6 GB | Q4_K_M (Quantized) | `phi3` |
| `qwen-7b-q4` | Qwen2.5-7B-Instruct-GGUF | 12 GB | Q4_K_M (Quantized) | `chatml` |
| `qwen-14b-q4` | Qwen2.5-14B-Instruct-GGUF | 12 GB | Q4_K_M (Quantized) | `chatml` |

> **Note:** The `llama-3.1-8b-q4` profile is currently archived in the source configuration due to Hugging Face repository gated access restrictions but remains available in the source file for manual reactivation.


## 💻 CLI Usage & Commands

You can control the runtime behavior directly through command-line arguments:

### 1. List Available Presets
To see the full matrix of available model configurations and their specific metadata, run:
```bash
cargo run -- --list-presets
```

### 2. Execute Inference with a Specific Profile

Pass your desired preset name using the --preset flag, accompanied by the prompt via -p:

```bash
# Run the lightweight Qwen model with automatic hardware validation
cargo run --release -- --preset qwen-1.5b-bf16 -p "Tell me about Overwatch 2"

# Run the heavy 14B model targeted for a 12GB VRAM environment
cargo run --release -- --preset qwen-14b-q4 -p "Explain quantum computing in simple terms"
```

### 3. Isolated Docker Environment

For reproducible environment setups, you can leverage the containerized execution matrix:

```bash
./run-container.sh --preset qwen-3b-q4 -p "Your custom query here"
```


## Environment Setup (Docker & NVIDIA Container Toolkit)
Prerequisites

Ensure your host machine has the NVIDIA Driver and Docker Engine installed.

### 1. Install NVIDIA Container Toolkit

Setup the package repository and install the runtime toolkit:

```bash
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -l [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 2. Configure Docker Runtime

Register the NVIDIA runtime within Docker configuration and restart the daemon:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Availability in Container

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Knowledge Base Customization

You can dynamically expand the model's memory by modifying the `storage/knowledge.txt` file. Organize your custom facts using explicit block headers to guide the vector search engine:

```
=== SUBJECT KEYWORDS ===
Your highly detailed text or custom data goes here...
```

The automation script mounts your local .cargo registry and huggingface cache inside the container, ensuring that subsequent runs reuse the build cache and start instantly.

## Tech Stack

* **Language**: Rust (for memory safety, predictable resource usage, and raw performance);
* **ML Framework**: Hugging Face Candle (a minimalist, pure-Rust tensor library with native CUDA support);
* **Data Serialization**: Serde & Serde-JSON (for robust chat transaction and history persistence).
