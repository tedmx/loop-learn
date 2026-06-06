# loop-learn

A minimalist, experimental playground designed to explore running local Large Language Models (LLMs) with a custom Retrieval-Augmented Generation (RAG) pipeline. Built entirely in Rust, **loop-learn** runs fully on local hardware, combining text generation and vector-based knowledge retrieval into a lightweight conversational terminal interface.

## Core Architecture

The system consists of isolated components working together in a clean, synchronous pipeline:

* **`src/loader.rs`**: Handles automated acquisition, verification, and local path-mapping of quantized GGUF blocks.
* **`src/inference.rs`**: Implements a native, token-by-token autoregressive generation engine backed by **`llama-cpp-2` bindings** for high-performance GGUF execution.
* **`src/presets.rs`**: Manages pre-configured model profiles (Presets), abstracting parameters like temperature, top_p, and hardware-specific VRAM allocation targets.
* **`src/storage.rs`**: Manages persistent conversation history (chat sessions) and the `VectorRegistry` — a memory-backed vector database for local knowledge lookup (RAG) featuring local JSON caching.

---

## Storage & Required Assets

Before running the application, you need to populate the `storage/` directory. Due to their large size, model files and embedding layers must be downloaded manually from the Hugging Face Hub (or your registry of choice):

1. **GGUF Models:** Download your target models (e.g., Qwen, Phi-3, or Llama) in `.gguf` format and place them under `storage/models/`.
2. **Embedding Weights:** If you are using a local vector-transformer configuration for RAG, ensure the corresponding configuration and weight files are placed inside `storage/embeddings/`.

On the first run, the engine indexes `storage/knowledge.txt` and caches the computed vectors into `storage/embeddings.json`. Subsequent boots skip the indexing phase entirely if the source file remains unchanged.

---

## Model Presets

The application features built-in execution profiles optimized for specific VRAM footprints:

| Preset Name | Model Target | VRAM Class | Format / Quantization | Template Layout |
| :--- | :--- | :--- | :--- | :--- |
| `qwen-3b-q4` | Qwen2.5-3B-Instruct-GGUF | 6 GB | Q4_K_M (Quantized) | `chatml` |
| `phi3-3.8b-q4` | Phi-3-mini-4k-instruct-GGUF | 6 GB | Q4_K_M (Quantized) | `phi3` |
| `llama-3b-q4` | Llama-3-Instruct-GGUF | 6 GB | Q4_K_M (Quantized) | `llama3` |
| `qwen-7b-q4` | Qwen2.5-7B-Instruct-GGUF | 12 GB | Q4_K_M (Quantized) | `chatml` |

---

## CLI Usage & Commands

You can control the runtime behavior directly through command-line arguments:

### 1. List Available Presets
To see the full matrix of available model configurations and their specific metadata, run:
```bash
cargo run -- --list-presets
```

### 2. Execute Inference with a Specific Profile

Pass your desired preset name using the --preset flag, accompanied by the prompt via -p:

```bash
# Run a lightweight model targeted for a 6GB VRAM environment
cargo run --release -- --preset qwen-3b-q4 -p "Tell me about local LLM inference optimizations"

# Run a heavy model targeted for a 12GB VRAM environment
cargo run --release -- --preset qwen-7b-q4 -p "Explain quantum computing in simple terms"
```

### 3. Isolated Docker Environment

For reproducible environment setups, you can leverage the containerized execution matrix:

```bash
./run-container.sh --preset qwen-3b-q4 -p "Your custom query here"
```

## Environment Setup (Docker & NVIDIA Container Toolkit)

### Prerequisites

Ensure your host machine has the NVIDIA Driver (supporting CUDA 12.2+) and Docker Engine installed.

### 1. Install NVIDIA Container Toolkit

Setup the package repository and install the runtime toolkit:

```bash
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
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
docker run --rm --gpus all nvidia/cuda:12.2.2-devel-ubuntu22.04 nvidia-smi
```
## Knowledge Base Customization

You can dynamically expand the model's memory by modifying the `storage/knowledge.txt` file. Organize your custom facts using explicit block headers to guide the vector search engine:

```
=== SUBJECT KEYWORDS ===
Your highly detailed text or custom data goes here...
```

💡 Performance Note: When you spin up the pipeline, `loop-learn` validates the state of `knowledge.txt`. If it hasn't changed, it instantly loads pre-computed vectors from `storage/embeddings.json` instead of processing the text again.

## Tech Stack

* Language: Rust (for memory safety, predictable resource usage, and raw performance);
* Inference Engine: llama-cpp-2 bindings (safe Rust wrapper around llama.cpp for hardware-accelerated GGUF execution);
* Data Serialization: Serde & Serde-JSON (for robust chat transaction and vector cache persistence).