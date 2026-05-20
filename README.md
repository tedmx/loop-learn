# loop-learn

An experimental playground designed to explore the construction of a local, fully autonomous self-learning AI system. Instead of treating Large Language Models as static black boxes via remote APIs, **loop-learn** runs entirely on local hardware, bridging real-time inference and continuous training into a unified feedback loop.

---

## 🚀 Core Architecture

The system is built as a closed-loop environment where the model learns directly from its own execution context and performance metrics:

* **`src/loader.rs`**: Handles automated acquisition and caching of models (e.g., TinyLlama) directly via the Hugging Face Hub API.
* **`src/inference.rs`**: Implements a low-level, token-by-token autoregressive generation engine with direct GPU memory and KV-cache management.
* **`src/storage.rs`**: Manages the local experience replay buffer and persistent memory.
* **`src/train.rs`**: Governs the background optimization and backpropagation loops to update model weights on the fly.

---

## 🛠 Tech Stack

* **Language**: Rust (for memory safety, predictability, and raw systems performance)
* **ML Framework**: Hugging Face Candle (minimalist, pure-Rust tensor library with native CUDA support)
* **Target Hardware**: CUDA-enabled GPUs (utilizes `F16` precision for efficient compute and footprint matching)

---

## 🏃‍♂️ Getting Started

### Prerequisites
Ensure you have Rust (stable) and the CUDA toolkit installed on your system / WSL environment.

### Run Inference
To spin up the system and test the current baseline generation:
```bash
cargo run --release
```