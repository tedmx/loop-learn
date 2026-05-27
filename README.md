# loop-learn

A minimalist, experimental playground designed to explore running local Large Language Models (LLMs) with a custom Retrieval-Augmented Generation (RAG) pipeline. Built entirely in Rust, **loop-learn** runs fully on local hardware, combining text generation and vector-based knowledge retrieval into a lightweight conversational terminal interface.

## Core Architecture

The system consists of a few isolated components working together in a clean, synchronous pipeline:

* **`src/loader.rs`**: Handles automated acquisition and caching of model files (Qwen LLM and sentence-transformer embeddings) directly from the Hugging Face Hub.
* **`src/inference.rs`**: Implements a low-level, token-by-token autoregressive generation engine using a strict greedy search decoding strategy.
* **`src/storage.rs`**: Manages persistent conversation history (chat sessions) and a memory-backed vector database for local knowledge lookup.

## 🛠 Tech Stack

* **Language**: Rust (for memory safety, predictable resource usage, and raw performance);
* **ML Framework**: Hugging Face Candle (a minimalist, pure-Rust tensor library with native CUDA support);
* **Models**: 
  * **LLM**: `Qwen2.5-1.5B-Instruct` (optimized for local, low-latency multi-turn dialogue);
  * **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (for cross-lingual semantic search);
* **Target Hardware**: CUDA-enabled GPUs (utilizes `F32` precision to guarantee out-of-the-box compatibility without precision artifacts);
* **Environment**: Docker + WSL2 (for deterministic isolation of CUDA toolkits and dependencies).

## Getting Started

### Prerequisites & Environment Setup

The system runs inside a Docker container, isolating the CUDA compilation environment from your host system. 

*(Note: If you prefer using Windows-side Docker Desktop with WSL integration, simply enable it in Docker Desktop settings. The guide below covers the native, pure Docker installation inside the WSL distribution).*

#### 1. Install Docker Engine inside WSL (Ubuntu)
Execute the following commands sequentially to set up the official Docker repository and install the engine:

```bash
# Update package indices and install system dependencies
sudo apt-get update;
sudo apt-get install -y ca-certificates curl gnupg;

# Add Docker’s official GPG key
sudo install -m 0755 -d /etc/apt/keyrings;
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg;
sudo chmod a+r /etc/apt/keyrings/docker.gpg;

# Set up the repository structure
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null;

# Install Docker components
sudo apt-get update;
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin;
```

#### 2. Configure Non-Root Permissions

To allow the execution script and rust-analyzer to communicate with the Docker daemon without constant sudo authentication, add your user to the docker group:

```bash
sudo usermod -aG docker $USER;
newgrp docker;
```

#### 3. Install NVIDIA Container Toolkit (GPU Passthrough)

To map your host's NVIDIA graphics card (Turing, Ampere, etc.) into the container, the Docker daemon requires the native NVIDIA runtime toolkit:

```bash
# Configure the production repository keys
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg;
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list;

# Install the toolkit packages
sudo apt-get update;
sudo apt-get install -y nvidia-container-toolkit;

# Register the NVIDIA runtime within Docker configuration and restart the daemon
sudo nvidia-ctk runtime configure --runtime=docker;
sudo systemctl restart docker;
```

#### 4. Verify GPU Availability in Container
Ensure that the setup was successful and Docker can transparently access the hardware:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```
If the command outputs the standard NVIDIA driver statistics table, the environment configuration is complete.


### Running Inference

To initialize the engine, index the local knowledge base, and pass a custom prompt directly to the underlying Qwen model, use the automated execution script:

```bash
# Grant execution permissions to the runner script
chmod +x run-container.sh

# Run inference with a specific question
./run-container.sh -p "What color is planet Earth?"
```

### Knowledge Base Customization

You can dynamically expand the model's memory by modifying the `storage/knowledge.txt` file. Organize your custom facts using explicit block headers to guide the vector search engine:

```
=== SUBJECT KEYWORDS ===
Your highly detailed text or custom data goes here...
```

The automation script mounts your local .cargo registry and huggingface cache inside the container, ensuring that subsequent runs reuse the build cache and start instantly.
