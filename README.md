# loop-learn

An experimental playground designed to explore the construction of a local, fully autonomous self-learning AI system. Instead of treating Large Language Models as static black boxes via remote APIs, **loop-learn** runs entirely on local hardware, bridging real-time inference and continuous training into a unified feedback loop.

---

## Core Architecture

The system is built as a closed-loop environment where the model learns directly from its own execution context and performance metrics:

* **`src/loader.rs`**: Handles automated acquisition and caching of models (e.g., Qwen2.5) directly via the Hugging Face Hub API.
* **`src/inference.rs`**: Implements a low-level, token-by-token autoregressive generation engine with direct GPU memory and KV-cache management.
* **`src/storage.rs`**: Manages the local experience replay buffer and persistent memory.
* **`src/train.rs`**: Governs the background optimization and backpropagation loops to update model weights on the fly.

---

## 🛠 Tech Stack

* **Language**: Rust (for memory safety, predictability, and raw systems performance);
* **ML Framework**: Hugging Face Candle (minimalist, pure-Rust tensor library with native CUDA support);
* **Target Architecture**: Qwen2.5-Instruct (optimized for local multi-turn conversations);
* **Target Hardware**: CUDA-enabled GPUs (utilizes `F32` precision to guarantee compatibility across NVIDIA Turing, Ampere, and newer architectures without NaN artifacts);
* **Environment**: Docker + WSL2 (for complete isolation of CUDA toolkits and build targets).

---

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

To initialize the engine and pass a custom prompt directly to the underlying Qwen model, use the automated shell script:

```bash
# Grant execution permissions to the runner script
chmod +x run-container.sh

# Execute inference
./run-container.sh
```

The automation script mounts your local .cargo registry inside the container, ensuring that subsequent compilations reuse the build cache and complete incrementally in under a second.
