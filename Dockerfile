FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install basic native build dependencies
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    curl \
    build-essential \
    cmake \
    clang \
    libssl-dev \
    pkg-config \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install local Rust compiler toolchain runtime environment
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/loop-learn

ENV CUDA_COMPUTE_CAP=75

COPY Cargo.toml ./
RUN mkdir src && touch src/lib.rs && echo 'fn main() {}' > src/main.rs
RUN RUSTFLAGS="-L /usr/local/cuda/lib64 -l nccl" cargo build --release

RUN rm -rf src/
COPY src/ ./src/

RUN touch src/main.rs && RUSTFLAGS="-L /usr/local/cuda/lib64 -l nccl" cargo build --release

ENTRYPOINT ["target/release/loop-learn"]