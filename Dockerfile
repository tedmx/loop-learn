FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install basic native build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install local Rust compiler toolchain runtime environment
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/loop-learn

COPY Cargo.toml ./
RUN mkdir src && touch src/lib.rs && echo 'fn main() {}' > src/main.rs

RUN cargo build --release

RUN rm -rf src/
COPY src/ ./src/

RUN touch src/main.rs && cargo build --release

ENTRYPOINT ["cargo", "run", "--release", "--"]