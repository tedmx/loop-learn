FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Устанавливаем системные зависимости один раз внутри образа
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Rust фиксированной версии
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app
