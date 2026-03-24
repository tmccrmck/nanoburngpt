# Use NVIDIA CUDA 12.1 as base (Standard for RunPod/Lambda)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (Edition 2024 requires 1.85+)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify versions
RUN rustc --version && cargo --version

# Workspace setup
WORKDIR /app

# Step 1: Cache dependencies (The "Cargo Chef" style hack)
# We copy ONLY Cargo.toml and Cargo.lock first to cache the heavy compile
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    mkdir -p src/bin && echo "fn main() {}" > src/lib.rs && \
    cargo build --release --features cuda && \
    rm -rf src/

# Step 2: Copy actual source and build the real binary
COPY . .
RUN cargo build --release --features cuda

# Set execution permissions for the cloud helper script
RUN chmod +x run_cloud.sh

# Default behavior: Print help and wait (RunPod typical pattern)
CMD ["./run_cloud.sh"]
