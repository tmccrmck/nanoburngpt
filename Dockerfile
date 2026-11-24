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

# Install Rust (rust-toolchain.toml pins the exact channel)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
ENV PATH="/root/.cargo/bin:${PATH}"

# Skip host LTO (GPU kernels are JIT-compiled) for faster container builds
ENV CARGO_PROFILE_RELEASE_LTO=off
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16

# Workspace setup
WORKDIR /app

# Step 1: Cache dependencies (including vendored pliron patch & pinned toolchain)
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY vendor ./vendor
RUN rustup show && \
    rustc --version && cargo --version && \
    mkdir src && echo "fn main() {}" > src/main.rs && \
    echo "" > src/lib.rs && \
    cargo build --release --no-default-features --features cuda && \
    rm -rf src/

# Step 2: Copy actual source and build the real binary
COPY . .
RUN touch src/main.rs src/lib.rs && cargo build --release --no-default-features --features cuda

# Set execution permissions for the cloud helper script
RUN chmod +x run_cloud.sh

# Default behavior: Print help and wait (RunPod typical pattern)
CMD ["./run_cloud.sh"]
