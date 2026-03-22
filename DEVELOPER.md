# NanoBurnGPT: Project Overview & Status

This document is intended for AI agents and developers to understand the project's goal, architecture, and current implementation status.

## 🎯 Goal
Implement a character-level **NanoGPT** (decoder-only Transformer) based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) using the **Rust Burn** deep learning framework.

The project is optimized for a **MacBook Air** using the `wgpu` (Metal) backend, with a portable architecture that can scale to dedicated GPUs (CUDA/Vulkan).

## 🛠 Technologies
- **Language**: Rust (Edition 2024)
- **Framework**: [Burn 0.20.1](https://burn.dev/)
- **Backend**: `wgpu` (Cross-platform GPU)
- **Dataset**: Tiny Shakespeare (Character-level)
- **Utilities**: `clap` (CLI), `serde` (Config), `reqwest` (Data Download), `anyhow` (Error handling)

## 📚 Key References
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Burn Book](https://burn.dev/book/)
- [Burn Examples (Text Generation)](https://github.com/burn-rs/burn/tree/main/examples/text-generation)

## 🏗 Project Structure
- `src/main.rs`: CLI entry point (`train` and `generate` commands).
- `src/lib.rs`: Library module declarations.
- `src/data.rs`: Data pipeline (Tokenizer, Dataset splitting, Batching).
- `src/model.rs`: Transformer architecture (Self-Attention, MLP, Blocks, GPT).
- `src/train.rs`: Training loop configuration and `Learner` setup.
- `src/inference.rs`: Text generation logic and model loading.

## 🚦 Current Status
The project is structurally complete but is currently facing **compilation errors** due to recent API changes in **Burn 0.20**.

### ✅ Completed
- Character-level tokenizer and Shakespeare dataset downloader.
- Multi-head causal self-attention and Transformer block modules.
- GPT model structure and generation loop logic.
- CLI scaffolding for training and inference.

### 🚧 Known Issues (Immediate Tasks)
1. **Burn 0.20 API Mismatches**:
   - `LearnerBuilder` vs `Learner::builder`: The trainer needs to be updated to the latest `Learner` API.
   - `Tensor` operations: `squeeze`, `unsqueeze`, and `transpose` now require explicit dimension counts or generic parameters (e.g., `squeeze::<D>()`).
   - `TensorData` access: `into_data().value` is deprecated; need to use `as_slice` or `convert` appropriately for the new `TensorData` structure.
2. **Metric Traits**: `ValidStep` has been renamed or moved; need to use the correct trait for validation (likely `InferenceStep`).

## 🚀 How to Continue
1. **Fix `src/train.rs`**: Resolve the `Learner` initialization and metric trait issues.
2. **Fix `src/inference.rs`**: Resolve the `TensorData` access and `squeeze` calls.
3. **Verify**: Run `cargo check` until it passes.
4. **Train**: Run `cargo run -- train` to verify the pipeline.
5. **Generate**: Run `cargo run -- generate` once artifacts are produced.
