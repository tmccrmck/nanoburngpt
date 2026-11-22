"""Kaggle GPU training runner for NanoBurnGPT (Burn 0.22.1 + CUDA)."""

import os
import shutil
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_URL = "https://github.com/tmccrmck/nanoburngpt.git"
BRANCH = "main"

MODEL = "nano"
DATASET = "shakespeare"
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
BLOCK_SIZE = 0  # 0 = use preset default
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
WARMUP_ITERS = 20
NUM_WORKERS = 2
# 0 = full dataset (e.g. set MAX_TRAIN_ITEMS = 4000, MODEL = "nano",
# NUM_EPOCHS = 1 for a quick end-to-end smoke test).
MAX_TRAIN_ITEMS = 4000

GENERATE_PROMPT = "HAMLET:\n"
GENERATE_TOKENS = 100

REPO_DIR = Path("/tmp/nanoburngpt")
OUTPUT_ARTIFACTS = Path("/kaggle/working/artifacts")


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, env=env)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return res.returncode


def main() -> None:
    # 1. GPU, CUDA, disk, and RAM diagnostics
    banner("GPU & System Diagnostics")
    if run(["nvidia-smi"], check=False) != 0:
        raise RuntimeError(
            "nvidia-smi failed; make sure the Kaggle kernel has a GPU accelerator enabled."
        )
    if shutil.which("nvcc"):
        run(["nvcc", "--version"], check=False)
    else:
        print("nvcc not on PATH; cubecl-cuda uses NVRTC/driver dynamic loading.", flush=True)
    print("CUDA dirs:", sorted(str(p) for p in Path("/usr/local").glob("cuda*")), flush=True)
    run(["df", "-h", "/tmp", "/kaggle/working"], check=False)
    run(["free", "-h"], check=False)

    # 2. Install Rust toolchain if needed
    banner("Rust Toolchain")
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    if cargo_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = cargo_bin + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("cargo"):
        print("Installing Rust via rustup...", flush=True)
        run([
            "sh",
            "-c",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        ])

    # 3. Clone into /tmp so Rust's target/ stays out of /kaggle/working
    banner(f"Clone {REPO_URL} ({BRANCH})")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(REPO_DIR)])

    # 4. Enter repo and confirm active toolchain (triggers rust-toolchain.toml)
    os.chdir(REPO_DIR)
    print(f"Working directory: {Path.cwd()}", flush=True)
    run(["git", "log", "-n", "1", "--oneline"])
    run(["rustc", "--version"])
    run(["cargo", "--version"])

    # 5. Build release binary with CUDA only (no wgpu, no TUI)
    banner("Building NanoBurnGPT (CUDA)")
    build_env = os.environ.copy()
    build_env["CARGO_PROFILE_RELEASE_LTO"] = "thin"
    build_env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "16"

    t0 = time.monotonic()
    run(
        ["cargo", "build", "--release", "--no-default-features", "--features", "cuda"],
        env=build_env,
    )
    print(f"Build completed in {time.monotonic() - t0:.1f}s", flush=True)

    binary = "./target/release/nanoburngpt"

    # 6. Train on GPU
    banner(f"Training ({MODEL} on {DATASET})")
    train_cmd = [
        binary,
        "train",
        "--model", MODEL,
        "--dataset", DATASET,
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum-steps", str(GRAD_ACCUM_STEPS),
        "--num-epochs", str(NUM_EPOCHS),
        "--learning-rate", str(LEARNING_RATE),
        "--warmup-iters", str(WARMUP_ITERS),
        "--num-workers", str(NUM_WORKERS),
        "--no-tui",
    ]
    if BLOCK_SIZE > 0:
        train_cmd += ["--block-size", str(BLOCK_SIZE)]
    if MAX_TRAIN_ITEMS > 0:
        train_cmd += ["--max-train-items", str(MAX_TRAIN_ITEMS)]

    t0 = time.monotonic()
    run(train_cmd)
    print(f"Training completed in {time.monotonic() - t0:.1f}s", flush=True)

    # 7. Generate sample text from trained checkpoint
    banner("Generating Sample Text")
    run([
        binary,
        "generate",
        "--artifact-dir", "artifacts",
        "--prompt", GENERATE_PROMPT,
        "--max-tokens", str(GENERATE_TOKENS),
        "--temperature", "0.8",
        "--top-k", "40",
        "--top-p", "0.95",
    ])

    # 8. Export artifacts to /kaggle/working/artifacts
    banner("Exporting Artifacts")
    src_artifacts = REPO_DIR / "artifacts"
    if not src_artifacts.exists():
        raise FileNotFoundError(f"Expected artifacts directory at {src_artifacts}")
    shutil.copytree(src_artifacts, OUTPUT_ARTIFACTS, dirs_exist_ok=True)
    for path in sorted(OUTPUT_ARTIFACTS.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{path} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
