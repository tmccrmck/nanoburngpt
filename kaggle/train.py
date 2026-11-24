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

MODEL = "gpt2-small"
DATASET = "shakespeare"
OPTIMIZER = "adamw"  # adamw or muon (Muon for 2D block weights + AdamW for embeddings/norms)
DTYPE = "flex32"  # flex32 unlocks FP16 Tensor Cores on T4 with FP32 storage
MULTI_GPU = "auto"  # auto uses 1x T4 for flex32 (NCCL rejects flex32) or 2x for f32/bf16
GRAD_CHECKPOINTING = False
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # 1x T4 x batch 8 x accum 4 = effective batch 32 sequences (matches v2 baseline)
BLOCK_SIZE = 256  # 0 = use preset default
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
WARMUP_ITERS = 30
NUM_WORKERS = 2
# 0 = full dataset (e.g. set MAX_TRAIN_ITEMS = 4000, MODEL = "nano",
# NUM_EPOCHS = 1 for a quick end-to-end smoke test).
MAX_TRAIN_ITEMS = 0

GENERATE_PROMPT = "HAMLET:\n"
GENERATE_TOKENS = 200
REPETITION_PENALTY = 1.1
GENERATE_DTYPE = "f16"
USE_BEST_CHECKPOINT = True

REPO_DIR = Path("/tmp/nanoburngpt")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_ARTIFACTS = OUTPUT_DIR / "artifacts"
CACHED_INPUT_DIR = Path("/kaggle/input/nanoburngpt-cuda-train")


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

    # 2. Clone into /tmp so Rust's target/ stays out of /kaggle/working
    banner(f"Clone {REPO_URL} ({BRANCH})")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(REPO_DIR)])

    os.chdir(REPO_DIR)
    print(f"Working directory: {Path.cwd()}", flush=True)
    run(["git", "log", "-n", "1", "--oneline"])
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"HEAD SHA: {head_sha}", flush=True)

    # 3. Restore cached CUDA binary if SHA matches, otherwise install Rust & build
    binary_path = REPO_DIR / "target/release/nanoburngpt"
    cached_bin = CACHED_INPUT_DIR / "nanoburngpt"
    cached_sha_file = CACHED_INPUT_DIR / "nanoburngpt.sha"

    if (
        cached_bin.exists()
        and cached_sha_file.exists()
        and cached_sha_file.read_text().strip() == head_sha
    ):
        banner("Using Cached CUDA Binary")
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_bin, binary_path)
        binary_path.chmod(0o755)
        print(f"Restored {cached_bin} ({head_sha[:8]})", flush=True)
    else:
        banner("Rust Toolchain")
        cargo_bin = str(Path.home() / ".cargo" / "bin")
        if cargo_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = cargo_bin + os.pathsep + os.environ.get("PATH", "")

        if not shutil.which("cargo"):
            print("Installing Rust via rustup...", flush=True)
            run([
                "sh",
                "-c",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none",
            ])

        run(["rustc", "--version"])
        run(["cargo", "--version"])

        banner("Building NanoBurnGPT (CUDA)")
        build_env = os.environ.copy()
        build_env["CARGO_PROFILE_RELEASE_LTO"] = "off"
        build_env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "16"

        t0 = time.monotonic()
        run(
            ["cargo", "build", "--release", "--no-default-features", "--features", "cuda"],
            env=build_env,
        )
        print(f"Build completed in {time.monotonic() - t0:.1f}s", flush=True)

    binary = str(binary_path)

    # 4. Train on GPU
    banner(f"Training ({MODEL} on {DATASET}, opt={OPTIMIZER}, dtype={DTYPE}, multi_gpu={MULTI_GPU})")
    train_cmd = [
        binary,
        "train",
        "--model", MODEL,
        "--dataset", DATASET,
        "--optimizer", OPTIMIZER,
        "--dtype", DTYPE,
        "--multi-gpu", MULTI_GPU,
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum-steps", str(GRAD_ACCUM_STEPS),
        "--num-epochs", str(NUM_EPOCHS),
        "--learning-rate", str(LEARNING_RATE),
        "--warmup-iters", str(WARMUP_ITERS),
        "--num-workers", str(NUM_WORKERS),
        "--no-tui",
    ]
    if GRAD_CHECKPOINTING:
        train_cmd.append("--grad-checkpointing")
    if BLOCK_SIZE > 0:
        train_cmd += ["--block-size", str(BLOCK_SIZE)]
    if MAX_TRAIN_ITEMS > 0:
        train_cmd += ["--max-train-items", str(MAX_TRAIN_ITEMS)]

    t0 = time.monotonic()
    run(train_cmd)
    print(f"Training completed in {time.monotonic() - t0:.1f}s", flush=True)

    # 5. Generate sample text from trained checkpoint
    banner("Generating Sample Text")
    gen_cmd = [
        binary,
        "generate",
        "--artifact-dir", "artifacts",
        "--prompt", GENERATE_PROMPT,
        "--max-tokens", str(GENERATE_TOKENS),
        "--temperature", "0.8",
        "--top-k", "40",
        "--top-p", "0.95",
        "--repetition-penalty", str(REPETITION_PENALTY),
        "--dtype", GENERATE_DTYPE,
    ]
    if USE_BEST_CHECKPOINT:
        gen_cmd.append("--best")
    run(gen_cmd)

    # 6. Export artifacts and CUDA binary to /kaggle/working
    banner("Exporting Artifacts & Binary")
    src_artifacts = REPO_DIR / "artifacts"
    if not src_artifacts.exists():
        raise FileNotFoundError(f"Expected artifacts directory at {src_artifacts}")
    shutil.copytree(src_artifacts, OUTPUT_ARTIFACTS, dirs_exist_ok=True)
    exported_bin = OUTPUT_DIR / "nanoburngpt"
    shutil.copy2(binary_path, exported_bin)
    exported_bin.chmod(0o755)
    (OUTPUT_DIR / "nanoburngpt.sha").write_text(head_sha + "\n")
    for path in sorted(OUTPUT_ARTIFACTS.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{path} ({size_mb:.2f} MB)", flush=True)
    print(
        f"{exported_bin} ({exported_bin.stat().st_size / (1024 * 1024):.2f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
