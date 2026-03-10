# Tech Stack

## Current Stack
- Python 3 for training, evaluation, and runtime orchestration.
- PyTorch reference HRM implementations under `hrm/`.
- Kotlin/Native bridge for ANE experiments under `kotlin/`.
- Local data and checkpoints stored on disk.

## Torch Option Intent
- Primary research path: PyTorch models with export to Core ML for inference.
- MPS fallback for training on Apple Silicon when ANE mapping gaps exist.
- Private ANE runtime is experimental, used only for targeted kernels where verified.
