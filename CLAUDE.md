# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**VieNeu-TTS** is an on-device Vietnamese text-to-speech library with instant voice cloning, emotion markers, and streaming capabilities.
- **v3 Turbo** (default): 48 kHz output, MOSS-Audio-Tokenizer-Nano codec, `sea-g2p` phonemizer. Runs on CPU (torch-free via ONNX Runtime) or GPU (PyTorch).
- **v3 Nano**: 24 kHz lightweight flow-matching model (48M params), torch-free ONNX on CPU for low-power devices.
- **Legacy backends**: Standard (GGUF), Fast (LMDeploy), Turbo GPU, XPU (Intel GPU), and Remote.

The core package stays minimal and torch-free by default; GPU, fine-tuning, and dev tooling are separated into optional dependency extras and groups in `pyproject.toml`.

---

## Development & Environment Setup

This project uses `uv` for dependency and environment management (`default-groups = []` in `pyproject.toml` keeps `uv sync` torch-free).

```bash
# Minimal install (torch-free: v3 Turbo ONNX + v3 Nano on CPU)
uv sync

# GPU / CUDA acceleration (v3 Turbo PyTorch backend)
uv sync --extra cuda

# Development & testing dependencies (pytest, pytest-asyncio, dev libraries)
uv sync --group dev

# Fine-tuning dependencies (PyTorch, PEFT, Accelerate, Safetensors)
uv sync --extra finetune

# Check system prerequisites and detected hardware
make check
```

---

## Key Commands

### Testing
Tests are located in `tests/` (and `finetune/tests/`). Because `pytest` is in the `dev` dependency group:

```bash
# Run all tests
uv run --group dev pytest

# Run with verbose output and short tracebacks (matching CI)
uv run --group dev pytest -v --tb=short

# Run a single test file
uv run --group dev pytest tests/test_capture_lock.py

# Run a specific test function
uv run --group dev pytest tests/test_capture_lock.py -k test_captures_do_not_overlap

# Run finetune data tests
uv run --group dev pytest finetune/tests/test_lora_data.py
```

### Linting & Formatting
```bash
ruff check .
ruff format --check .
ruff format .
```

### Running Applications
```bash
# Gradio Web UI
uv run vieneu-web
# or via Makefile
make run

# OpenAI-Compatible Streaming API Server (POST /v1/audio/speech on port 8000)
uv run python -m apps.openai_speech

# Force CPU ONNX backend on API server
VIENEU_BACKEND=onnx uv run python -m apps.openai_speech

# Web Stream Demo (FastAPI CPU int8 streaming on port 8001)
uv run vieneu-stream
# or via Makefile
make stream
```

### Docker
```bash
# Gradio UI on CPU / GPU
make docker-cpu
make docker-gpu

# OpenAI-compatible streaming API
docker compose -f docker/docker-compose.yml --profile api-gpu up
docker compose -f docker/docker-compose.yml --profile api-cpu up
```

### LoRA Fine-Tuning
```bash
# 1. Prepare dataset from metadata.csv and audio clips
uv run python finetune/prepare_dataset.py --dataset-dir finetune/dataset

# 2. Train LoRA adapter on GPU
uv run python finetune/train_lora.py --data finetune/dataset/train.parquet --run my_voice --merge
```

---

## Architecture & Code Structure

```
VieNeu-TTS/
├── src/
│   ├── vieneu/
│   │   ├── factory.py             # Vieneu(...) entrypoint dispatching backends
│   │   ├── base.py                # BaseVieneuTTS abstract class (voice management, audio I/O)
│   │   ├── v3turbo.py             # Primary v3 Turbo implementation (PyTorch / ONNX)
│   │   ├── v3nano.py              # v3 Nano implementation (flow-matching, ONNX, CPU-only)
│   │   ├── _v3_turbo_engine/      # Core modeling, ONNX runtime lite, denoiser, prompt logic
│   │   │   ├── capture_lock.py    # CAPTURE_LOCK: synchronizes CUDA graph captures vs GPU inference
│   │   │   ├── modeling_v3_turbo.py
│   │   │   ├── onnx_runtime_lite.py
│   │   │   └── rep_history.py     # Repetition penalty tracker
│   │   ├── v3_turbo_serve/        # High-throughput serving and streaming execution
│   │   │   ├── engine.py          # V3TurboBatchEngine static batching coordinator
│   │   │   ├── fused.py           # Whole-frame CUDA graph execution
│   │   │   ├── batched_backbone.py
│   │   │   ├── batched_acoustic.py
│   │   │   └── stream.py          # Streaming chunk generators
│   │   ├── standard.py            # Legacy GGUF / standard engine
│   │   ├── fast.py                # Legacy LMDeploy GPU engine
│   │   └── remote.py              # Remote API client engine
│   └── vieneu_utils/
│       ├── phonemize_text.py      # sea-g2p integration, punctuation normalization, chunking
│       ├── core_utils.py          # Audio joining, padding, babble/silence guards
│       └── url_extract.py         # URL extraction utilities
├── apps/
│   ├── gradio_main.py             # Web UI (vieneu-web entrypoint)
│   ├── openai_speech.py           # FastAPI OpenAI-compatible streaming endpoint (/v1/audio/speech)
│   ├── web_stream.py              # FastAPI CPU streaming demo (vieneu-stream entrypoint)
│   ├── srt_speech.py              # Subtitle / SRT batch synthesis with gap preservation
│   └── user_voices.py             # User voice profile management (JSON embeddings)
├── finetune/
│   ├── prepare_dataset.py         # Offline extraction of phonemes, codes, and speaker embeddings
│   ├── train_lora.py              # PEFT LoRA training loop
│   ├── merge_lora.py              # Merges trained LoRA adapters back into base models
│   └── vieneu_lora/               # Custom PyTorch Dataset and LoRA helpers
└── tests/                         # Pytest test suite covering engines, locks, utils, API
```

### Critical Architectural Concepts

1. **Backend Factory Dispatch (`src/vieneu/factory.py`)**:
   - `Vieneu(mode="v3turbo", ...)` is the standard interface.
   - `mode="v3turbo"` automatically selects between ONNX Runtime (CPU) and PyTorch (GPU/CUDA) based on environment and options.
   - On v3 Turbo, `style` is deprecated and ignored because voice style is derived directly from speaker embeddings and reference codes.

2. **Concurrency & GPU CUDA Graphs (`capture_lock.py` & `v3_turbo_serve/fused.py`)**:
   - In CUDA mode, v3 Turbo captures entire frame decoding loops into CUDA graphs for low latency.
   - Capturing graphs cannot overlap with other GPU executions. `CAPTURE_LOCK` enforces that graph captures hold an exclusive lock while standard inference jobs run with shared locks. Always respect this locking discipline when touching multi-threaded or serving paths.

3. **Text Normalization & Chunking (`vieneu_utils/phonemize_text.py`)**:
   - Vietnamese phonemization is powered by Rust-based `sea-g2p` with punctuation normalization.
   - Long texts are chunked via `normalize_to_chunks_v3_with_gaps` to maintain proper prosody, pause durations, and prevent repetition/babbling on long contexts.

4. **Serving & Streaming Contract (`apps/openai_speech.py`)**:
   - Supports raw chunked audio (`response_format="pcm"` / `"wav"`) and Server-Sent Events (`stream_format="sse"`).
   - Concurrency is throttled via `VIENEU_MAX_STREAMS` and `VIENEU_QUEUE`. CPU streaming uses ONNX while GPU streaming uses continuous/batched PyTorch streams.
