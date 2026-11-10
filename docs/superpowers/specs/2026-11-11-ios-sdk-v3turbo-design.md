# Design Document: VieNeu-TTS v3 Turbo iOS SDK (On-Device 48 kHz)

- **Date:** 2026-11-11
- **Status:** Approved / In Progress
- **Target OS:** iOS 16.0+
- **Distribution:** Swift Package Manager (SPM)

---

## 1. Intent & Architectural Overview

VieNeu-TTS v3 Turbo is an autoregressive transformer-based Vietnamese speech synthesis model with a 12-layer backbone (GQA attention, 768 hidden dimension) coupled with a 2-layer local acoustic decoder (16 RVQ codebooks) and the high-fidelity MOSS Audio Tokenizer (48 kHz output).

While v3 Nano provides lightweight flow-matching on CPU (24 kHz), v3 Turbo delivers **studio-grade 48 kHz fidelity, native handling of loan words/code-switching, natural emotion tags, and instant time-to-first-audio (TTFA < 180 ms)** by streaming acoustic frames as they are autoregressively generated.

This design brings **VieNeu-TTS v3 Turbo completely on-device for iOS 16.0+**, using quantized INT8 ONNX models optimized for Apple Silicon (ARM NEON) via ONNX Runtime Mobile.

---

## 2. Technical Stack & Component Architecture

```
+-----------------------------------------------------------------------------------+
|                                   Host iOS App                                    |
|                      (SwiftUI / UIKit / Audio Streaming Engine)                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        VieNeuTTS (Facade: mode = .v3turbo)                         |
|                                                                                   |
|  +---------------------------+       +-----------------------------------------+  |
|  |     SeaG2P (Rust C-ABI)   | ----> |  Phoneme Tokenizer (v3 Turbo vocab)     |  |
|  +---------------------------+       +-----------------------------------------+  |
|                                                          |                        |
|                                                          v                        |
|                                      +-----------------------------------------+  |
|                                      |  V3TurboEngine                          |  |
|                                      |  - Embedding Matrix & Speaker Anchor    |  |
|                                      |  - vieneu_prefill.onnx                  |  |
|                                      |  - Loop:                                |  |
|                                      |    * vieneu_acoustic_cached.onnx        |  |
|                                      |    * Top-K / Top-P / Rep-Pen Sampling   |  |
|                                      |    * vieneu_decode_step.onnx (KV Cache) |  |
|                                      |    * MOSS Codec Stream Decoder (48 kHz) |  |
|                                      +-----------------------------------------+  |
|                                                          |                        |
|                                                          v                        |
|                                      +-----------------------------------------+  |
|                                      |  Streaming PCM Audio (48,000 Hz, Mono)  |  |
|                                      +-----------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Component 1: `V3TurboEngine` & Weights
- **Model Files:**
  - `vieneu_prefill.onnx` (1.04 MB): Prefills prompt token embeddings and computes initial KV cache.
  - `vieneu_decode_step.onnx` + `vieneu_backbone_shared.data` (99.08 MB): Autoregressively advances the 12-layer backbone step-by-step with 12 KV cache pairs.
  - `vieneu_acoustic_cached.onnx` (6.87 MB): 2-layer local acoustic transformer generating 16 RVQ codebooks per audio frame.
  - `vieneu_v3_heads.bin` / `.data` (49.80 MB): Memory-mapped embedding tables (`text_emb: [419, 768]`, `audio_emb: [16, 1024, 768]`, and `xvec_proj` linear & LayerNorm weights).
  - `moss_audio_tokenizer_decode_full.onnx` + `.data` (42 MB) / `moss_audio_tokenizer_decode_step.onnx`: Decodes 16-RVQ frames to 48,000 Hz PCM audio.

### Component 2: Embedding & Speaker Projection (Swift SIMD / Accelerate)
- Maps phoneme tokens + audio prompt codes into 768-dimensional float vectors.
- Projects 192-dimensional x-vector speaker embedding into a 768-dimensional global conditioning anchor:
  $$\mathbf{a} = \text{LayerNorm}(\mathbf{s} \cdot \mathbf{W}_{\text{xvec}}^T + \mathbf{b}_{\text{xvec}})$$
- Employs Apple `Accelerate` (`vDSP_mmul`, `vDSP_normalize`) for sub-millisecond anchor projection.

### Component 3: Sampling Pipeline (Top-K, Top-P, Repetition Penalty)
- Logit manipulation with repetition penalty applied to recent history window.
- Fast Top-K selection (`argpartition`) followed by softmax over top-$K$ candidates and nucleus (Top-P) cumulative truncation.

### Component 4: Streaming Codec Synthesizer (48 kHz)
- Each acoustic step produces 16 codes $\in [0, 1024)$.
- Decodes into 960 audio samples per frame ($48,000 \text{ Hz} / 50 \text{ fps} = 960$ samples).
- Yields immediate audio buffers for gapless playback via `AVAudioPlayer` or `AVAudioEngine`.

---

## 3. Public API Design

```swift
import VieNeuTTS

// 1. Initialize with v3 Turbo mode (48 kHz output)
let config = VieNeuConfig(
    mode: .v3turbo,
    modelProvider: CachedModelProvider(directory: turboModelDir),
    temperature: 0.8,
    topK: 25,
    topP: 0.95
)
let tts = try await VieNeuTTS(config: config)

// 2. Synthesize with natural 48 kHz voice
let voice = try tts.loadPresetVoice(name: "Minh Quân")
let audio = try await tts.synthesize(text: "Xin chào! Đây là VieNeu-TTS v3 Turbo 48 kHz.", voice: voice)
print("Audio sample rate: \(audio.sampleRate) Hz") // 48000 Hz

// 3. Real-time streaming (First audio < 180 ms)
for try await chunk in tts.synthesizeStream(text: "Đọc văn bản tức thì theo thời gian thực.", voice: voice) {
    chunk.audio.play()
}
```

---

## 4. Resource & Memory Budget

- **Model Download Footprint:** ~190 MB total (Backbone INT8 99MB + Heads 50MB + MOSS Codec 42MB).
- **RAM Runtime Budget:** $\le 320$ MB (well within 4 GB – 8 GB iOS limits).
- **Inference Speed:** RTF $\approx 0.25 - 0.40$ on A14+ Apple Silicon, real-time audio playback without underrun.
