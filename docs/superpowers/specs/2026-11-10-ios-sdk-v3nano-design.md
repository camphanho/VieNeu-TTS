# Design Document: VieNeu-TTS v3 Nano iOS SDK (On-Device)

- **Date:** 2026-11-10
- **Status:** Review / Ready for Approval
- **Target OS:** iOS 16.0+
- **Distribution:** Swift Package Manager (SPM)

---

## 1. Intent & The Why

VieNeu-TTS v3 Nano is a 48M-parameter flow-matching Vietnamese text-to-speech model producing 24 kHz audio. It is designed specifically for resource-constrained, CPU-bound devices where heavier transformer models (such as v3 Turbo) cannot meet real-time requirements.

The goal of this project is to build an iOS-native, completely on-device (offline) SDK (`VieNeuTTS`) that enables mobile applications to synthesize natural Vietnamese speech and clone voices locally without cloud dependencies, API keys, or privacy concerns.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **100% On-Device Offline Operation:** Synthesize Vietnamese speech locally on iPhone and iPad (iOS 16.0+) without internet connectivity.
- **Low Latency & Efficient Inference:** Leverage ONNX Runtime Mobile for iOS (CPU execution provider optimized with Accelerate/NEON) to achieve faster-than-real-time generation on modern Apple Silicon (A14+).
- **Native Language Processing:** Package `sea-g2p` (Rust) into a native Apple XCFramework (`SeaG2P.xcframework`) for punctuation normalization and Vietnamese grapheme-to-phoneme conversion.
- **Preset Voice Synthesis:** Provide zero-latency generation using preset voices (built-in x-vector speaker embeddings).
- **Instant Voice Cloning:** Allow users to enroll and clone a new voice from a 3–8 second audio clip directly on the device, producing reusable `VoiceProfile` artifacts.
- **Dual Model Provisioning:** Support loading model assets either directly from the application bundle (`Bundle.main`) or dynamically downloaded and cached in the app sandbox (`Application Support/VieNeu-TTS/Cache`).
- **Idiomatic Swift API:** Provide a modern Swift API using `async/await`, structured concurrency, chunk-level streaming (`AsyncThrowingStream`), and seamless integration with `AVAudioEngine` / `AVAudioPlayer`.

### Non-Goals
- Support for v3 Turbo or legacy GGUF/LMDeploy backends in this milestone (v3 Nano flow-matching only).
- Android, Flutter, or React Native wrappers (focus strictly on a clean, native Swift/iOS SPM package first).
- Server-assisted generation or cloud streaming fallback.

### Anti-Goals
- **No Python Runtimes or Heavy Intermediaries:** Zero reliance on Python bridging, PyTorch Mobile, or external network requests during synthesis.
- **No Main-Thread Blocking:** Intensive computation (phonemization, ODE flow integration, audio spectrogram extraction, and neural decoding) must never freeze the UI.
- **No Inflexible Monolithic Assets:** Applications must not be forced to bundle 200MB+ models inside their App Store binary if they prefer on-demand downloads.

---

## 3. Architecture & Technical Components

```
+---------------------------------------------------------------------------------+
|                                 Host iOS App                                    |
|                      (SwiftUI / UIKit / Audio Playback)                         |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                              VieNeuTTS (Swift SDK)                              |
|                                                                                 |
|  +---------------------------+       +---------------------------------------+  |
|  |     VieNeuTTS (Facade)     | <---> |  ModelProvider (Bundle / Cache)       |  |
|  +---------------------------+       +---------------------------------------+  |
|                |                                         |                      |
|                v                                         v                      |
|  +---------------------------+       +---------------------------------------+  |
|  |     Text Normalizer       |       |  ONNXEngine                           |  |
|  |  (SeaG2P.xcframework)     |       |  - text_encoder.onnx                  |  |
|  +---------------------------+       |  - duration_predictor.onnx            |  |
|                |                     |  - vector_estimator.onnx              |  |
|                v                     |  - codec_decoder.onnx                 |  |
|  +---------------------------+       +---------------------------------------+  |
|  |     Phoneme Tokenizer     |                           |                      |
|  +---------------------------+                           v                      |
|                |                     +---------------------------------------+  |
|                +-------------------> |  FlowSolver (Euler ODE + CFG)         |  |
|                                      +---------------------------------------+  |
|                                                          |                      |
|                                                          v                      |
|                                      +---------------------------------------+  |
|                                      |  Audio Output (24 kHz PCM / WAV)      |  |
|                                      +---------------------------------------+  |
+---------------------------------------------------------------------------------+
                                       | (Voice Cloning Pipeline)
                                       v
+---------------------------------------------------------------------------------+
|  VoiceCloner (Native Accelerate/vDSP fbank + speaker/codec/reference encoders) |
+---------------------------------------------------------------------------------+
```

### Component 1: `SeaG2P.xcframework` (C-ABI Rust Bridge)
- **Source:** Upstream Rust crate `sea-g2p`.
- **C Header:** `sea_g2p.h` providing:
  - `char* sea_g2p_normalize_punc(const char* input)`
  - `char* sea_g2p_phonemize(const char* input)`
  - `void sea_g2p_free_string(char* s)`
- **Compilation Tooling:** Build script `ios/scripts/build_seag2p.sh` compiling targets:
  - `aarch64-apple-ios` (Physical devices)
  - `aarch64-apple-ios-sim` (Apple Silicon Simulator)
  - `x86_64-apple-ios-sim` (Intel Mac Simulator)
- Bundled into a multi-architecture `SeaG2P.xcframework` linked as a binary target in `Package.swift`.

### Component 2: `ONNXEngine` & Models
- Utilizes official `microsoft/onnxruntime-swift-package` (version 1.20+).
- **Core Synthesis Graphs:**
  1. `text_encoder.onnx`: Embeds token IDs and speaker representations into text hidden states.
  2. `duration_predictor.onnx`: Predicts frame counts per phoneme.
  3. `vector_estimator.onnx`: Evaluates the vector field during Euler numerical integration.
  4. `codec_decoder.onnx`: Decodes latent vectors into raw 24 kHz PCM waveforms.
- **Cloning Graphs (Loaded on Demand):**
  1. `speaker_encoder.onnx`: Extracts a 192-dimensional x-vector from 80-channel log-mel filterbanks.
  2. `codec_encoder.onnx`: Converts reference audio frames into acoustic codes.
  3. `reference_encoder.onnx`: Derives a `[50, 256]` style token matrix from reference codes.

### Component 3: Flow-Matching Euler Solver
- Ported from `src/vieneu/v3nano.py`:
  - Uniform time stepping over interval $t \in [0, 1]$.
  - Default Euler steps: 16 (high quality); fast mode: 8 steps with `sway = -1.0`.
  - Classifier-Free Guidance (CFG): default scale 3.0.
  - Implemented with vector acceleration using Apple's `Accelerate` framework (`vDSP`).

### Component 4: Native Voice Cloning Preprocessor (`Accelerate/vDSP`)
- Decodes input audio using `AVAudioFile` / `AVAudioConverter` (supporting WAV, M4A, MP3, CAF) resampled to 24 kHz mono Float32.
- 80-bin Mel-filterbank spectrogram computation conforming to Kaldi fbank standards implemented natively in Swift using Apple's `vDSP` / `vForce`.
- Automatic silence trimming at boundaries based on energy threshold.

### Component 5: Model Management (`ModelProvider` & `ModelDownloader`)
- `protocol ModelProvider`:
  - `func modelURL(for filename: String) throws -> URL`
  - `func constantsURL() throws -> URL`
- Implementations:
  - `BundleModelProvider`: Reads files directly from any specified `Bundle` (e.g. `Bundle.main`).
  - `CachedModelProvider`: Manages models stored in `FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)`.
- `ModelDownloader`: Actor handling background session downloads from Hugging Face (`pnnbao-ump/VieNeu-TTS-v3-Nano`), checksum verification, and resume capabilities.

---

## 4. Public API Design

```swift
import VieNeuTTS

// 1. Initialization
let config = VieNeuConfig(
    modelProvider: try BundleModelProvider(bundle: .main), // or CachedModelProvider()
    eulerSteps: 16,
    cfgScale: 3.0
)
let tts = try await VieNeuTTS(config: config)

// 2. Synthesize using preset voice (Full audio)
let voice = try await tts.loadPresetVoice(name: "Minh Quân")
let audio = try await tts.synthesize(
    text: "Xin chào! Đây là VieNeu-TTS chạy trực tiếp trên iPhone.",
    voice: voice
)

// 3. Sentence-level Streaming for long paragraphs
for try await chunk in tts.synthesizeStream(text: longArticle, voice: voice) {
    chunk.audio.play()
}

// 4. Playback or Export
audio.play() // uses AVAudioPlayer under the hood
try audio.writeWAV(to: outputURL)

// 5. Voice Cloning (Enrollment from audio file or mic recording)
let customVoice = try await tts.cloneVoice(
    from: recordingWAVURL,
    name: "MyCustomVoice"
)
try customVoice.export(to: voiceProfileURL)
```

---

## 5. Repository Structure

```
VieNeu-TTS/
├── ios/
│   ├── VieNeuTTS/                     # Swift Package (SPM)
│   │   ├── Package.swift
│   │   ├── Sources/
│   │   │   └── VieNeuTTS/
│   │   │       ├── Core/              # Facade, Config, Error types
│   │   │       ├── Text/              # SeaG2P bridge, Tokenizer, Chunking
│   │   │       ├── Engine/            # ONNXEngine, FlowSolver, ModelProvider
│   │   │       ├── Audio/             # Accelerate vDSP fbank, AudioBuffer, WAV
│   │   │       └── Cloning/           # VoiceCloner, VoiceProfile
│   │   ├── Frameworks/
│   │   │   └── SeaG2P.xcframework    # Generated Rust C-ABI XCFramework
│   │   └── Tests/
│   │       └── VieNeuTTSTests/        # Unit & Integration tests
│   ├── scripts/
│   │   ├── build_seag2p.sh           # Cargo cross-compile & xcodebuild xcframework
│   │   └── download_nano_models.sh   # Downloads ONNX models for testing
│   └── examples/
│       └── VieNeuTTSDemo/             # SwiftUI sample application
```

---

## 6. Constraints & Non-Functional Requirements

- **Memory Budget:** Maximum RAM usage during synthesis $\le 250$ MB.
- **Latency / RTF:** Real-Time Factor (RTF) $\le 0.5$ on A15 Bionic / M1 or newer (1 second of audio takes $\le 0.5$ seconds to compute).
- **Threading:** All heavy work executed on detached background tasks via Swift concurrency actors.
- **Thread Safety:** ONNX sessions and ODE loops encapsulated within thread-safe actors.

---

## 7. Decisions Made by the Builder (Builder Calls)

1. **Repository Layout:** Located at `ios/VieNeuTTS` within this monorepo, maintaining close alignment with core Python algorithms and models.
2. **ONNX Runtime Distribution:** Consumed via the official `microsoft/onnxruntime-swift-package` SPM dependency.
3. **Streaming Paradigm:** Chunk/sentence-level streaming via `AsyncThrowingStream<AudioChunk, Error>` matching v3 Nano's non-autoregressive flow-matching nature.
4. **Input Audio Compatibility:** `AVAudioFile` decodes any Apple-supported format (WAV, M4A, MP3, CAF) and converts it to 24 kHz mono Float32 for voice cloning.
5. **Rust Tooling:** `ios/scripts/build_seag2p.sh` handles cross-compiling the `sea-g2p` crate and assembling `SeaG2P.xcframework`.
