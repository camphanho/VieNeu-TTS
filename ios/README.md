# VieNeuTTS for Apple Platforms (iOS & macOS)

`VieNeuTTS` is a production-grade native Swift Package delivering **100% on-device offline Vietnamese text-to-speech** for iOS 16.0+ and macOS 14.0+.

It supports both state-of-the-art architectures:
- **v3 Turbo (48 kHz)**: 12-layer Autoregressive Transformer + 16-RVQ local acoustic decoder + MOSS Audio Tokenizer. Runs via ONNX Runtime Mobile INT8 with autoreleased tensor reclamation, keeping peak RAM flat at ~300 MB (immune to iOS Jetsam OOM).
- **v3 Nano (24 kHz)**: 48M-parameter flow-matching model with Euler ODE solver and Classifier-Free Guidance (CFG).

---

## ✨ Features

- 📱 **100% Offline & On-Device**: Zero cloud dependencies or network telemetry after downloading weights.
- 🎙️ **48 kHz High Fidelity**: Crystal-clear studio-grade voice synthesis via MOSS Audio Tokenizer.
- 🇻🇳 **Native Vietnamese Phonemization**: Powered by Rust `sea-g2p` compiled into a native `SeaG2P.xcframework` with full diacritics and punctuation normalization.
- 🎭 **37+ Preset Voices**: Out-of-the-box support for beloved voices such as **Nguyễn Ngọc Ngạn**, **An Nhiên**, **Giang (Podcast)**, **Mai (Tự nhiên)**, **Ngân Nguyễn**, **Quang Tiến**, **Thu Thủy**, **Triệu Dương**, **Hải Đăng**, **Mai Anh**, **Trúc Ly**, and more.
- 🧬 **Instant Voice Cloning**: Enroll any voice on-device from a short reference audio clip ($\ge 1.0$s) with automatic silence trimming and acoustic feature extraction.
- ⚡ **Sentence-by-Sentence Streaming**: Low-latency `AsyncThrowingStream` emitting audio chunks as sentences finish generating.
- 📦 **Dual Model Loading**: Bundled directly in your app bundle or dynamically downloaded and cached in `CachesDirectory`.

---

## 📋 Requirements

- iOS 16.0+ / macOS 14.0+
- Xcode 15.0+ / Swift 5.9+
- Device: A12 Bionic or newer recommended for v3 Turbo

---

## 🚀 Quick Start (Swift)

### 1. Add Swift Package

Add `ios/VieNeuTTS` to your `Package.swift` or Xcode Project:

```swift
dependencies: [
    .package(path: "path/to/VieNeu-TTS/ios/VieNeuTTS")
]
```

### 2. Speech Synthesis

```swift
import VieNeuTTS

// 1. Download or load models
let downloader = ModelDownloader()
let provider = try await downloader.downloadTurbo { progress in
    print("Download progress: \(Int(progress * 100))%")
}

// 2. Initialize the synthesis engine (v3 Turbo 48 kHz)
let config = VieNeuConfig(
    mode: .v3turbo,
    modelProvider: provider,
    speed: 1.0
)
let tts = try await VieNeuTTS(config: config)

// 3. Load a preset voice (e.g. Nguyễn Ngọc Ngạn)
let voice = try tts.loadPresetVoice(name: "Nguyễn Ngọc Ngạn")

// 4. Synthesize text to 48 kHz mono AudioBuffer
let audio = try await tts.synthesize(
    text: "Anh em nào phải người xa, Cùng chung bác mẹ, một nhà cùng thân.",
    voice: voice
)

// 5. Play or save WAV
let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("output.wav")
try audio.writeWAV(to: fileURL)
```

### 3. Sentence-by-Sentence Streaming

Stream sentence audio as it is generated to minimize time-to-first-audio (TTFA):

```swift
let stream = tts.synthesizeStream(text: longText, voice: voice)

for try await chunk in stream {
    print("Received chunk #\(chunk.index): \(chunk.text)")
    // Play or buffer chunk.audio immediately
}
```

### 4. Instant Voice Cloning

```swift
// Enroll a new voice profile from a reference WAV/M4A clip
let referenceURL = URL(fileURLWithPath: "path/to/reference_voice.wav")
let customProfile = try await tts.cloneVoice(from: referenceURL, name: "Giọng Mẫu")

// Synthesize using the cloned voice
let clonedAudio = try await tts.synthesize(
    text: "Xin chào, đây là giọng nói được nhân bản trực tiếp trên điện thoại.",
    voice: customProfile
)
```

---

## 📱 SwiftUI Demo Application

A complete SwiftUI reference application is located in `ios/examples/VieNeuTTSDemo`:

- **Segmented Architecture Switcher**: Seamlessly toggle between **v3 Nano (24 kHz)** and **v3 Turbo (48 kHz)**.
- **Voice Library**: Select from 37+ system presets or record / import your own custom voices.
- **Microphone Enrollment**: Record voice samples with live countdown and waveform feedback.
- **Playback Telemetry**: Live RTF (Real-Time Factor), generation latency, and duration metrics.

### Building & Running on Physical Device

Using `xtool` over USB:
```bash
cd ios/examples/VieNeuTTSDemo
USE_XCFRAMEWORK=1 xtool dev build
rm -rf xtool/VieNeuTTSDemo.app/Frameworks && xtool install --usb xtool/VieNeuTTSDemo.app
```

---

## 🛠️ Building SeaG2P.xcframework from Scratch

`SeaG2P.xcframework` bridges the Rust G2P engine to Swift. To recompile on macOS:

```bash
bash ios/scripts/build_seag2p.sh
```

This cross-compiles `ios/SeaG2PBridge` for:
- `aarch64-apple-ios` (iOS Physical Devices)
- `aarch64-apple-ios-sim` (Apple Silicon Simulator)
- `x86_64-apple-ios-sim` (Intel Simulator)
and packages them into `ios/VieNeuTTS/Frameworks/SeaG2P.xcframework`.

---

## 🧪 Testing

Run all 112+ unit and integration tests across both architectures:

```bash
cd ios/VieNeuTTS
swift test
```
