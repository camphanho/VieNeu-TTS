# VieNeu-TTS v3 Nano iOS SDK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an on-device, offline iOS SDK (`VieNeuTTS`) for VieNeu-TTS v3 Nano (24 kHz) with instant voice cloning, preset voices, and dynamic model provisioning for iOS 16.0+.

**Architecture:** A Swift Package (`ios/VieNeuTTS`) embedding a cross-compiled Rust C-ABI XCFramework (`SeaG2P.xcframework`) for Vietnamese G2P phonemization, `onnxruntime-swift-package` for neural network evaluation, native Apple `Accelerate` (`vDSP`) for Mel-filterbank signal processing, and a flow-matching Euler ODE solver exposed through a modern async/await Swift API.

**Tech Stack:** Swift 5.9+, iOS 16.0+, Rust (sea-g2p), Apple Accelerate/vDSP, ONNX Runtime Mobile, AVFoundation, SwiftUI.

**Spec:** `docs/superpowers/specs/2026-11-10-ios-sdk-v3nano-design.md`

## Global Constraints

- Target OS: iOS 16.0+ (deployment target).
- Package format: Swift Package Manager (SPM) at `ios/VieNeuTTS`.
- G2P Frontend: Rust `sea-g2p` compiled into multi-architecture `SeaG2P.xcframework` (`aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios-sim`).
- Inference Engine: `microsoft/onnxruntime-swift-package` (version 1.20+).
- Audio format: 24,000 Hz, 1-channel mono, 32-bit float PCM / 16-bit WAV.
- Memory limit: Maximum RAM during synthesis $\le 250$ MB.
- Concurrency: All compute runs on detached background actors; UI thread is never blocked.

## Review Focus

1. **Empty or punctuation-only input text:** Passing `""`, `"   "`, or `"...!?"` throws `VieNeuError.emptyInput` gracefully instead of feeding empty tensors to the ODE solver.
2. **Reference audio shorter than 1.0s or pure silence during voice cloning:** Throws `VieNeuError.referenceAudioTooShortOrSilent` instead of generating NaN filterbanks.
3. **Corrupted or truncated ONNX model file in cache:** Checksum validation fails and raises `VieNeuError.modelCorrupted` instead of crashing inside ONNX Runtime.
4. **Multi-channel high-sample-rate audio input (e.g. 48kHz stereo mic input):** Automatically downmixed to mono and resampled to 24 kHz via `AVAudioConverter` before feature extraction.
5. **Mid-stream task cancellation:** Canceling a Task iterating `synthesizeStream` immediately halts Euler ODE execution and frees model session memory.

---

### Task 1: Rust C-ABI Bridge & XCFramework Build Script for SeaG2P

**Files:**
- Create: `ios/SeaG2PBridge/Cargo.toml`
- Create: `ios/SeaG2PBridge/src/lib.rs`
- Create: `ios/SeaG2PBridge/include/sea_g2p.h`
- Create: `ios/scripts/build_seag2p.sh`
- Test: `ios/SeaG2PBridge/tests/bridge_tests.rs`

**Interfaces:**
- Consumes: `sea-g2p` Rust crate.
- Produces: C-ABI dynamic/static functions exported in `sea_g2p.h`:
  - `char* sea_g2p_normalize_punc(const char* input)`
  - `char* sea_g2p_phonemize(const char* input)`
  - `void sea_g2p_free_string(char* s)`

- [ ] **Step 1: Write the Rust C-ABI test**

```rust
// ios/SeaG2PBridge/tests/bridge_tests.rs
use std::ffi::{CStr, CString};
use sea_g2p_bridge::*;

#[test]
fn test_c_abi_normalization_and_phonemization() {
    let input = CString::new("Xin chào Việt Nam!").unwrap();
    let norm_ptr = sea_g2p_normalize_punc(input.as_ptr());
    assert!(!norm_ptr.is_null());
    let norm_str = unsafe { CStr::from_ptr(norm_ptr).to_str().unwrap() };
    assert!(norm_str.contains("Xin chào Việt Nam !"));
    sea_g2p_free_string(norm_ptr);

    let phone_ptr = sea_g2p_phonemize(input.as_ptr());
    assert!(!phone_ptr.is_null());
    let phone_str = unsafe { CStr::from_ptr(phone_ptr).to_str().unwrap() };
    assert!(!phone_str.is_empty());
    sea_g2p_free_string(phone_ptr);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path ios/SeaG2PBridge/Cargo.toml`
Expected: FAIL with missing package or unresolved symbols.

- [ ] **Step 3: Implement C-ABI wrapper & XCFramework build script**

In `ios/SeaG2PBridge/src/lib.rs`, wrap `sea_g2p::Normalizer` and `sea_g2p::phonemize` returning `CString::into_raw`. Provide `sea_g2p_free_string` using `CString::from_raw`.
In `ios/scripts/build_seag2p.sh`, cross-compile for `aarch64-apple-ios`, `aarch64-apple-ios-sim`, and `x86_64-apple-ios-sim`, lipo the simulator targets, and package into `ios/VieNeuTTS/Frameworks/SeaG2P.xcframework` via `xcodebuild -create-xcframework`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path ios/SeaG2PBridge/Cargo.toml`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/SeaG2PBridge ios/scripts/build_seag2p.sh
git commit -m "feat(ios): add sea-g2p C-ABI bridge and xcframework build script"
```

---

### Task 2: Swift Package Scaffolding & SeaG2P Swift Wrapper

**Files:**
- Create: `ios/VieNeuTTS/Package.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Text/SeaG2P.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/SeaG2PTests.swift`

**Interfaces:**
- Consumes: `SeaG2P.xcframework` (`sea_g2p.h`).
- Produces:
  - `public final class SeaG2P: Sendable`
  - `public func normalizePunctuation(_ text: String) throws -> String`
  - `public func phonemize(_ text: String) throws -> String`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/SeaG2PTests.swift
import XCTest
@testable import VieNeuTTS

final class SeaG2PTests: XCTestCase {
    func testNormalizationAndPhonemize() throws {
        let g2p = SeaG2P()
        let normalized = try g2p.normalizePunctuation("Hà Nội, thủ đô của Việt Nam.")
        XCTAssertTrue(normalized.contains(","))
        let phonemes = try g2p.phonemize(normalized)
        XCTAssertFalse(phonemes.isEmpty)
    }

    func testEmptyStringHandledGracefully() throws {
        let g2p = SeaG2P()
        let normalized = try g2p.normalizePunctuation("")
        XCTAssertEqual(normalized, "")
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter SeaG2PTests`
Expected: FAIL with compilation error (module not found or SeaG2P undefined).

- [ ] **Step 3: Implement `Package.swift` and `SeaG2P.swift`**

In `Package.swift`: define targets for `SeaG2P` binaryTarget and `VieNeuTTS` library target adding `microsoft/onnxruntime-swift-package`.
In `SeaG2P.swift`: import `SeaG2P` C-module, call `sea_g2p_normalize_punc` and `sea_g2p_phonemize` converting Swift `String` to UTF8 C strings, free memory via `defer { sea_g2p_free_string(ptr) }`.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter SeaG2PTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Package.swift ios/VieNeuTTS/Sources/VieNeuTTS/Text/SeaG2P.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/SeaG2PTests.swift
git commit -m "feat(ios): add Package.swift and SeaG2P Swift wrapper"
```

---

### Task 3: Phoneme Tokenizer & Sentence Chunker

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Text/Tokenizer.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Text/Chunker.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/TokenizerTests.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/ChunkerTests.swift`

**Interfaces:**
- Consumes: Phoneme vocabulary map from v3 Nano configuration.
- Produces:
  - `public final class PhonemeTokenizer: Sendable`
  - `public func encode(phonemes: String) -> [Int64]`
  - `public struct SentenceChunk: Equatable, Sendable { public let text: String; public let pauseDuration: Float }`
  - `public struct Chunker { public static func splitToChunks(_ text: String, maxChars: Int = 200) -> [SentenceChunk] }`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/ChunkerTests.swift
import XCTest
@testable import VieNeuTTS

final class ChunkerTests: XCTestCase {
    func testSplitsLongParagraphIntoSentences() {
        let text = "Hôm nay trời rất đẹp. Tôi đi dạo quanh bờ hồ! Bạn có muốn đi cùng không?"
        let chunks = Chunker.splitToChunks(text, maxChars: 50)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks[0].text.contains("đẹp"))
    }

    func testEmptyStringReturnsEmptyChunks() {
        let chunks = Chunker.splitToChunks("   ", maxChars: 50)
        XCTAssertTrue(chunks.isEmpty)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter ChunkerTests`
Expected: FAIL with "Chunker undefined".

- [ ] **Step 3: Implement Tokenizer & Chunker**

Implement `Tokenizer` mapping phonemes and emotion tags (`[cười]`, `[thở dài]`) to token IDs matching Python `tokenizers` config.
Implement `Chunker` splitting text on sentence boundaries (`.`, `!`, `?`, `;`, newline) while preserving natural pause durations (0.15s – 0.3s).

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --package-path ios/VieNeuTTS --filter "TokenizerTests|ChunkerTests"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Text ios/VieNeuTTS/Tests/VieNeuTTSTests
git commit -m "feat(ios): implement phoneme tokenizer and sentence chunker"
```

---

### Task 4: Audio Engine & Native vDSP Mel-Filterbank Preprocessor

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Audio/AudioBuffer.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Audio/WAVWriter.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Audio/MelFilterbank.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/AudioBufferTests.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/MelFilterbankTests.swift`

**Interfaces:**
- Consumes: Apple `Accelerate` (`vDSP`), `AVFoundation`.
- Produces:
  - `public struct AudioBuffer: Sendable` (`samples: [Float]`, `sampleRate: Int`, `duration: TimeInterval`, `func makeAVAudioPCMBuffer() -> AVAudioPCMBuffer`, `func writeWAV(to url: URL) throws`)
  - `public final class MelFilterbank: Sendable`
  - `public func compute(samples: [Float], sampleRate: Int) throws -> [[Float]]` (80-channel log-mel filterbanks)
  - `public static func convertToMono24kHz(from url: URL) throws -> [Float]`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/MelFilterbankTests.swift
import XCTest
@testable import VieNeuTTS

final class MelFilterbankTests: XCTestCase {
    func testComputes80BinMelFilterbank() throws {
        let fbank = MelFilterbank()
        let sampleCount = 24000 // 1 second of audio
        let sineWave = (0..<sampleCount).map { sin(2.0 * Float.pi * 440.0 * Float($0) / 24000.0) }
        let mel = try fbank.compute(samples: sineWave, sampleRate: 24000)
        XCTAssertFalse(mel.isEmpty)
        XCTAssertEqual(mel[0].count, 80) // 80 filter channels
    }

    func testDownmixStereoAndResample() throws {
        // Review Focus #4: Stereo high-sample-rate audio resampled cleanly to 24kHz mono Float32
        let dummyWavURL = try makeTestStereoWav(sampleRate: 48000, duration: 1.0)
        defer { try? FileManager.default.removeItem(at: dummyWavURL) }
        let mono24k = try MelFilterbank.convertToMono24kHz(from: dummyWavURL)
        XCTAssertEqual(mono24k.count, 24000, accuracy: 240)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter MelFilterbankTests`
Expected: FAIL with "MelFilterbank undefined".

- [ ] **Step 3: Implement MelFilterbank with vDSP and WAVWriter**

Implement FFT windowing (Povey / Hanning window), power spectrum via `vDSP_zvabs`, and triangular filterbank dot products matching Kaldi fbank parameters (frame_shift=10ms, frame_length=25ms, num_mel_bins=80).
Implement `convertToMono24kHz` using `AVAudioFile` and `AVAudioConverter`.
Implement `WAVWriter` writing 16-bit PCM RIFF headers and clamped sample data.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter "MelFilterbankTests|AudioBufferTests"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Audio ios/VieNeuTTS/Tests/VieNeuTTSTests
git commit -m "feat(ios): add audio buffer, wav writer, and vDSP mel-filterbank"
```

---

### Task 5: Model Providers & Background Downloader

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ModelProvider.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ModelDownloader.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/ModelProviderTests.swift`

**Interfaces:**
- Consumes: `FileManager`, `URLSession`.
- Produces:
  - `public protocol ModelProvider: Sendable { func modelURL(for filename: String) throws -> URL; func constantsURL() throws -> URL }`
  - `public final class BundleModelProvider: ModelProvider`
  - `public final class CachedModelProvider: ModelProvider`
  - `public actor ModelDownloader { public func download(progress: @Sendable (Double) -> Void) async throws -> CachedModelProvider }`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/ModelProviderTests.swift
import XCTest
@testable import VieNeuTTS

final class ModelProviderTests: XCTestCase {
    func testBundleModelProviderThrowsOnMissingModel() {
        let provider = BundleModelProvider(bundle: .main)
        XCTAssertThrowsError(try provider.modelURL(for: "non_existent.onnx")) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound = vieNeuError else {
                XCTFail("Expected .modelNotFound error")
                return
            }
        }
    }

    func testCorruptedModelDetectionThrows() throws {
        // Review Focus #3: Corrupted model file triggers .modelCorrupted error
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let corruptedFile = tempDir.appendingPathComponent("text_encoder.onnx")
        try "corrupted data".write(to: corruptedFile, atomically: true, encoding: .utf8)

        let provider = CachedModelProvider(directory: tempDir, expectedChecksums: ["text_encoder.onnx": "valid_sha256_hash"])
        XCTAssertThrowsError(try provider.validateChecksums()) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelCorrupted = vieNeuError else {
                XCTFail("Expected .modelCorrupted error")
                return
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter ModelProviderTests`
Expected: FAIL with "ModelProvider undefined".

- [ ] **Step 3: Implement ModelProvider and ModelDownloader**

Implement `BundleModelProvider` and `CachedModelProvider` conforming to `ModelProvider`.
Implement SHA-256 validation using `CryptoKit`.
Implement `ModelDownloader` downloading v3 Nano assets (`text_encoder.onnx`, `duration_predictor.onnx`, `vector_estimator.onnx`, `codec_decoder.onnx`, `config.json`, `constants.npz`) from Hugging Face with progress callbacks.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter ModelProviderTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Engine/Model* ios/VieNeuTTS/Tests/VieNeuTTSTests/ModelProviderTests.swift
git commit -m "feat(ios): implement model providers and background downloader"
```

---

### Task 6: ONNX Runtime Engine & Euler ODE Flow Solver

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ONNXEngine.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Engine/FlowSolver.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/FlowSolverTests.swift`

**Interfaces:**
- Consumes: `onnxruntime-swift-package`, `Accelerate`.
- Produces:
  - `public actor ONNXEngine`
  - `func runTextEncoder(tokenIDs: [Int64], speakerEmbedding: [Float]) throws -> (mu: [[Float]], mask: [Bool])`
  - `func runDurationPredictor(mu: [[Float]]) throws -> [Int]`
  - `func runVectorEstimator(x: [[Float]], t: Float, cond: [[Float]], style: [[Float]]) throws -> [[Float]]`
  - `func runCodecDecoder(latents: [[Float]]) throws -> [Float]`
  - `public struct FlowSolver { public static func solveEuler(...) throws -> [[Float]] }`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/FlowSolverTests.swift
import XCTest
@testable import VieNeuTTS

final class FlowSolverTests: XCTestCase {
    func testEulerIntegrationStepsShapeAndValues() throws {
        let steps = 4
        let dim = 8
        let frames = 10
        let initialNoise = (0..<frames).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        // Test with mock vector field: dX/dt = -X
        let trajectory = try FlowSolver.solveEuler(
            initialLatents: initialNoise,
            steps: steps,
            cfgScale: 1.0,
            vectorEstimator: { x, t in x.map { row in row.map { -$0 } } }
        )
        XCTAssertEqual(trajectory.count, frames)
        XCTAssertEqual(trajectory[0].count, dim)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter FlowSolverTests`
Expected: FAIL with "FlowSolver undefined".

- [ ] **Step 3: Implement FlowSolver and ONNXEngine**

In `FlowSolver.swift`: Implement Euler ODE stepper with uniform time grid `t ∈ [0, 1]`, sway schedule (`sway = -1.0` or standard), and Classifier-Free Guidance extrapolation `v = v_uncond + cfg * (v_cond - v_uncond)`.
In `ONNXEngine.swift`: Initialize `ORTEnv` and `ORTSession` instances with CPU execution provider and intra-op thread tuning. Implement tensor conversions between Swift `[Float]` / `[Int64]` and `ORTValue`.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter FlowSolverTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Engine/FlowSolver.swift ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ONNXEngine.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/FlowSolverTests.swift
git commit -m "feat(ios): implement onnx engine and euler ode flow solver"
```

---

### Task 7: Voice Profiles & Cloning Pipeline

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Cloning/VoiceProfile.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Cloning/VoiceCloner.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/VoiceClonerTests.swift`

**Interfaces:**
- Consumes: `MelFilterbank`, `ONNXEngine`.
- Produces:
  - `public struct VoiceProfile: Codable, Sendable`
    - `public let name: String`
    - `public let speakerEmbedding: [Float]` (192-d x-vector)
    - `public let styleTokens: [[Float]]` (50 x 256 matrix)
    - `public func export(to url: URL) throws`
    - `public static func load(from url: URL) throws -> VoiceProfile`
  - `public actor VoiceCloner`
    - `public func clone(from audioURL: URL, name: String) async throws -> VoiceProfile`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/VoiceClonerTests.swift
import XCTest
@testable import VieNeuTTS

final class VoiceClonerTests: XCTestCase {
    func testVoiceProfileSerialization() throws {
        let profile = VoiceProfile(
            name: "TestSpeaker",
            speakerEmbedding: Array(repeating: 0.1, count: 192),
            styleTokens: Array(repeating: Array(repeating: 0.2, count: 256), count: 50)
        )
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("profile.json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try profile.export(to: tempURL)
        let loaded = try VoiceProfile.load(from: tempURL)
        XCTAssertEqual(loaded.name, "TestSpeaker")
        XCTAssertEqual(loaded.speakerEmbedding.count, 192)
        XCTAssertEqual(loaded.styleTokens.count, 50)
    }

    func testTooShortOrSilentAudioThrows() async throws {
        // Review Focus #2: Reference audio < 1s or silent throws referenceAudioTooShortOrSilent
        let silentURL = try makeSilentWav(duration: 0.5)
        defer { try? FileManager.default.removeItem(at: silentURL) }

        let cloner = VoiceCloner(onnxEngine: MockONNXEngine())
        do {
            _ = try await cloner.clone(from: silentURL, name: "BadClip")
            XCTFail("Expected error on silent short audio")
        } catch let error as VieNeuError {
            guard case .referenceAudioTooShortOrSilent = error else {
                XCTFail("Expected .referenceAudioTooShortOrSilent error")
                return
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter VoiceClonerTests`
Expected: FAIL with "VoiceProfile undefined".

- [ ] **Step 3: Implement VoiceProfile and VoiceCloner**

In `VoiceProfile.swift`: Codable JSON representation of speaker embedding and style codes.
In `VoiceCloner.swift`: Trim silence from reference audio using RMS threshold. Validate duration $\ge 1.0$s and energy $> 10^{-4}$. Extract 80-bin mel-filterbank, pass through `speaker_encoder.onnx` to produce 192-d x-vector, run `codec_encoder.onnx` and `reference_encoder.onnx` to produce style tokens `[50, 256]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter VoiceClonerTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Cloning ios/VieNeuTTS/Tests/VieNeuTTSTests/VoiceClonerTests.swift
git commit -m "feat(ios): implement voice profiles and voice cloning pipeline"
```

---

### Task 8: VieNeuTTS Facade & Sentence-level Streaming API

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuConfig.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuError.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuTTS.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/VieNeuTTSSynthesisTests.swift`

**Interfaces:**
- Consumes: All earlier components (`SeaG2P`, `Tokenizer`, `Chunker`, `ONNXEngine`, `FlowSolver`, `VoiceCloner`, `AudioBuffer`).
- Produces:
  - `public actor VieNeuTTS`
  - `public init(config: VieNeuConfig) async throws`
  - `public func loadPresetVoice(name: String) throws -> VoiceProfile`
  - `public func synthesize(text: String, voice: VoiceProfile) async throws -> AudioBuffer`
  - `public func synthesizeStream(text: String, voice: VoiceProfile) -> AsyncThrowingStream<AudioChunk, Error>`
  - `public func cloneVoice(from audioURL: URL, name: String) async throws -> VoiceProfile`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/VieNeuTTSSynthesisTests.swift
import XCTest
@testable import VieNeuTTS

final class VieNeuTTSSynthesisTests: XCTestCase {
    func testEmptyTextInputThrows() async throws {
        // Review Focus #1: Empty or whitespace input throws emptyInput error
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        do {
            _ = try await tts.synthesize(text: "    ", voice: voice)
            XCTFail("Expected emptyInput error")
        } catch let error as VieNeuError {
            guard case .emptyInput = error else {
                XCTFail("Expected .emptyInput error")
                return
            }
        }
    }

    func testCancellationStopsStreamExecution() async throws {
        // Review Focus #5: Mid-stream task cancellation immediately stops computation
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        let task = Task {
            for try await _ in tts.synthesizeStream(text: "Một đoạn văn dài gồm rất nhiều câu.", voice: voice) {
                // cancel immediately
                throw CancellationError()
            }
        }
        _ = await task.result
        XCTAssertTrue(task.isCancelled)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter VieNeuTTSSynthesisTests`
Expected: FAIL with "VieNeuTTS undefined".

- [ ] **Step 3: Implement VieNeuTTS Facade**

Tie together the pipeline:
1. Validate text $\to$ normalize punctuation via `SeaG2P` $\to$ chunk text via `Chunker`.
2. For each chunk: convert text to phonemes $\to$ tokenize to IDs.
3. Call `ONNXEngine` text encoder + duration predictor.
4. Integrate latent flow via `FlowSolver.solveEuler`. Check `Task.isCancelled` each step.
5. Decode to 24 kHz waveform with codec decoder.
6. Yield chunks in `synthesizeStream` and concatenate with silence gaps in `synthesize`.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter VieNeuTTSSynthesisTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Core ios/VieNeuTTS/Tests/VieNeuTTSTests/VieNeuTTSSynthesisTests.swift
git commit -m "feat(ios): implement VieNeuTTS facade and streaming synthesis API"
```

---

### Task 9: SwiftUI Demo Application (`VieNeuTTSDemo`)

**Files:**
- Create: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/App.swift`
- Create: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/Views/ContentView.swift`
- Create: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/Views/SynthesisView.swift`
- Create: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/Views/VoiceCloningView.swift`
- Create: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/ViewModels/TTSViewModel.swift`

**Interfaces:**
- Consumes: `VieNeuTTS` package.
- Produces: Working iOS demo app with tabs:
  - Tab 1: Text synthesis with preset voice selection, playback controls, and real-time RTF / duration display.
  - Tab 2: Instant voice cloning with live microphone recording, waveform preview, enrollment, and profile export.

- [ ] **Step 1: Write SwiftUI App entrypoint and ViewModel**

Implement `TTSViewModel` managing `VieNeuTTS` lifecycle, model download progress, audio synthesis state, and recording buffer capture via `AVAudioRecorder`.

- [ ] **Step 2: Implement UI views in SwiftUI**

`SynthesisView`: Text editor with sample Vietnamese sentences, voice picker (`Minh Quân`, `Mai Anh`), speed/step slider, "Đọc văn bản" button, audio playback visualizer.
`VoiceCloningView`: Record button (3–8s countdown), record playback, "Sao chép giọng" button, save/load custom voice profiles.

- [ ] **Step 3: Verify build via xcodebuild or swift build**

Run: `swift build --package-path ios/VieNeuTTS`
Expected: Build succeeds without errors.

- [ ] **Step 4: Commit**

```bash
git add ios/examples/VieNeuTTSDemo
git commit -m "feat(ios): add SwiftUI sample application demonstrating speech synthesis and voice cloning"
```

---

## Plan Review & Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-10-ios-sdk-v3nano.md`.

Which execution approach would you prefer?

- **Subagent-driven** (Recommended) — A fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; catches interface and type mismatches early.
- **Native** — I implement every task myself in this session, the way this harness runs work, then one fresh reviewer on the most capable model checks the whole branch. Faster, single thread.

Does the plan capture what you want, and which approach should we use?
