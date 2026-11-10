# VieNeu-TTS v3 Turbo (48 kHz) On-Device iOS SDK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement completely on-device, offline inference for VieNeu-TTS v3 Turbo (48 kHz studio audio) using quantized INT8 ONNX models and MOSS Audio Tokenizer on iOS 16.0+.

**Architecture:** An autoregressive 12-layer transformer engine (`V3TurboEngine`) managing KV-caching across `vieneu_prefill.onnx` and `vieneu_decode_step.onnx`, local 2-layer acoustic generation via `vieneu_acoustic_cached.onnx`, native Apple `Accelerate` for speaker embedding projection and Top-K/Top-P sampling, and MOSS Codec decoding yielding 48,000 Hz audio buffers.

**Tech Stack:** Swift 5.9+, iOS 16.0+, ONNX Runtime Mobile (`OnnxRuntimeBindings`), Apple Accelerate (`vDSP`), AVFoundation, SwiftUI.

**Spec:** `docs/superpowers/specs/2026-11-11-ios-sdk-v3turbo-design.md`

## Global Constraints

- Target OS: iOS 16.0+ deployment target.
- Output Sample Rate: 48,000 Hz mono PCM for v3 Turbo (vs 24,000 Hz for v3 Nano).
- Models: v3 Turbo INT8 package from `pnnbao-ump/VieNeu-TTS-v3-Turbo/onnx_int8` and MOSS Codec from `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`.
- Memory Limit: Total runtime memory $\le 350$ MB.
- Concurrency: Background actor isolation; UI thread never blocked.

## Review Focus

1. **Speaker Anchor Projection Accuracy:** Projecting a 192-d x-vector via `xvec_w [768, 192]` + LayerNorm produces exact 768-d anchor matching Python `_speaker_anchor`.
2. **Repetition Penalty & Top-K Sampling:** Temperature scaling, repetition penalty over history window, and Top-K pruning select valid token IDs in `[0, 1024)` without NaN or out-of-bounds index crashes.
3. **KV-Cache Tensor Shuffling:** 12 layers of key-value caches (`past_k_0..11`, `past_v_0..11`) are preserved and fed sequentially into `vieneu_decode_step.onnx` without memory leaks.
4. **EOS Token Termination:** Text logits reaching `speech_generation_end_token_id` terminates generation cleanly without producing trailing noise.
5. **Mode Switching in Facade:** `VieNeuTTS(config:)` with `mode: .v3turbo` produces 48,000 Hz audio while `mode: .v3nano` continues to produce 24,000 Hz audio.

---

### Task 1: V3 Turbo Head Weights & Speaker Anchor Projection

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboWeights.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboWeightsTests.swift`

**Interfaces:**
- Consumes: `vieneu_v3_heads` data (raw floats or binary NPZ).
- Produces:
  - `public final class V3TurboWeights: Sendable`
  - `public func textEmbedding(for tokenID: Int) -> [Float]` (768 floats)
  - `public func audioEmbedding(codebook: Int, tokenID: Int) -> [Float]` (768 floats)
  - `public func speakerAnchor(for speakerEmb: [Float]) throws -> [Float]` (192-d $\to$ 768-d via Linear + LayerNorm)

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboWeightsTests.swift
import XCTest
@testable import VieNeuTTS

final class V3TurboWeightsTests: XCTestCase {
    func testSpeakerAnchorProjectionShape() throws {
        let weights = V3TurboWeights.mock()
        let spk192 = Array(repeating: Float(0.5), count: 192)
        let anchor = try weights.speakerAnchor(for: spk192)
        XCTAssertEqual(anchor.count, 768)
    }

    func testZeroSpeakerEmbeddingThrows() {
        let weights = V3TurboWeights.mock()
        let zeros = Array(repeating: Float(0.0), count: 192)
        XCTAssertThrowsError(try weights.speakerAnchor(for: zeros))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboWeightsTests`
Expected: FAIL with "V3TurboWeights undefined".

- [ ] **Step 3: Implement V3TurboWeights**

Implement `V3TurboWeights` with memory buffers for text/audio embeddings and Linear + LayerNorm projection using `vDSP_mmul` and `vDSP_normalize`. Provide `mock()` with deterministic weights.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboWeightsTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboWeights.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboWeightsTests.swift
git commit -m "feat(ios): implement v3 turbo weights container and speaker anchor projection"
```

---

### Task 2: Sampling Pipeline (Top-K, Top-P, Repetition Penalty)

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboSampler.swift`
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/RepetitionHistory.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboSamplerTests.swift`

**Interfaces:**
- Consumes: Logits vector `[Float]`, temperature, topK, topP, repetition penalty.
- Produces:
  - `public struct V3TurboSampler: Sendable`
  - `public static func sample(logits: [Float], temperature: Float, topK: Int, topP: Float, repetitionPenalty: Float, history: [Int]?) -> Int`
  - `public final class RepetitionHistory: Sendable` (`add(token:)`, `recentTokens() -> [Int]`)

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboSamplerTests.swift
import XCTest
@testable import VieNeuTTS

final class V3TurboSamplerTests: XCTestCase {
    func testArgmaxWhenTemperatureZero() {
        let logits: [Float] = [1.0, 5.0, 2.0, 0.5]
        let token = V3TurboSampler.sample(logits: logits, temperature: 0.0, topK: 0, topP: 1.0, repetitionPenalty: 1.0, history: nil)
        XCTAssertEqual(token, 1)
    }

    func testRepetitionPenaltySuppressesSeenTokens() {
        let logits: [Float] = [5.0, 4.9, 1.0]
        let token = V3TurboSampler.sample(logits: logits, temperature: 0.0, topK: 0, topP: 1.0, repetitionPenalty: 2.0, history: [0])
        XCTAssertEqual(token, 1)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboSamplerTests`
Expected: FAIL with "V3TurboSampler undefined".

- [ ] **Step 3: Implement V3TurboSampler & RepetitionHistory**

Implement repetition penalty scaling, Top-K candidate gathering, Accelerate softmax, and Top-P cumulative thresholding matching Python `_sample`.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboSamplerTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboSampler.swift ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/RepetitionHistory.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboSamplerTests.swift
git commit -m "feat(ios): implement v3 turbo sampling and repetition penalty tracker"
```

---

### Task 3: V3 Turbo Model Downloader & Provisioning

**Files:**
- Modify: `ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ModelDownloader.swift`
- Modify: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuConfig.swift`
- Test: `ios/VieNeuTTS/Tests/VieNeuTTSTests/ModelProviderTests.swift`

**Interfaces:**
- Consumes: Hugging Face repos `pnnbao-ump/VieNeu-TTS-v3-Turbo` (subfolder `onnx_int8`) and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`.
- Produces:
  - `public enum VieNeuModelMode: String, Codable, Sendable { case v3nano, v3turbo }`
  - `ModelDownloader.downloadTurboModels(...)`

- [ ] **Step 1: Write the failing test**

```swift
// In ModelProviderTests.swift
func testTurboRequiredFilesConfiguration() {
    XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("vieneu_prefill.onnx"))
    XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("vieneu_decode_step.onnx"))
    XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("moss_audio_tokenizer_decode_full.onnx"))
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter testTurboRequiredFilesConfiguration`
Expected: FAIL with "turboRequiredFiles undefined".

- [ ] **Step 3: Update ModelDownloader & VieNeuConfig**

Add `VieNeuModelMode` enum. Configure download URLs for v3 Turbo INT8 files and MOSS Codec files.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter testTurboRequiredFilesConfiguration`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Engine/ModelDownloader.swift ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuConfig.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/ModelProviderTests.swift
git commit -m "feat(ios): support v3 turbo INT8 model downloading and mode configuration"
```

---

### Task 4: Autoregressive V3TurboEngine

**Files:**
- Create: `ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboEngine.swift`
- Create: `ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboEngineTests.swift`

**Interfaces:**
- Consumes: `V3TurboWeights`, `V3TurboSampler`, `OnnxRuntimeBindings`.
- Produces:
  - `public actor V3TurboEngine`
  - `public func generate(phonemes: String, speakerEmbedding: [Float], refCodes: [[Int64]]?, temperature: Float, topK: Int, topP: Float, maxNewFrames: Int) async throws -> [Float]` (48 kHz samples)
  - `public func generateStream(...) -> AsyncThrowingStream<[Float], Error>`

- [ ] **Step 1: Write the failing test**

```swift
// ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboEngineTests.swift
import XCTest
@testable import VieNeuTTS

final class V3TurboEngineTests: XCTestCase {
    func testMockV3TurboEngineGenerates48kHzAudio() async throws {
        let engine = MockV3TurboEngine()
        let spk = Array(repeating: Float(0.1), count: 192)
        let samples = try await engine.generate(phonemes: "sˈin tʃˈaː2w", speakerEmbedding: spk)
        XCTAssertFalse(samples.isEmpty)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboEngineTests`
Expected: FAIL with "V3TurboEngine undefined".

- [ ] **Step 3: Implement V3TurboEngine**

Implement prompt embeddings, `vieneu_prefill` invocation, the 16-codebook local acoustic sampling loop, `vieneu_decode_step` KV cache updates, and MOSS codec decoding to 48 kHz.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter V3TurboEngineTests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Turbo/V3TurboEngine.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/V3TurboEngineTests.swift
git commit -m "feat(ios): implement autoregressive v3 turbo inference engine"
```

---

### Task 5: VieNeuTTS Facade Unified Support & Streaming Integration

**Files:**
- Modify: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuTTS.swift`
- Modify: `ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuConfig.swift`
- Modify: `ios/VieNeuTTS/Tests/VieNeuTTSTests/VieNeuTTSSynthesisTests.swift`

**Interfaces:**
- Consumes: `V3TurboEngine` and `ONNXEngine` (Nano).
- Produces:
  - Unified `synthesize` and `synthesizeStream` returning `AudioBuffer` with proper sample rate (24,000 Hz for Nano, 48,000 Hz for Turbo).

- [ ] **Step 1: Write the failing test**

```swift
func testV3TurboModeReports48kHzSampleRate() async throws {
    let config = VieNeuConfig.mock(mode: .v3turbo)
    let tts = try await VieNeuTTS(config: config)
    let audio = try await tts.synthesize(text: "Xin chào", voice: VoiceProfile.mock())
    XCTAssertEqual(audio.sampleRate, 48000)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --package-path ios/VieNeuTTS --filter testV3TurboModeReports48kHzSampleRate`
Expected: FAIL with sample rate mismatch (24000 != 48000).

- [ ] **Step 3: Implement Mode Dispatching in VieNeuTTS**

Dispatch to `V3TurboEngine` when `config.mode == .v3turbo`. Yield 48 kHz audio buffers.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --package-path ios/VieNeuTTS --filter testV3TurboModeReports48kHzSampleRate`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuTTS.swift ios/VieNeuTTS/Sources/VieNeuTTS/Core/VieNeuConfig.swift ios/VieNeuTTS/Tests/VieNeuTTSTests/VieNeuTTSSynthesisTests.swift
git commit -m "feat(ios): integrate v3 turbo mode into VieNeuTTS facade with 48 kHz audio output"
```

---

### Task 6: SwiftUI Demo Engine Picker & On-Device Deployment

**Files:**
- Modify: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/ViewModels/TTSViewModel.swift`
- Modify: `ios/examples/VieNeuTTSDemo/VieNeuTTSDemo/Views/SynthesisView.swift`

**Interfaces:**
- Consumes: `VieNeuConfig.mode`.
- Produces: Engine selection picker in UI ("v3 Nano (24 kHz)" vs "v3 Turbo (48 kHz)").

- [ ] **Step 1: Update ViewModel and View**

Add `selectedMode: VieNeuModelMode` picker in SwiftUI header. Dynamically load/switch between Nano and Turbo.

- [ ] **Step 2: Verify Build and Deploy to iPhone**

Run: `cd ios/examples/VieNeuTTSDemo && USE_XCFRAMEWORK=1 xtool dev build`
Run: `xtool install --usb xtool/VieNeuTTSDemo.app`

- [ ] **Step 3: Commit**

```bash
git add ios/examples/VieNeuTTSDemo
git commit -m "feat(ios): add engine selector for v3 Nano and v3 Turbo in demo app"
```
