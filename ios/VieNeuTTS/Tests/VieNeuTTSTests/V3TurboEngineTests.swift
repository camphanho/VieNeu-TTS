import XCTest
@testable import VieNeuTTS

final class V3TurboEngineTests: XCTestCase {
    func testMockV3TurboEngineGenerates48kHzAudio() async throws {
        let engine = MockV3TurboEngine()
        let spk = Array(repeating: Float(0.1), count: 192)
        let samples = try await engine.generate(
            phonemes: "sˈin tʃˈaː2w",
            speakerEmbedding: spk,
            refCodes: nil,
            temperature: 0.8,
            topK: 25,
            topP: 0.95,
            maxNewFrames: 10
        )
        XCTAssertFalse(samples.isEmpty)
        XCTAssertEqual(samples.count % 960, 0)
    }

    func testV3TurboEngineStreamingYieldsChunks() async throws {
        let engine = MockV3TurboEngine()
        let spk = Array(repeating: Float(0.1), count: 192)
        var receivedChunks = 0
        for try await chunk in engine.generateStream(
            phonemes: "sˈin tʃˈaː2w",
            speakerEmbedding: spk,
            refCodes: nil,
            temperature: 0.8,
            topK: 25,
            topP: 0.95,
            maxNewFrames: 5
        ) {
            receivedChunks += 1
            XCTAssertFalse(chunk.isEmpty)
            XCTAssertEqual(chunk.count, 960) // 1 frame @ 48 kHz = 960 samples
        }
        XCTAssertGreaterThan(receivedChunks, 0)
    }

    func testCancellationHaltsTurboGeneration() async throws {
        let engine = MockV3TurboEngine()
        let spk = Array(repeating: Float(0.1), count: 192)
        let task = Task {
            for try await _ in engine.generateStream(
                phonemes: "sˈin tʃˈaː2w",
                speakerEmbedding: spk,
                refCodes: nil,
                temperature: 0.8,
                topK: 25,
                topP: 0.95,
                maxNewFrames: 100
            ) {
                // Yield one frame then cancel
                break
            }
        }
        task.cancel()
        _ = await task.result
        XCTAssertTrue(task.isCancelled)
    }

    func testRealV3TurboEngineGeneratesAudioWithRealModels() async throws {
        let dir = URL(fileURLWithPath: "/tmp/test_v3_turbo_models")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("vieneu_prefill.onnx").path) else {
            return
        }
        let provider = CachedModelProvider(directory: dir)
        let engine = try V3TurboEngine(modelProvider: provider)
        let spk = Array(repeating: Float(0.1), count: 192)
        let samples = try await engine.generate(
            phonemes: "sˈin tʃˈaː2w",
            speakerEmbedding: spk,
            refCodes: nil,
            temperature: 0.8,
            topK: 25,
            topP: 0.95,
            maxNewFrames: 3
        )
        print(">>> REAL TURBO ENGINE GENERATED SAMPLES:", samples.count)
        XCTAssertGreaterThan(samples.count, 0)
    }

    func testRealV3TurboNganSynthesis() async throws {
        let dir = URL(fileURLWithPath: "/tmp/test_v3_turbo_models")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("vieneu_prefill.onnx").path) else {
            return
        }
        let provider = CachedModelProvider(directory: dir)
        let config = VieNeuConfig(
            mode: .v3turbo,
            modelProvider: provider,
            maxNewFrames: 50
        )
        let tts = try await VieNeuTTS(config: config)
        let voice = try tts.loadPresetVoice(name: "Nguyễn Ngọc Ngạn")
        print(">>> Loaded voice Nguyễn Ngọc Ngạn: refCodes count =", voice.refCodes?.count ?? 0)
        let text = "Anh em nào phải người xa, Cùng chung bác mẹ, một nhà cùng thân."
        let audio = try await tts.synthesize(text: text, voice: voice)
        print(">>> SYNTHESIZED AUDIO SAMPLES:", audio.samples.count, "duration:", audio.duration)
        XCTAssertGreaterThan(audio.samples.count, 0)
    }
}
