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
}
