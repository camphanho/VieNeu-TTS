import XCTest
@testable import VieNeuTTS

final class V3TurboWeightsTests: XCTestCase {
    func testSpeakerAnchorProjectionShapeAndValues() throws {
        let weights = V3TurboWeights.mock()
        let spk192 = Array(repeating: Float(0.5), count: 192)
        let anchor = try weights.speakerAnchor(for: spk192)
        XCTAssertEqual(anchor.count, 768)
        XCTAssertFalse(anchor.allSatisfy { $0 == 0 })
    }

    func testZeroSpeakerEmbeddingThrows() {
        let weights = V3TurboWeights.mock()
        let zeros = Array(repeating: Float(0.0), count: 192)
        XCTAssertThrowsError(try weights.speakerAnchor(for: zeros)) { error in
            guard let vieNeuError = error as? VieNeuError, case .synthesisFailed = vieNeuError else {
                XCTFail("Expected synthesisFailed on all-zero speaker embedding")
                return
            }
        }
    }

    func testEmbeddingLookup() {
        let weights = V3TurboWeights.mock()
        let textVector = weights.textEmbedding(for: 1)
        XCTAssertEqual(textVector.count, 768)

        let audioVector = weights.audioEmbedding(codebook: 0, tokenID: 100)
        XCTAssertEqual(audioVector.count, 768)
    }

    func testEmbedRowsMatchesExpectedDimension() throws {
        let weights = V3TurboWeights.mock()
        let spk192 = Array(repeating: Float(0.5), count: 192)
        let anchor = try weights.speakerAnchor(for: spk192)

        // rows: 3 frames, 17 columns (1 text + 16 audio)
        let rows: [[Int64]] = [
            [1, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            [2, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
        ]
        let embeds = weights.embedRows(rows: rows, anchor: anchor, audioPadID: 1024)
        XCTAssertEqual(embeds.count, 2)
        XCTAssertEqual(embeds[0].count, 768)
    }

    func testAudioAndTextLogitsComputation() {
        let weights = V3TurboWeights.mock()
        let vec = [Float](repeating: 0.1, count: 768)
        let aLogits = weights.audioLogits(codebook: 0, vector: vec)
        XCTAssertEqual(aLogits.count, 1024)

        let tLogits = weights.textLogits(vector: vec)
        XCTAssertEqual(tLogits.count, 419)
    }

    func testParseNPZFromRealTurboHeadsFile() throws {
        let headsPath = "/home/camph/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo/snapshots/61b85e3d937fbbacb387714180e8182823512523/onnx_int8/vieneu_v3_heads.npz"
        guard FileManager.default.fileExists(atPath: headsPath) else { return }
        let data = try Data(contentsOf: URL(fileURLWithPath: headsPath))
        let weights = V3TurboWeights.parseNPZ(data: data)
        XCTAssertNotNil(weights)
        XCTAssertEqual(weights?.textEmb.count, 419 * 768)
        XCTAssertEqual(weights?.audioEmb.count, 16 * 1024 * 768)
        XCTAssertEqual(weights?.xvecW.count, 768 * 192)
    }
}
