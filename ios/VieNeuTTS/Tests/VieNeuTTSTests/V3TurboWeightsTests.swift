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
}
