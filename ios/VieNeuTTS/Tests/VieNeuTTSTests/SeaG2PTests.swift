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
        let phonemes = try g2p.phonemize("")
        XCTAssertEqual(phonemes, "")
    }

    func testNormalizationVariousPunctuation() throws {
        let g2p = SeaG2P()
        let input = "Xin chào! Bạn khỏe không? Tôi ổn, cảm ơn: đây là 1; 2."
        let normalized = try g2p.normalizePunctuation(input)

        // Punctuation marks should have space before them
        XCTAssertTrue(normalized.contains(" !"))
        XCTAssertTrue(normalized.contains(" ?"))
        XCTAssertTrue(normalized.contains(" ,"))
        XCTAssertTrue(normalized.contains(" :"))
        XCTAssertTrue(normalized.contains(" ;"))
        XCTAssertTrue(normalized.contains(" ."))

        let phonemes = try g2p.phonemize(normalized)
        XCTAssertFalse(phonemes.isEmpty)
    }

    func testAlreadyNormalizedPunctuationNotDuplicated() throws {
        let g2p = SeaG2P()
        let input = "Xin chào ! Bạn khỏe không ?"
        let normalized = try g2p.normalizePunctuation(input)
        XCTAssertEqual(normalized, "Xin chào ! Bạn khỏe không ?")
    }

    func testCustomDictionaryPath() throws {
        let accepted = SeaG2P.setDictionaryPath(".venv/lib/python3.12/site-packages/sea_g2p/sea_g2p.bin")
        XCTAssertTrue(accepted)

        let g2p = SeaG2P(dictionaryPath: ".venv/lib/python3.12/site-packages/sea_g2p/sea_g2p.bin")
        let phonemes = try g2p.phonemize("Việt Nam")
        XCTAssertFalse(phonemes.isEmpty)
    }

    func testConcurrentPhonemization() async throws {
        let g2p = SeaG2P()
        let sentences = [
            "Hà Nội là thủ đô của Việt Nam.",
            "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn.",
            "Đà Nẵng là thành phố đáng sống.",
            "Cần Thơ gạo trắng nước trong.",
            "Hải Phòng thành phố hoa phượng đỏ."
        ]

        try await withThrowingTaskGroup(of: String.self) { group in
            for text in sentences {
                group.addTask {
                    let norm = try g2p.normalizePunctuation(text)
                    return try g2p.phonemize(norm)
                }
            }

            var count = 0
            for try await result in group {
                XCTAssertFalse(result.isEmpty)
                count += 1
            }
            XCTAssertEqual(count, sentences.count)
        }
    }
}
