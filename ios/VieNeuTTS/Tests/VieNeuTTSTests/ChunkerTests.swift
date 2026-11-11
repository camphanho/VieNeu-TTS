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

    func testEmptyRawStringReturnsEmptyChunks() {
        let chunks = Chunker.splitToChunks("", maxChars: 50)
        XCTAssertTrue(chunks.isEmpty)
    }

    func testSingleSentenceWithinMaxChars() {
        let text = "Xin chào Việt Nam."
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, "Xin chào Việt Nam.")
        XCTAssertEqual(chunks[0].pauseDuration, 0.2, accuracy: 0.01)
    }

    func testCaDaoChunks() {
        let text = """
        Anh em nào phải người xa,
        Cùng chung bác mẹ, một nhà cùng thân.
        Yêu nhau như thể tay chân,
        Anh em hòa thuận, hai thân vui vầy.
        """
        let chunks = Chunker.splitToChunks(text, maxChars: 140)
        print(">>> SWIFT CA DAO CHUNKS COUNT:", chunks.count)
        for (i, c) in chunks.enumerated() {
            print("  Chunk \(i): [\(c.text)] pause=\(c.pauseDuration)")
        }
    }

    func testParagraphBreaksProduceParagraphPauses() {
        let text = "Đoạn văn thứ nhất.\n\nĐoạn văn thứ hai."
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].text, "Đoạn văn thứ nhất.")
        XCTAssertEqual(chunks[0].pauseDuration, 0.3, accuracy: 0.01)
        XCTAssertEqual(chunks[1].text, "Đoạn văn thứ hai.")
        XCTAssertEqual(chunks[1].pauseDuration, 0.2, accuracy: 0.01)
    }

    func testSentenceTerminatorsPauseDuration() {
        let text1 = "Câu này kết thúc bằng dấu chấm."
        let text2 = "Câu này kết thúc bằng dấu chấm than!"
        let text3 = "Câu này kết thúc bằng dấu hỏi?"

        let chunk1 = Chunker.splitToChunks(text1, maxChars: 100).first
        let chunk2 = Chunker.splitToChunks(text2, maxChars: 100).first
        let chunk3 = Chunker.splitToChunks(text3, maxChars: 100).first

        XCTAssertNotNil(chunk1)
        XCTAssertNotNil(chunk2)
        XCTAssertNotNil(chunk3)

        XCTAssertEqual(chunk1?.pauseDuration ?? 0, 0.2, accuracy: 0.01)
        XCTAssertEqual(chunk2?.pauseDuration ?? 0, 0.2, accuracy: 0.01)
        XCTAssertEqual(chunk3?.pauseDuration ?? 0, 0.2, accuracy: 0.01)
    }

    func testLongSentenceSplitsOnSecondaryPunctuation() {
        let longSentence = "VieNeu-TTS là mô hình tổng hợp giọng nói tiếng Việt; nó hỗ trợ nhân bản giọng tức thì, và chạy cực nhanh trên thiết bị."
        let chunks = Chunker.splitToChunks(longSentence, maxChars: 60)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.text.count, 75) // within reasonable slack
            XCTAssertTrue(chunk.pauseDuration >= 0.15 && chunk.pauseDuration <= 0.3)
        }
    }

    func testSentenceChunkEquality() {
        let chunkA = SentenceChunk(text: "Chào bạn.", pauseDuration: 0.2)
        let chunkB = SentenceChunk(text: "Chào bạn.", pauseDuration: 0.2)
        let chunkC = SentenceChunk(text: "Chào bạn.", pauseDuration: 0.3)
        XCTAssertEqual(chunkA, chunkB)
        XCTAssertNotEqual(chunkA, chunkC)
    }

    func testDoesNotSplitOnDecimalNumbers() {
        let text = "Tỷ lệ tăng trưởng đạt 3.5 phần trăm vào lúc 8.30 sáng nay."
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertTrue(chunks[0].text.contains("3.5"))
        XCTAssertTrue(chunks[0].text.contains("8.30"))
    }

    func testQuotesAndParenthesesStayTogether() {
        let text = "Người dẫn chuyện nói: \"VieNeu-TTS chạy rất nhanh!\" rồi mỉm cười."
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertTrue(chunks[0].text.contains("\"VieNeu-TTS chạy rất nhanh!\""))
    }

    func testEllipsisPreserved() {
        let text = "Đang suy nghĩ… và chờ đợi."
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertTrue(chunks[0].text.contains("…"))
    }

    func testWhitespaceAndTabsNormalized() {
        let text = "\t  Hôm nay\t   trời đẹp.   \n\n\t   Ngày mai cũng vậy.  \t "
        let chunks = Chunker.splitToChunks(text, maxChars: 100)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].text, "Hôm nay trời đẹp.")
        XCTAssertEqual(chunks[1].text, "Ngày mai cũng vậy.")
    }

    func testUnbalancedQuotesFallBackToSplitting() {
        // An unclosed quote should not swallow all subsequent sentences into one huge chunk
        let text = "Một dấu nháy mở \" nhưng không bao giờ đóng. Câu thứ hai vẫn phải được tách! Và câu thứ ba nữa?"
        let chunks = Chunker.splitToChunks(text, maxChars: 40)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.contains { $0.text.contains("Câu thứ hai") })
    }
}
