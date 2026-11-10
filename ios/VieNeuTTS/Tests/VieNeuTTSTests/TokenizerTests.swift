// ios/VieNeuTTS/Tests/VieNeuTTSTests/TokenizerTests.swift
import XCTest
@testable import VieNeuTTS

final class TokenizerTests: XCTestCase {

    func testEncodeBasicPhonemesWithBosEos() {
        let tokenizer = PhonemeTokenizer()
        // Basic phonemes "a b"
        let ids = tokenizer.encode(phonemes: "a b")
        XCTAssertGreaterThan(ids.count, 2)
        XCTAssertEqual(ids.first, tokenizer.bosId)
        XCTAssertEqual(ids.last, tokenizer.eosId)

        // Verify "a" maps to 15, " " to 3, "b" to 16 in default v3 Nano vocab
        XCTAssertEqual(ids, [1, 15, 3, 16, 2])
    }

    func testEncodeEmptyPhonemesHasBosEos() {
        let tokenizer = PhonemeTokenizer()
        let ids = tokenizer.encode(phonemes: "")
        XCTAssertEqual(ids, [1, 2])
    }

    func testEncodeDropsOutOfVocabCharacters() {
        let tokenizer = PhonemeTokenizer()
        // 'Z' is not in default vocab (ids 81 tokens)
        let idsWithOOV = tokenizer.encode(phonemes: "a Z b")
        let idsWithoutOOV = tokenizer.encode(phonemes: "a  b")
        XCTAssertEqual(idsWithOOV, idsWithoutOOV)
    }

    func testEncodeMapsEmotionTagTokens() {
        let tokenizer = PhonemeTokenizer()
        // <|emotion_1|> -> ① (id 72)
        // <|emotion_2|> -> ② (id 73)
        // <|emotion_3|> -> ③ (id 74)
        let ids1 = tokenizer.encode(phonemes: "<|emotion_1|>")
        XCTAssertEqual(ids1, [1, 72, 2])

        let ids2 = tokenizer.encode(phonemes: "<|emotion_2|>")
        XCTAssertEqual(ids2, [1, 73, 2])

        let ids3 = tokenizer.encode(phonemes: "<|emotion_3|>")
        XCTAssertEqual(ids3, [1, 74, 2])
    }

    func testEncodeMapsBracketedEmotionTags() {
        let tokenizer = PhonemeTokenizer()
        // Vietnamese and English aliases:
        // [cười] / [chuckle] -> <|emotion_1|> -> ① (72)
        // [thở dài] / [sigh] -> <|emotion_2|> -> ② (73)
        // [hắng giọng] / [clear throat] -> <|emotion_3|> -> ③ (74)
        let idsChuckle = tokenizer.encode(phonemes: "[cười]")
        XCTAssertEqual(idsChuckle, [1, 72, 2])

        let idsEnglishChuckle = tokenizer.encode(phonemes: "[chuckle]")
        XCTAssertEqual(idsEnglishChuckle, [1, 72, 2])

        let idsSigh = tokenizer.encode(phonemes: "[thở dài]")
        XCTAssertEqual(idsSigh, [1, 73, 2])

        let idsClearThroat = tokenizer.encode(phonemes: "[hắng giọng]")
        XCTAssertEqual(idsClearThroat, [1, 74, 2])
    }

    func testEncodeWithPunctuationAndText() {
        let tokenizer = PhonemeTokenizer()
        let ids = tokenizer.encode(phonemes: "a, b! [cười].")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertTrue(ids.contains(72)) // contains emotion 1
        XCTAssertTrue(ids.contains(5))  // ','
        XCTAssertTrue(ids.contains(4))  // '!'
        XCTAssertTrue(ids.contains(7))  // '.'
    }

    func testCustomVocabInitialization() {
        let customVocab: [String: Int64] = [
            "<pad>": 0,
            "<bos>": 10,
            "<eos>": 20,
            "x": 30,
            "y": 40
        ]
        let tokenizer = PhonemeTokenizer(vocab: customVocab, bosId: 10, eosId: 20, padId: 0)
        let ids = tokenizer.encode(phonemes: "xy")
        XCTAssertEqual(ids, [10, 30, 40, 20])
    }

    func testInitFromConfigJSON() throws {
        let jsonString = """
        {
            "bos_id": 100,
            "eos_id": 200,
            "pad_id": 0,
            "emotion_tags": {
                "<|emotion_1|>": "①"
            },
            "vocab": {
                "<pad>": 0,
                "<bos>": 100,
                "<eos>": 200,
                "a": 50,
                "①": 99
            }
        }
        """
        let data = jsonString.data(using: .utf8)!
        let tokenizer = try PhonemeTokenizer(configData: data)
        XCTAssertEqual(tokenizer.bosId, 100)
        XCTAssertEqual(tokenizer.eosId, 200)

        let ids = tokenizer.encode(phonemes: "a <|emotion_1|>")
        XCTAssertEqual(ids, [100, 50, 99, 200])
    }

    func testInitFromInvalidConfigDataThrows() {
        let invalidData = "not valid json".data(using: .utf8)!
        XCTAssertThrowsError(try PhonemeTokenizer(configData: invalidData)) { error in
            XCTAssertEqual(error as? TokenizerError, TokenizerError.invalidConfigData)
        }

        let missingVocabData = "{\"bos_id\": 1}".data(using: .utf8)!
        XCTAssertThrowsError(try PhonemeTokenizer(configData: missingVocabData)) { error in
            XCTAssertEqual(error as? TokenizerError, TokenizerError.missingVocab)
        }
    }

    func testInitFromNonExistentURLThrows() {
        let fakeURL = URL(fileURLWithPath: "/path/to/nonexistent/config.json")
        XCTAssertThrowsError(try PhonemeTokenizer(configURL: fakeURL)) { error in
            if case .fileReadError = error as? TokenizerError {
                // Expected
            } else {
                XCTFail("Expected fileReadError, got: \(error)")
            }
        }
    }

    func testPhonemesFromSeaG2PTokenizeCleanly() throws {
        let tokenizer = PhonemeTokenizer()
        // Typical phonemes output: "sˈin tʃˈaː2w vˈiɛ6t̪ nˈaːm!"
        let rawPhones = "sˈin tʃˈaː2w vˈiɛ6t̪ nˈaːm!"
        let ids = tokenizer.encode(phonemes: rawPhones)
        XCTAssertGreaterThan(ids.count, 5)
        XCTAssertEqual(ids.first, tokenizer.bosId)
        XCTAssertEqual(ids.last, tokenizer.eosId)

        // All IDs must be within vocab range [0, 80]
        for id in ids {
            XCTAssertTrue(id >= 0 && id <= 80, "Token ID \(id) is outside default vocab range")
        }
    }
}
