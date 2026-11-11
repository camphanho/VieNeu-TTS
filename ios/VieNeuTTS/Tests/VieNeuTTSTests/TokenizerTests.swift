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

    func testCombiningDiacriticsScalarsPreserved() {
        let tokenizer = PhonemeTokenizer()
        // Dental t: 't' (id 31) + combining dental diacritic '\u{032A}' / '̪' (id 63)
        let phonemes = "t\u{032A}"
        let ids = tokenizer.encode(phonemes: phonemes)
        XCTAssertEqual(ids, [1, 31, 63, 2])
    }

    func testComparePythonExactMatch() {
        let tokenizer = PhonemeTokenizer()
        let p = "sˈin tʃˈaː2w! t̪ˈoj lˌaː2 mˈo hˈi2ɲ vˈaɪ nˈuː tˈiː tˈiː ˈɛs tʃˈa6j tʃˈy6c t̪ˈiɛɜp tʃˈen ˈaɪ fˈoʊn."
        let ids = tokenizer.encode(phonemes: p)
        print(">>> SWIFT TOKENIZER IDS COUNT:", ids.count)
        print(">>> SWIFT TOKENIZER IDS:", ids)
        let expected: [Int64] = [1, 30, 60, 22, 27, 3, 31, 55, 60, 15, 62, 9, 34, 4, 3, 31, 63, 60, 28, 23, 3, 25, 61, 15, 62, 9, 3, 26, 60, 28, 3, 21, 60, 22, 9, 52, 3, 33, 60, 15, 51, 3, 27, 60, 32, 62, 3, 31, 60, 22, 62, 3, 31, 60, 22, 62, 3, 60, 47, 30, 3, 31, 55, 60, 15, 12, 23, 3, 31, 55, 60, 36, 12, 17, 3, 31, 63, 60, 22, 47, 48, 29, 3, 31, 55, 60, 19, 27, 3, 60, 15, 51, 3, 20, 60, 28, 56, 27, 7, 2]
        XCTAssertEqual(ids, expected)
    }

    func testV3TurboTokenizerAgainstHuggingFaceVector() throws {
        let tokPath = "/home/camph/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo/snapshots/61b85e3d937fbbacb387714180e8182823512523/onnx_int8/tokenizer.json"
        guard FileManager.default.fileExists(atPath: tokPath) else { return }
        let tokenizer = try V3TurboTokenizer.load(from: URL(fileURLWithPath: tokPath))
        let ph = "sˈin tʃˈaː2w"
        let ids = tokenizer.encode(phonemes: ph)
        XCTAssertEqual(ids, [159, 325, 149, 154, 358, 319, 325, 141, 327, 94, 163])
    }
}
