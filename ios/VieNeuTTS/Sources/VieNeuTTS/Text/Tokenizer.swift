// ios/VieNeuTTS/Sources/VieNeuTTS/Text/Tokenizer.swift
import Foundation

/// Error encountered during tokenizer initialization or processing.
public enum TokenizerError: Error, LocalizedError, Equatable, Sendable {
    case invalidConfigData
    case missingVocab
    case fileReadError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfigData:
            return "Failed to parse tokenizer configuration JSON."
        case .missingVocab:
            return "Configuration JSON is missing required 'vocab' field."
        case .fileReadError(let path):
            return "Failed to read configuration file at path: \(path)"
        }
    }
}

/// Tokenizer mapping Vietnamese phonemes and emotion tags to token IDs for v3 Nano.
public final class PhonemeTokenizer: @unchecked Sendable {

    /// Vocabulary map from phoneme / character string to token ID.
    public let vocab: [String: Int64]

    /// Beginning-of-sequence token ID.
    public let bosId: Int64

    /// End-of-sequence token ID.
    public let eosId: Int64

    /// Padding token ID.
    public let padId: Int64

    /// Mapping from emotion tag tokens (e.g. `<|emotion_1|>`) to replacement characters (e.g. `①`).
    public let emotionMap: [String: String]

    /// Default 81-token vocabulary for VieNeu-TTS v3 Nano.
    public static let defaultVocab: [String: Int64] = [
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        " ": 3,
        "!": 4,
        ",": 5,
        "-": 6,
        ".": 7,
        "1": 8,
        "2": 9,
        "4": 10,
        "5": 11,
        "6": 12,
        "7": 13,
        "?": 14,
        "a": 15,
        "b": 16,
        "c": 17,
        "d": 18,
        "e": 19,
        "f": 20,
        "h": 21,
        "i": 22,
        "j": 23,
        "k": 24,
        "l": 25,
        "m": 26,
        "n": 27,
        "o": 28,
        "p": 29,
        "s": 30,
        "t": 31,
        "u": 32,
        "v": 33,
        "w": 34,
        "x": 35,
        "y": 36,
        "z": 37,
        "æ": 38,
        "ð": 39,
        "ŋ": 40,
        "ɐ": 41,
        "ɑ": 42,
        "ɔ": 43,
        "ɗ": 44,
        "ə": 45,
        "ɚ": 46,
        "ɛ": 47,
        "ɜ": 48,
        "ɡ": 49,
        "ɣ": 50,
        "ɪ": 51,
        "ɲ": 52,
        "ɹ": 53,
        "ɾ": 54,
        "ʃ": 55,
        "ʊ": 56,
        "ʌ": 57,
        "ʐ": 58,
        "ʒ": 59,
        "ˈ": 60,
        "ˌ": 61,
        "ː": 62,
        "̪": 63,
        "θ": 64,
        "ᵻ": 65,
        "ʔ": 66,
        "̩": 67,
        "̃": 68,
        "ʲ": 69,
        "ɬ": 70,
        "r": 71,
        "①": 72,
        "②": 73,
        "③": 74,
        "④": 75,
        "⑤": 76,
        "⑥": 77,
        "⑦": 78,
        "⑧": 79,
        "⑨": 80
    ]

    /// Default mapping for non-verbal emotion tags in v3 Nano.
    public static let defaultEmotionMap: [String: String] = [
        "<|emotion_1|>": "①",
        "<|emotion_2|>": "②",
        "<|emotion_3|>": "③",
        "<|emotion_4|>": "④",
        "<|emotion_5|>": "⑤",
        "<|emotion_6|>": "⑥",
        "<|emotion_7|>": "⑦",
        "<|emotion_8|>": "⑧",
        "<|emotion_9|>": "⑨"
    ]

    /// Inline natural-language emotion tag aliases mapping to emotion keys.
    private static let inlineEmotionAliases: [String: String] = [
        "chuckle": "<|emotion_1|>",
        "cười": "<|emotion_1|>",
        "cuoi": "<|emotion_1|>",
        "sigh": "<|emotion_2|>",
        "thở dài": "<|emotion_2|>",
        "tho dai": "<|emotion_2|>",
        "clear throat": "<|emotion_3|>",
        "hắng giọng": "<|emotion_3|>",
        "hang giong": "<|emotion_3|>"
    ]

    /// Initializes a tokenizer with custom or default parameters.
    public init(
        vocab: [String: Int64] = PhonemeTokenizer.defaultVocab,
        bosId: Int64 = 1,
        eosId: Int64 = 2,
        padId: Int64 = 0,
        emotionMap: [String: String] = PhonemeTokenizer.defaultEmotionMap
    ) {
        self.vocab = vocab
        self.bosId = bosId
        self.eosId = eosId
        self.padId = padId
        self.emotionMap = emotionMap
    }

    /// Initializes a tokenizer by decoding a JSON `config.json` payload.
    public convenience init(configData: Data) throws {
        guard let jsonObject = try? JSONSerialization.jsonObject(with: configData, options: []) as? [String: Any] else {
            throw TokenizerError.invalidConfigData
        }

        guard let rawVocab = jsonObject["vocab"] as? [String: Any] else {
            throw TokenizerError.missingVocab
        }

        var parsedVocab: [String: Int64] = [:]
        for (key, val) in rawVocab {
            if let intVal = val as? Int64 {
                parsedVocab[key] = intVal
            } else if let intVal = val as? Int {
                parsedVocab[key] = Int64(intVal)
            } else if let numVal = val as? NSNumber {
                parsedVocab[key] = numVal.int64Value
            }
        }

        let bos = (jsonObject["bos_id"] as? NSNumber)?.int64Value ?? 1
        let eos = (jsonObject["eos_id"] as? NSNumber)?.int64Value ?? 2
        let pad = (jsonObject["pad_id"] as? NSNumber)?.int64Value ?? 0

        var emotionMap = Self.defaultEmotionMap
        if let rawEmotionMap = jsonObject["emotion_tags"] as? [String: String] {
            for (k, v) in rawEmotionMap {
                emotionMap[k] = v
            }
        }

        self.init(vocab: parsedVocab, bosId: bos, eosId: eos, padId: pad, emotionMap: emotionMap)
    }

    /// Initializes a tokenizer by loading a `config.json` file from a local URL.
    public convenience init(configURL: URL) throws {
        guard let data = try? Data(contentsOf: configURL) else {
            throw TokenizerError.fileReadError(configURL.path)
        }
        try self.init(configData: data)
    }

    /// Encodes a phoneme string into a sequence of token IDs bracketed by `bosId` and `eosId`.
    ///
    /// Special emotion bracket tags (e.g. `[cười]`, `[thở dài]`) and `<|emotion_k|>` tags
    /// are mapped to their respective character tokens according to `emotionMap`.
    /// Any characters not present in `vocab` are dropped gracefully without failing.
    public func encode(phonemes: String) -> [Int64] {
        var processed = phonemes

        // 1. Map bracketed emotion tags like [cười] or [chuckle] to <|emotion_k|>
        for (alias, canonicalTag) in Self.inlineEmotionAliases {
            let pattern = "[\(alias)]"
            if processed.contains(pattern) {
                processed = processed.replacingOccurrences(of: pattern, with: canonicalTag)
            }
        }

        // Also handle case-insensitive or space-padded bracketed tags via regex
        let bracketPattern = "\\[([^\\]]+)\\]"
        if let regex = try? NSRegularExpression(pattern: bracketPattern, options: [.caseInsensitive]) {
            let nsString = processed as NSString
            let matches = regex.matches(in: processed, options: [], range: NSRange(location: 0, length: nsString.length))
            for match in matches.reversed() {
                let inner = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespaces).lowercased()
                if let canonical = Self.inlineEmotionAliases[inner] {
                    let fullRange = match.range(at: 0)
                    processed = (processed as NSString).replacingCharacters(in: fullRange, with: canonical)
                }
            }
        }

        // 2. Map canonical emotion tags (e.g. <|emotion_1|>) to mapped characters (e.g. ①)
        for (tag, replacement) in emotionMap {
            if processed.contains(tag) {
                processed = processed.replacingOccurrences(of: tag, with: replacement)
            }
        }

        // 3. Bracket with bosId and eosId, mapping characters to vocab token IDs
        var tokenIDs: [Int64] = [bosId]

        for character in processed {
            let charString = String(character)
            if let id = vocab[charString] {
                tokenIDs.append(id)
            }
            // Out-of-vocab characters are omitted
        }

        tokenIDs.append(eosId)
        return tokenIDs
    }
}
