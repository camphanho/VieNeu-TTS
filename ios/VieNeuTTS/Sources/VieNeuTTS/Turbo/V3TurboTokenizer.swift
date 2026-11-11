import Foundation

/// Byte-Level BPE Tokenizer for VieNeu-TTS v3 Turbo, compatible with HuggingFace `tokenizer.json`.
public final class V3TurboTokenizer: Sendable {
    public let vocab: [String: Int64]
    public let merges: [(String, String)]
    private let byteToUnicodeMap: [UInt8: String]

    public init(vocab: [String: Int64], merges: [(String, String)]) {
        self.vocab = vocab
        self.merges = merges

        // Build GPT-2 Byte-Level Unicode mapping
        var bs = [Int]()
        for b in (33...126) { bs.append(b) }
        for b in (161...172) { bs.append(b) }
        for b in (174...255) { bs.append(b) }

        var cs = bs
        var n = 0
        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        var map = [UInt8: String]()
        for i in 0..<bs.count {
            let byteVal = UInt8(bs[i])
            if let scalar = UnicodeScalar(cs[i]) {
                map[byteVal] = String(scalar)
            }
        }
        self.byteToUnicodeMap = map
    }

    /// Loads tokenizer configuration from a `tokenizer.json` file.
    public static func load(from url: URL) throws -> V3TurboTokenizer {
        let data = try Data(contentsOf: url)
        return try load(from: data)
    }

    /// Decodes tokenizer configuration from JSON data.
    public static func load(from data: Data) throws -> V3TurboTokenizer {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabDict = model["vocab"] as? [String: Int],
              let rawMerges = model["merges"] as? [Any] else {
            throw VieNeuError.modelCorrupted("Invalid tokenizer.json format.")
        }

        var vocab: [String: Int64] = [:]
        for (k, v) in vocabDict {
            vocab[k] = Int64(v)
        }

        var merges: [(String, String)] = []
        for item in rawMerges {
            if let pair = item as? [String], pair.count >= 2 {
                merges.append((pair[0], pair[1]))
            } else if let str = item as? String {
                let parts = str.components(separatedBy: " ")
                if parts.count >= 2 {
                    merges.append((parts[0], parts[1]))
                }
            }
        }

        return V3TurboTokenizer(vocab: vocab, merges: merges)
    }

    /// Encodes a phoneme string into token IDs matching HuggingFace Tokenizer output.
    public func encode(phonemes: String) -> [Int64] {
        guard !phonemes.isEmpty else { return [] }
        let utf8Bytes = [UInt8](phonemes.utf8)

        var tokens = utf8Bytes.map { byteToUnicodeMap[$0] ?? String(UnicodeScalar($0)) }

        // Apply BPE merges sequentially
        for merge in merges {
            let left = merge.0
            let right = merge.1
            let merged = left + right

            var i = 0
            var newTokens = [String]()
            newTokens.reserveCapacity(tokens.count)

            while i < tokens.count {
                if i < tokens.count - 1 && tokens[i] == left && tokens[i + 1] == right {
                    newTokens.append(merged)
                    i += 2
                } else {
                    newTokens.append(tokens[i])
                    i += 1
                }
            }
            tokens = newTokens
        }

        return tokens.map { vocab[$0] ?? 0 }
    }
}
