import Foundation

/// Model conditioning constants for VieNeu-TTS v3 Nano, corresponding to `constants.npz`.
public struct ModelConstants: Codable, Sendable {
    public let null_spk: [Float]
    public let null_style: [[Float]]
    public let latent_mean: [Float]
    public let latent_std: [Float]
    public let latent_scale: Float

    public static let fallbackScale: Float = 0.25

    public init(
        null_spk: [Float],
        null_style: [[Float]],
        latent_mean: [Float],
        latent_std: [Float],
        latent_scale: Float = fallbackScale
    ) {
        self.null_spk = null_spk
        self.null_style = null_style
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_scale = latent_scale
    }

    /// Loads model constants from a model directory, `constants.npz` archive, bundled resources, or fallback defaults.
    public static func load(from directory: URL? = nil) -> ModelConstants {
        // 1. Try parsing binary constants.npz directly from the directory if present
        if let dir = directory {
            let npzURL = dir.appendingPathComponent("constants.npz")
            if let data = try? Data(contentsOf: npzURL),
               let parsed = parseNPZ(data: data) {
                return parsed
            }

            let jsonURL = dir.appendingPathComponent("constants_v3_nano.json")
            if let data = try? Data(contentsOf: jsonURL),
               let decoded = try? JSONDecoder().decode(ModelConstants.self, from: data) {
                return decoded
            }
        }

        // 2. Try universal resource locator for constants.npz or JSON
        if let npzURL = VieNeuResources.url(for: "constants.npz"),
           let data = try? Data(contentsOf: npzURL),
           let parsed = parseNPZ(data: data) {
            return parsed
        }

        if let resURL = VieNeuResources.url(for: "constants_v3_nano.json"),
           let data = try? Data(contentsOf: resURL),
           let decoded = try? JSONDecoder().decode(ModelConstants.self, from: data) {
            return decoded
        }

        // 3. Fallback defaults
        return ModelConstants(
            null_spk: Array(repeating: 0.0, count: 192),
            null_style: Array(repeating: Array(repeating: 0.0, count: 256), count: 50),
            latent_mean: Array(repeating: 0.0, count: 24),
            latent_std: Array(repeating: 1.0, count: 24),
            latent_scale: fallbackScale
        )
    }

    /// Parses uncompressed NumPy arrays directly from raw `constants.npz` (ZIP format) bytes.
    public static func parseNPZ(data: Data) -> ModelConstants? {
        guard data.count > 100 else { return nil }

        var arrays: [String: [Float]] = [:]
        let bytes = [UInt8](data)
        let total = bytes.count

        // Search for central directory header signatures: PK\x01\x02 (0x02014b50)
        var pos = 0
        while pos + 46 <= total {
            if bytes[pos] == 0x50 && bytes[pos + 1] == 0x4b && bytes[pos + 2] == 0x01 && bytes[pos + 3] == 0x02 {
                let compSize = Int(readUInt32(bytes: bytes, offset: pos + 20))
                let fnLen = Int(readUInt16(bytes: bytes, offset: pos + 28))
                let extraLen = Int(readUInt16(bytes: bytes, offset: pos + 30))
                let commentLen = Int(readUInt16(bytes: bytes, offset: pos + 32))
                let localHdrOffset = Int(readUInt32(bytes: bytes, offset: pos + 42))

                if pos + 46 + fnLen <= total {
                    let nameBytes = Array(bytes[(pos + 46)..<(pos + 46 + fnLen)])
                    let name = String(decoding: nameBytes, as: UTF8.self).replacingOccurrences(of: ".npy", with: "")

                    if localHdrOffset + 30 <= total {
                        let locFnLen = Int(readUInt16(bytes: bytes, offset: localHdrOffset + 26))
                        let locExtraLen = Int(readUInt16(bytes: bytes, offset: localHdrOffset + 28))
                        let dataStart = localHdrOffset + 30 + locFnLen + locExtraLen

                        if dataStart + compSize <= total && compSize >= 16 {
                            // Check \x93NUMPY header magic: 0x93, 'N', 'U', 'M', 'P', 'Y'
                            if bytes[dataStart] == 0x93 && bytes[dataStart + 1] == 0x4e && bytes[dataStart + 2] == 0x55 &&
                               bytes[dataStart + 3] == 0x4d && bytes[dataStart + 4] == 0x50 && bytes[dataStart + 5] == 0x59 {
                                let hdrLen = Int(readUInt16(bytes: bytes, offset: dataStart + 8))
                                let floatStart = dataStart + 10 + hdrLen
                                let floatBytesCount = compSize - (10 + hdrLen)
                                if floatBytesCount > 0 && floatStart + floatBytesCount <= total {
                                    let floatCount = floatBytesCount / 4
                                    var floats = [Float](repeating: 0.0, count: floatCount)
                                    data.withUnsafeBytes { rawPtr in
                                        let src = rawPtr.baseAddress!.advanced(by: floatStart)
                                        floats.withUnsafeMutableBytes { dst in
                                            dst.baseAddress!.copyMemory(from: src, byteCount: floatCount * 4)
                                        }
                                    }
                                    arrays[name] = floats
                                }
                            }
                        }
                    }
                }
                pos += 46 + fnLen + extraLen + commentLen
            } else {
                pos += 1
            }
        }

        guard let nullSpk = arrays["null_spk"], nullSpk.count == 192,
              let rawStyle = arrays["null_style"], rawStyle.count == 50 * 256 else {
            return nil
        }

        var nullStyle = [[Float]](repeating: [Float](repeating: 0.0, count: 256), count: 50)
        for r in 0..<50 {
            for c in 0..<256 {
                nullStyle[r][c] = rawStyle[r * 256 + c]
            }
        }

        let mean = arrays["latent_mean"] ?? Array(repeating: 0.0, count: 24)
        let std = arrays["latent_std"] ?? Array(repeating: 1.0, count: 24)
        let scale = arrays["latent_scale"]?.first ?? fallbackScale

        return ModelConstants(
            null_spk: nullSpk,
            null_style: nullStyle,
            latent_mean: mean,
            latent_std: std,
            latent_scale: scale
        )
    }

    private static func readUInt16(bytes: [UInt8], offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    private static func readUInt32(bytes: [UInt8], offset: Int) -> UInt32 {
        UInt32(bytes[offset]) |
        (UInt32(bytes[offset + 1]) << 8) |
        (UInt32(bytes[offset + 2]) << 16) |
        (UInt32(bytes[offset + 3]) << 24)
    }
}
