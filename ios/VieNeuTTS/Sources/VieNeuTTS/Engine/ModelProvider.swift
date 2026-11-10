import Foundation

#if canImport(CryptoKit)
import CryptoKit
#endif

/// Protocol defining access to model weights, configs, and constant assets.
public protocol ModelProvider: Sendable {
    /// Returns the file URL for the specified model asset filename.
    ///
    /// - Parameter filename: Filename including extension (e.g., `text_encoder.onnx`).
    /// - Throws: `VieNeuError.modelNotFound` if the asset does not exist.
    func modelURL(for filename: String) throws -> URL

    /// Returns the file URL for the model's `constants.npz` asset.
    func constantsURL() throws -> URL

    /// Returns the file URL for the model's `config.json` configuration file.
    func configURL() throws -> URL
}

public extension ModelProvider {
    func constantsURL() throws -> URL {
        try modelURL(for: "constants.npz")
    }

    func configURL() throws -> URL {
        try modelURL(for: "config.json")
    }
}

/// Model provider accessing pre-bundled models from an Apple bundle resource folder.
public final class BundleModelProvider: ModelProvider {
    public let bundle: Bundle
    public let subdirectory: String?

    public init(bundle: Bundle = .main, subdirectory: String? = "models") {
        self.bundle = bundle
        self.subdirectory = subdirectory
    }

    public func modelURL(for filename: String) throws -> URL {
        let name: String
        let ext: String?
        if let dotIndex = filename.lastIndex(of: ".") {
            name = String(filename[..<dotIndex])
            ext = String(filename[filename.index(after: dotIndex)...])
        } else {
            name = filename
            ext = nil
        }

        var candidateURL: URL?
        if let ext = ext {
            candidateURL = bundle.url(forResource: name, withExtension: ext, subdirectory: subdirectory)
        } else {
            candidateURL = bundle.url(forResource: name, withExtension: nil, subdirectory: subdirectory)
        }

        if candidateURL == nil {
            candidateURL = bundle.url(forResource: name, withExtension: ext)
        }

        if candidateURL == nil, let resourceURL = bundle.resourceURL {
            let candidate = subdirectory.map {
                resourceURL.appendingPathComponent($0).appendingPathComponent(filename)
            } ?? resourceURL.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                candidateURL = candidate
            }
        }

        guard let resolvedURL = candidateURL, FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw VieNeuError.modelNotFound(filename)
        }

        return resolvedURL
    }
}

/// Model provider accessing downloaded or pre-cached models in a local directory.
public final class CachedModelProvider: ModelProvider {
    public let directory: URL
    public let expectedChecksums: [String: String]?

    public init(directory: URL, expectedChecksums: [String: String]? = nil) {
        self.directory = directory
        self.expectedChecksums = expectedChecksums
    }

    public func modelURL(for filename: String) throws -> URL {
        let fileURL = directory.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw VieNeuError.modelNotFound(filename)
        }
        return fileURL
    }

    /// Validates SHA-256 checksums of model assets if expected checksums were provided.
    ///
    /// - Throws:
    ///   - `VieNeuError.modelNotFound` if an expected model file is missing.
    ///   - `VieNeuError.modelCorrupted` if the calculated SHA-256 checksum does not match.
    public func validateChecksums() throws {
        guard let expectedChecksums = expectedChecksums, !expectedChecksums.isEmpty else {
            return
        }

        for (filename, expectedHash) in expectedChecksums {
            let fileURL = directory.appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw VieNeuError.modelNotFound(filename)
            }

            guard let data = try? Data(contentsOf: fileURL) else {
                throw VieNeuError.modelCorrupted("Could not read file for verification: \(filename)")
            }

            let calculatedHash = SHA256Helper.hexString(for: data)
            if calculatedHash.lowercased() != expectedHash.lowercased() {
                throw VieNeuError.modelCorrupted(
                    "Checksum mismatch for \(filename): expected \(expectedHash), got \(calculatedHash)"
                )
            }
        }
    }
}

// MARK: - SHA-256 Helper

enum SHA256Helper {
    #if canImport(CryptoKit)
    static func hexString(for data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
    #else
    private static let k: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]

    private static func rotr(_ x: UInt32, _ n: UInt32) -> UInt32 {
        return (x >> n) | (x << (32 - n))
    }

    static func hexString(for data: Data) -> String {
        var h0: UInt32 = 0x6a09e667
        var h1: UInt32 = 0xbb67ae85
        var h2: UInt32 = 0x3c6ef372
        var h3: UInt32 = 0xa54ff53a
        var h4: UInt32 = 0x510e527f
        var h5: UInt32 = 0x9b05688c
        var h6: UInt32 = 0x1f83d9ab
        var h7: UInt32 = 0x5be0cd19

        let bitLength = UInt64(data.count) * 8
        var message = [UInt8](data)
        message.append(0x80)
        while (message.count % 64) != 56 {
            message.append(0)
        }
        for i in (0..<8).reversed() {
            message.append(UInt8((bitLength >> (i * 8)) & 0xff))
        }

        var w = [UInt32](repeating: 0, count: 64)
        for chunkStart in stride(from: 0, to: message.count, by: 64) {
            for i in 0..<16 {
                let offset = chunkStart + i * 4
                w[i] = (UInt32(message[offset]) << 24) |
                       (UInt32(message[offset + 1]) << 16) |
                       (UInt32(message[offset + 2]) << 8) |
                       UInt32(message[offset + 3])
            }
            for i in 16..<64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
                w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
            }

            var a = h0
            var b = h1
            var c = h2
            var d = h3
            var e = h4
            var f = h5
            var g = h6
            var h = h7

            for i in 0..<64 {
                let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
                let ch = (e & f) ^ ((~e) & g)
                let temp1 = h &+ s1 &+ ch &+ k[i] &+ w[i]
                let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
                let maj = (a & b) ^ (a & c) ^ (b & c)
                let temp2 = s0 &+ maj

                h = g
                g = f
                f = e
                e = d &+ temp1
                d = c
                c = b
                b = a
                a = temp1 &+ temp2
            }

            h0 = h0 &+ a
            h1 = h1 &+ b
            h2 = h2 &+ c
            h3 = h3 &+ d
            h4 = h4 &+ e
            h5 = h5 &+ f
            h6 = h6 &+ g
            h7 = h7 &+ h
        }

        return String(format: "%08x%08x%08x%08x%08x%08x%08x%08x", h0, h1, h2, h3, h4, h5, h6, h7)
    }
    #endif
}
