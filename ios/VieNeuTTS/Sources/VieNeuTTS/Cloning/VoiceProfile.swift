import Foundation

/// Represents an enrolled voice profile containing speaker identity and reference style representations.
public struct VoiceProfile: Codable, Sendable, Equatable {
    /// Human-readable identifier for the voice profile.
    public let name: String

    /// 192-dimensional x-vector speaker embedding representing acoustic vocal tract characteristics.
    public let speakerEmbedding: [Float]

    /// 50 x 256 matrix of reference style tokens representing prosody, cadence, and delivery.
    public let styleTokens: [[Float]]

    /// Initializes a new voice profile.
    /// - Parameters:
    ///   - name: Voice profile identifier.
    ///   - speakerEmbedding: 192-dimensional speaker embedding vector.
    ///   - styleTokens: 50 x 256 style token representations.
    public init(
        name: String,
        speakerEmbedding: [Float],
        styleTokens: [[Float]]
    ) {
        self.name = name
        self.speakerEmbedding = speakerEmbedding
        self.styleTokens = styleTokens
    }

    /// Exports the voice profile as formatted JSON to the specified file URL.
    /// - Parameter url: Destination file URL.
    /// - Throws: An error if JSON encoding or filesystem write fails.
    public func export(to url: URL) throws {
        let parentDir = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// Loads a voice profile from a JSON file at the specified URL.
    /// - Parameter url: Source file URL.
    /// - Returns: Decoded `VoiceProfile` instance.
    /// - Throws: An error if file reading or JSON decoding fails.
    public static func load(from url: URL) throws -> VoiceProfile {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(VoiceProfile.self, from: data)
    }

    /// Returns a mock voice profile with default 192-d speaker embedding and 50x256 style tokens for testing.
    /// - Parameter name: Name for the mock profile.
    /// - Returns: A populated `VoiceProfile` instance.
    public static func mock(name: String = "MockVoice") -> VoiceProfile {
        VoiceProfile(
            name: name,
            speakerEmbedding: Array(repeating: Float(0.1), count: 192),
            styleTokens: Array(repeating: Array(repeating: Float(0.2), count: 256), count: 50)
        )
    }
}
