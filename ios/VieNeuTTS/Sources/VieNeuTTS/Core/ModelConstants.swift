import Foundation

/// Model conditioning constants for VieNeu-TTS v3 Nano, corresponding to `constants.npz`.
public struct ModelConstants: Codable, Sendable {
    public let null_spk: [Float]
    public let null_style: [[Float]]
    public let latent_mean: [Float]
    public let latent_std: [Float]
    public let latent_scale: Float

    public static let fallbackScale: Float = 0.25

    /// Loads model constants from a model directory, resources, or fallback defaults.
    public static func load(from directory: URL? = nil) -> ModelConstants {
        // 1. Try directory URL if provided (e.g. cached model dir)
        if let dir = directory {
            let jsonURL = dir.appendingPathComponent("constants_v3_nano.json")
            if let data = try? Data(contentsOf: jsonURL),
               let decoded = try? JSONDecoder().decode(ModelConstants.self, from: data) {
                return decoded
            }
        }

        // 2. Try universal resource locator
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
}
