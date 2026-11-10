import Foundation

/// Mock model provider used for offline testing or placeholder configurations.
public struct MockModelProvider: ModelProvider, Sendable {
    public init() {}

    public func modelURL(for filename: String) throws -> URL {
        URL(fileURLWithPath: "/dev/null")
    }
}

/// Configuration options for the VieNeuTTS speech synthesis pipeline.
public struct VieNeuConfig: Sendable {
    /// Provider resolving model asset URLs.
    public var modelProvider: any ModelProvider

    /// Number of Euler ODE integration steps (default: 16).
    public var eulerSteps: Int

    /// Classifier-Free Guidance extrapolation scale (default: 3.0).
    public var cfgScale: Float

    /// Time-grid warping schedule parameter (default: 0.0).
    public var sway: Float

    /// Speech synthesis speed multiplier (default: 1.0).
    public var speed: Float

    /// Maximum character length per chunk for sentence splitting (default: 140).
    public var maxChars: Int

    /// Optional neural engine override, useful for injecting `MockONNXEngine` in unit tests.
    public var onnxEngineOverride: (any ONNXInferenceEngine)?

    /// Creates a new configuration instance.
    public init(
        modelProvider: any ModelProvider = MockModelProvider(),
        eulerSteps: Int = 16,
        cfgScale: Float = 3.0,
        sway: Float = 0.0,
        speed: Float = 1.0,
        maxChars: Int = 140,
        onnxEngineOverride: (any ONNXInferenceEngine)? = nil
    ) {
        self.modelProvider = modelProvider
        self.eulerSteps = eulerSteps
        self.cfgScale = cfgScale
        self.sway = sway
        self.speed = speed
        self.maxChars = maxChars
        self.onnxEngineOverride = onnxEngineOverride
    }

    /// Convenience factory creating a mock configuration with `MockModelProvider` and `MockONNXEngine`.
    public static func mock(
        modelProvider: any ModelProvider = MockModelProvider(),
        eulerSteps: Int = 16,
        cfgScale: Float = 3.0,
        sway: Float = 0.0,
        speed: Float = 1.0,
        maxChars: Int = 140,
        onnxEngineOverride: (any ONNXInferenceEngine)? = MockONNXEngine()
    ) -> VieNeuConfig {
        VieNeuConfig(
            modelProvider: modelProvider,
            eulerSteps: eulerSteps,
            cfgScale: cfgScale,
            sway: sway,
            speed: speed,
            maxChars: maxChars,
            onnxEngineOverride: onnxEngineOverride
        )
    }
}

/// Convenience function returning a mock `VieNeuConfig` for test environments.
public func MockVieNeuConfig(
    modelProvider: any ModelProvider = MockModelProvider(),
    eulerSteps: Int = 16,
    cfgScale: Float = 3.0,
    sway: Float = 0.0,
    speed: Float = 1.0,
    maxChars: Int = 140,
    onnxEngineOverride: (any ONNXInferenceEngine)? = MockONNXEngine()
) -> VieNeuConfig {
    VieNeuConfig.mock(
        modelProvider: modelProvider,
        eulerSteps: eulerSteps,
        cfgScale: cfgScale,
        sway: sway,
        speed: speed,
        maxChars: maxChars,
        onnxEngineOverride: onnxEngineOverride
    )
}

/// A sentence-level audio chunk emitted by the streaming synthesis pipeline.
public struct AudioChunk: Sendable, Equatable {
    /// Zero-based sequential chunk index.
    public let index: Int

    /// Normalized sentence text for this chunk.
    public let text: String

    /// Synthesized 24 kHz audio buffer.
    public let audio: AudioBuffer

    /// Natural pause duration (in seconds) following this chunk.
    public let pauseAfter: Float

    public init(
        index: Int,
        text: String,
        audio: AudioBuffer,
        pauseAfter: Float
    ) {
        self.index = index
        self.text = text
        self.audio = audio
        self.pauseAfter = pauseAfter
    }
}
