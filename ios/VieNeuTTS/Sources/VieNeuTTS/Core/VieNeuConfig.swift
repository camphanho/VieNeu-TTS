import Foundation

/// Architecture modes supported by VieNeuTTS.
public enum VieNeuModelMode: String, Codable, Sendable {
    /// v3 Nano flow-matching model (48M params, 24 kHz output).
    case v3nano
    /// v3 Turbo autoregressive model (768-d backbone + 16-RVQ heads, 48 kHz output).
    case v3turbo
}

/// Mock model provider used for offline testing or placeholder configurations.
public struct MockModelProvider: ModelProvider, Sendable {
    public init() {}

    public func modelURL(for filename: String) throws -> URL {
        URL(fileURLWithPath: "/dev/null")
    }
}

/// Configuration options for the VieNeuTTS speech synthesis pipeline.
public struct VieNeuConfig: Sendable {
    /// Model architecture mode (.v3nano or .v3turbo).
    public var mode: VieNeuModelMode

    /// Provider resolving model asset URLs.
    public var modelProvider: any ModelProvider

    /// Number of Euler ODE integration steps for v3 Nano (default: 16).
    public var eulerSteps: Int

    /// Classifier-Free Guidance extrapolation scale for v3 Nano (default: 3.0).
    public var cfgScale: Float

    /// Time-grid warping schedule parameter for v3 Nano (default: 0.0).
    public var sway: Float

    /// Speech synthesis speed multiplier (default: 1.0).
    public var speed: Float

    /// Maximum character length per chunk for sentence splitting (default: 140).
    public var maxChars: Int

    /// Sampling temperature for v3 Turbo (default: 0.8).
    public var temperature: Float

    /// Top-K sampling cutoff for v3 Turbo (default: 25).
    public var topK: Int

    /// Top-P (nucleus) sampling threshold for v3 Turbo (default: 0.95).
    public var topP: Float

    /// Repetition penalty multiplier for v3 Turbo (default: 1.2).
    public var repetitionPenalty: Float

    /// Maximum generated audio frames for v3 Turbo (default: 300).
    public var maxNewFrames: Int

    /// Optional neural engine override, useful for injecting `MockONNXEngine` in unit tests.
    public var onnxEngineOverride: (any ONNXInferenceEngine)?

    /// Optional v3 Turbo engine override.
    public var turboEngineOverride: (any V3TurboInferenceEngine)?

    /// Creates a new configuration instance.
    public init(
        mode: VieNeuModelMode = .v3nano,
        modelProvider: any ModelProvider = MockModelProvider(),
        eulerSteps: Int = 16,
        cfgScale: Float = 3.0,
        sway: Float = 0.0,
        speed: Float = 1.0,
        maxChars: Int = 140,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        maxNewFrames: Int = 300,
        onnxEngineOverride: (any ONNXInferenceEngine)? = nil,
        turboEngineOverride: (any V3TurboInferenceEngine)? = nil
    ) {
        self.mode = mode
        self.modelProvider = modelProvider
        self.eulerSteps = eulerSteps
        self.cfgScale = cfgScale
        self.sway = sway
        self.speed = speed
        self.maxChars = maxChars
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxNewFrames = maxNewFrames
        self.onnxEngineOverride = onnxEngineOverride
        self.turboEngineOverride = turboEngineOverride
    }

    /// Convenience factory creating a mock configuration with `MockModelProvider` and `MockONNXEngine`.
    public static func mock(
        mode: VieNeuModelMode = .v3nano,
        modelProvider: any ModelProvider = MockModelProvider(),
        eulerSteps: Int = 16,
        cfgScale: Float = 3.0,
        sway: Float = 0.0,
        speed: Float = 1.0,
        maxChars: Int = 140,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        maxNewFrames: Int = 300,
        onnxEngineOverride: (any ONNXInferenceEngine)? = MockONNXEngine()
    ) -> VieNeuConfig {
        VieNeuConfig(
            mode: mode,
            modelProvider: modelProvider,
            eulerSteps: eulerSteps,
            cfgScale: cfgScale,
            sway: sway,
            speed: speed,
            maxChars: maxChars,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxNewFrames: maxNewFrames,
            onnxEngineOverride: onnxEngineOverride
        )
    }
}

/// Convenience function returning a mock `VieNeuConfig` for test environments.
public func MockVieNeuConfig(
    mode: VieNeuModelMode = .v3nano,
    modelProvider: any ModelProvider = MockModelProvider(),
    eulerSteps: Int = 16,
    cfgScale: Float = 3.0,
    sway: Float = 0.0,
    speed: Float = 1.0,
    maxChars: Int = 140,
    onnxEngineOverride: (any ONNXInferenceEngine)? = MockONNXEngine()
) -> VieNeuConfig {
    VieNeuConfig.mock(
        mode: mode,
        modelProvider: modelProvider,
        eulerSteps: eulerSteps,
        cfgScale: cfgScale,
        sway: sway,
        speed: speed,
        maxChars: maxChars,
        onnxEngineOverride: onnxEngineOverride
    )
}
