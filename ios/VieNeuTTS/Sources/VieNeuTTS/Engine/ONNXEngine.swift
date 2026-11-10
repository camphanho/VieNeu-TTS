import Foundation

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime)
import onnxruntime
#endif

/// Protocol defining the neural inference engine contract for VieNeu-TTS v3 Nano models.
public protocol ONNXInferenceEngine: Sendable {
    /// Runs the phoneme text encoder.
    ///
    /// - Parameters:
    ///   - tokenIDs: Array of tokenized phoneme IDs.
    ///   - style: Reference style tokens with shape `[50, 256]`.
    /// - Returns: Text context and its validity mask.
    func runTextEncoder(tokenIDs: [Int64], style: [[Float]]) async throws -> (ctx: [[Float]], mask: [Bool])

    /// Converts the duration predictor's scalar log-duration output to acoustic frames.
    func runDurationPredictor(
        ctx: [[Float]],
        mask: [Bool],
        speakerEmbedding: [Float],
        speed: Float
    ) async throws -> Int

    /// Evaluates the conditional vector field at state `x` and time `t` for flow matching ODE integration.
    func runVectorEstimator(
        x: [[Float]],
        t: Float,
        ctx: [[Float]],
        mask: [Bool],
        speakerEmbedding: [Float],
        style: [[Float]]
    ) async throws -> [[Float]]

    /// Decodes acoustic latent representations into raw 24 kHz audio samples.
    ///
    /// - Parameter latents: Flow solver output latents.
    /// - Returns: Float32 waveform samples at 24 kHz.
    func runCodecDecoder(latents: [[Float]]) async throws -> [Float]

    /// Runs the speaker encoder (x-vector extractor) on 80-bin Mel filterbank frames.
    ///
    /// - Parameter melFilterbank: 80-bin Mel spectrogram frames.
    /// - Returns: 192-dimensional speaker embedding vector.
    func runSpeakerEncoder(melFilterbank: [[Float]]) async throws -> [Float]

    /// Runs the acoustic codec encoder on reference audio samples.
    ///
    /// - Parameter audio: 24 kHz mono reference audio samples.
    /// - Returns: Encoded acoustic latent codes.
    func runCodecEncoder(audio: [Float]) async throws -> [[Float]]

    /// Runs the reference style encoder to extract style tokens from reference codes.
    ///
    /// - Parameter refCodes: Normalized acoustic codes from the codec encoder.
    /// - Returns: Style token matrix of shape `[50, 256]`.
    func runReferenceEncoder(refCodes: [[Float]]) async throws -> [[Float]]
}

public extension ONNXInferenceEngine {
    func runSpeakerEncoder(melFilterbank: [[Float]]) async throws -> [Float] {
        Array(repeating: Float(0.0), count: 192)
    }

    func runCodecEncoder(audio: [Float]) async throws -> [[Float]] {
        Array(repeating: Array(repeating: Float(0.0), count: 256), count: 50)
    }

    func runReferenceEncoder(refCodes: [[Float]]) async throws -> [[Float]] {
        Array(repeating: Array(repeating: Float(0.0), count: 256), count: 50)
    }

    func runSpeakerEncoder(mel: [[Float]]) async throws -> [Float] {
        try await runSpeakerEncoder(melFilterbank: mel)
    }

    func runCodecEncoder(samples: [Float]) async throws -> [[Float]] {
        try await runCodecEncoder(audio: samples)
    }

    func runReferenceEncoder(latents: [[Float]]) async throws -> [[Float]] {
        try await runReferenceEncoder(refCodes: latents)
    }
}

// MARK: - Mock ONNX Engine

/// Deterministic mock implementation of `ONNXInferenceEngine` for fast, offline unit testing.
public actor MockONNXEngine: ONNXInferenceEngine {
    public init() {}

    public func runTextEncoder(tokenIDs: [Int64], style: [[Float]]) async throws -> (ctx: [[Float]], mask: [Bool]) {
        let count = max(1, tokenIDs.count)
        let ctx = (0..<count).map { _ in (0..<256).map { _ in Float(0.1) } }
        let mask = Array(repeating: true, count: count)
        return (ctx, mask)
    }

    public func runDurationPredictor(ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], speed: Float) async throws -> Int {
        ONNXEngine.durationFrames(logSeconds: 0.0, speed: speed)
    }

    public func runVectorEstimator(x: [[Float]], t: Float, ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], style: [[Float]]) async throws -> [[Float]] {
        x.map { row in row.map { $0 * 0.5 } }
    }

    public func runCodecDecoder(latents: [[Float]]) async throws -> [Float] {
        let frameCount = max(1, latents.count)
        return Array(repeating: Float(0.0), count: frameCount * 256)
    }

    public func runSpeakerEncoder(melFilterbank: [[Float]]) async throws -> [Float] {
        Array(repeating: Float(0.1), count: 192)
    }

    public func runCodecEncoder(audio: [Float]) async throws -> [[Float]] {
        Array(repeating: Array(repeating: Float(0.1), count: 24), count: 6)
    }

    public func runReferenceEncoder(refCodes: [[Float]]) async throws -> [[Float]] {
        Array(repeating: Array(repeating: Float(0.2), count: 256), count: 50)
    }
}

// MARK: - Production ONNX Engine

/// Actor managing ONNX Runtime inference sessions and tensor conversions for VieNeu-TTS v3 Nano.
public actor ONNXEngine: ONNXInferenceEngine {
    public let modelProvider: ModelProvider
    public let intraOpThreads: Int

    #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
    private var env: ORTEnv?
    private var sessionOptions: ORTSessionOptions?

    private var textSession: ORTSession?
    private var durationSession: ORTSession?
    private var vectorSession: ORTSession?
    private var decoderSession: ORTSession?

    private var speakerSession: ORTSession?
    private var codecEncSession: ORTSession?
    private var refSession: ORTSession?
    #endif

    /// Initializes the ONNX inference engine with model files supplied by a `ModelProvider`.
    ///
    /// - Parameters:
    ///   - modelProvider: Provider resolving local or cached ONNX asset URLs.
    ///   - intraOpThreads: Thread pool size for intra-operator parallelism (0 defaults to half of CPU cores, capped at 8).
    /// - Throws: `VieNeuError.modelNotFound` or `VieNeuError.synthesisFailed` if sessions cannot be initialized.
    public init(modelProvider: ModelProvider, intraOpThreads: Int = 0) throws {
        self.modelProvider = modelProvider
        self.intraOpThreads = intraOpThreads

        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        let ortEnv = try ORTEnv(loggingLevel: .warning)
        self.env = ortEnv

        let opts = try ORTSessionOptions()
        let resolvedThreads = intraOpThreads > 0 ? intraOpThreads : min(max(ProcessInfo.processInfo.processorCount / 2, 1), 8)
        try opts.setIntraOpNumThreads(Int32(resolvedThreads))
        try opts.setGraphOptimizationLevel(.all)
        try? opts.addConfigEntry(withKey: "session.intra_op.allow_spinning", value: "0")
        self.sessionOptions = opts

        let textURL = try modelProvider.modelURL(for: "text_encoder.onnx")
        self.textSession = try ORTSession(env: ortEnv, modelPath: textURL.path, sessionOptions: opts)

        let durURL = try modelProvider.modelURL(for: "duration_predictor.onnx")
        self.durationSession = try ORTSession(env: ortEnv, modelPath: durURL.path, sessionOptions: opts)

        let veURL = try modelProvider.modelURL(for: "vector_estimator.onnx")
        self.vectorSession = try ORTSession(env: ortEnv, modelPath: veURL.path, sessionOptions: opts)

        let decURL = try modelProvider.modelURL(for: "codec_decoder.onnx")
        self.decoderSession = try ORTSession(env: ortEnv, modelPath: decURL.path, sessionOptions: opts)
        #endif
    }

    /// Converts the v3 Nano scalar duration prediction to codec frames.
    public nonisolated static func durationFrames(logSeconds: Float, speed: Float) -> Int {
        let framesPerSecond: Float = 24000.0 / 256.0 / 6.0
        let seconds = min(exp(logSeconds) / max(speed, 1e-3), 15.0)
        return max(2, Int((seconds * framesPerSecond).rounded()))
    }

    public func runTextEncoder(tokenIDs: [Int64], style: [[Float]]) async throws -> (ctx: [[Float]], mask: [Bool]) {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        guard let session = textSession else {
            throw VieNeuError.synthesisFailed("Text encoder session not loaded.")
        }
        let length = tokenIDs.count
        let idsTensor = try ORTTensorHelper.createTensor(from: tokenIDs, shape: [1, NSNumber(value: length)])
        let styleTensor = try ORTTensorHelper.createTensor(
            from: style,
            shape: [1, 50, 256]
        )
        let outputs = try session.run(
            withInputs: ["ids": idsTensor, "style": styleTensor],
            outputNames: Set(try session.outputNames()),
            runOptions: nil
        )
        guard let firstOutput = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Text encoder produced no outputs.")
        }
        let flat = try ORTTensorHelper.extractFloatArray(from: firstOutput)
        let dim = 256
        let rows = length
        var mu = [[Float]](repeating: [Float](repeating: 0.0, count: dim), count: rows)
        for r in 0..<rows {
            for c in 0..<dim {
                let idx = r * dim + c
                if idx < flat.count {
                    mu[r][c] = flat[idx]
                }
            }
        }
        let mask = tokenIDs.map { $0 != 0 }
        return (mu, mask)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runDurationPredictor(ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], speed: Float) async throws -> Int {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        guard let session = durationSession else {
            throw VieNeuError.synthesisFailed("Duration predictor session not loaded.")
        }
        let rows = ctx.count
        let cols = ctx.first?.count ?? 256
        let ctxTensor = try ORTTensorHelper.createTensor(from: ctx, shape: [1, NSNumber(value: rows), NSNumber(value: cols)])
        let maskTensor = try ORTTensorHelper.createTensor(from: mask, shape: [1, NSNumber(value: rows)])
        let spkTensor = try ORTTensorHelper.createTensor(from: speakerEmbedding, shape: [1, 192])

        let outputs = try session.run(
            withInputs: ["ctx": ctxTensor, "ctx_mask": maskTensor, "spk": spkTensor],
            outputNames: Set(try session.outputNames()),
            runOptions: nil
        )
        guard let firstOutput = outputs.values.first,
              let logSeconds = try ORTTensorHelper.extractFloatArray(from: firstOutput).first else {
            throw VieNeuError.synthesisFailed("Duration predictor produced no scalar output.")
        }
        return Self.durationFrames(logSeconds: logSeconds, speed: speed)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runVectorEstimator(x: [[Float]], t: Float, ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], style: [[Float]]) async throws -> [[Float]] {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        guard let session = vectorSession else {
            throw VieNeuError.synthesisFailed("Vector estimator session not loaded.")
        }
        let frames = x.count
        let dim = x.first?.count ?? 144
        let xTensor = try ORTTensorHelper.createTransposedTensor(from: x, shape: [1, NSNumber(value: dim), NSNumber(value: frames)])
        let tTensor = try ORTTensorHelper.createTensor(from: [t], shape: [1])
        let ctxTensor = try ORTTensorHelper.createTensor(from: ctx, shape: [1, NSNumber(value: ctx.count), NSNumber(value: ctx.first?.count ?? 256)])
        let maskTensor = try ORTTensorHelper.createTensor(from: mask, shape: [1, NSNumber(value: ctx.count)])
        let spkTensor = try ORTTensorHelper.createTensor(from: speakerEmbedding, shape: [1, 192])
        let styleTensor = try ORTTensorHelper.createTensor(from: style, shape: [1, NSNumber(value: style.count), NSNumber(value: style.first?.count ?? 256)])

        let outputs = try session.run(
            withInputs: [
                "x": xTensor,
                "t": tTensor,
                "ctx": ctxTensor,
                "ctx_mask": maskTensor,
                "spk": spkTensor,
                "style": styleTensor
            ],
            outputNames: Set(try session.outputNames()),
            runOptions: nil
        )
        guard let firstOutput = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Vector estimator produced no outputs.")
        }
        return try ORTTensorHelper.extractTransposed2D(from: firstOutput, frames: frames, dim: dim)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runCodecDecoder(latents: [[Float]]) async throws -> [Float] {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        guard let session = decoderSession else {
            throw VieNeuError.synthesisFailed("Codec decoder session not loaded.")
        }
        let frames = latents.count
        let dim = latents.first?.count ?? 144
        let xTensor = try ORTTensorHelper.createTransposedTensor(from: latents, shape: [1, NSNumber(value: dim), NSNumber(value: frames)])
        let outputs = try session.run(
            withInputs: ["x": xTensor],
            outputNames: Set(try session.outputNames()),
            runOptions: nil
        )
        guard let firstOutput = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Codec decoder produced no outputs.")
        }
        return try ORTTensorHelper.extractFloatArray(from: firstOutput)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runSpeakerEncoder(melFilterbank: [[Float]]) async throws -> [Float] {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        if speakerSession == nil {
            let url = try modelProvider.modelURL(for: "speaker_encoder.onnx")
            speakerSession = try ORTSession(env: env!, modelPath: url.path, sessionOptions: sessionOptions!)
        }
        let frames = melFilterbank.count
        let melTensor = try ORTTensorHelper.createTensor(from: melFilterbank, shape: [1, NSNumber(value: frames), 80])
        let inputName = (try? speakerSession!.inputNames())?.first ?? "input"
        let outputs = try speakerSession!.run(withInputs: [inputName: melTensor], outputNames: Set(try speakerSession!.outputNames()), runOptions: nil)
        guard let first = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Speaker encoder produced no outputs.")
        }
        return try ORTTensorHelper.extractFloatArray(from: first)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runCodecEncoder(audio: [Float]) async throws -> [[Float]] {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        if codecEncSession == nil {
            let url = try modelProvider.modelURL(for: "codec_encoder.onnx")
            codecEncSession = try ORTSession(env: env!, modelPath: url.path, sessionOptions: sessionOptions!)
        }
        let audioTensor = try ORTTensorHelper.createTensor(from: audio, shape: [1, 1, NSNumber(value: audio.count)])
        let inputName = (try? codecEncSession!.inputNames())?.first ?? "wav"
        let outputs = try codecEncSession!.run(withInputs: [inputName: audioTensor], outputNames: Set(try codecEncSession!.outputNames()), runOptions: nil)
        guard let first = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Codec encoder produced no outputs.")
        }
        let flat = try ORTTensorHelper.extractFloatArray(from: first)
        let channels = 24
        let frames = flat.count / channels
        return TensorLayoutHelper.transposeFromChannelFirst(flat: flat, frames: frames, dim: channels)
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }

    public func runReferenceEncoder(refCodes: [[Float]]) async throws -> [[Float]] {
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        if refSession == nil {
            let url = try modelProvider.modelURL(for: "reference_encoder.onnx")
            refSession = try ORTSession(env: env!, modelPath: url.path, sessionOptions: sessionOptions!)
        }
        let frames = refCodes.count
        let dim = refCodes.first?.count ?? 256
        let refTensor = try ORTTensorHelper.createTransposedTensor(from: refCodes, shape: [1, NSNumber(value: dim), NSNumber(value: frames)])
        let maskTensor = try ORTTensorHelper.createTensor(from: Array(repeating: true, count: frames), shape: [1, NSNumber(value: frames)])
        let outputs = try refSession!.run(withInputs: ["ref": refTensor, "ref_mask": maskTensor], outputNames: Set(try refSession!.outputNames()), runOptions: nil)
        guard let first = outputs.values.first else {
            throw VieNeuError.synthesisFailed("Reference encoder produced no outputs.")
        }
        let flat = try ORTTensorHelper.extractFloatArray(from: first)
        let rows = 50
        let cols = 256
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: cols), count: rows)
        for r in 0..<rows {
            for c in 0..<cols {
                let idx = r * cols + c
                if idx < flat.count {
                    result[r][c] = flat[idx]
                }
            }
        }
        return result
        #else
        throw VieNeuError.synthesisFailed("ONNX Runtime is not available on this platform.")
        #endif
    }
}

// MARK: - Tensor Layout Helpers

/// Helper for tensor transposition between row-major `[frames, dim]` and channel-first `[dim, frames]`.
public enum TensorLayoutHelper {
    /// Transposes row-major 2D array `[frames, dim]` into channel-first flat layout `[dim, frames]`
    /// such that `transposed[c * frames + r] = array2D[r][c]`.
    public static func transposeToChannelFirst(array2D: [[Float]]) -> [Float] {
        guard !array2D.isEmpty else { return [] }
        let frames = array2D.count
        let dim = array2D[0].count
        var transposed = [Float](repeating: 0.0, count: frames * dim)
        for r in 0..<frames {
            let row = array2D[r]
            for c in 0..<dim {
                transposed[c * frames + r] = row[c]
            }
        }
        return transposed
    }

    /// Unpacks flat channel-first tensor `[dim, frames]` into row-major 2D array `[frames, dim]`
    /// such that `result[r][c] = flat[c * frames + r]`.
    public static func transposeFromChannelFirst(flat: [Float], frames: Int, dim: Int) -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: dim), count: frames)
        for r in 0..<frames {
            for c in 0..<dim {
                let idx = c * frames + r
                if idx < flat.count {
                    result[r][c] = flat[idx]
                }
            }
        }
        return result
    }
}

// MARK: - ONNX Tensor Helpers

#if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
enum ORTTensorHelper {
    static func createTensor(from array: [Float], shape: [NSNumber]) throws -> ORTValue {
        var mutableArray = array
        let data = NSMutableData(bytes: &mutableArray, length: array.count * MemoryLayout<Float>.size)
        return try ORTValue(tensorData: data, elementType: .float, shape: shape)
    }

    static func createTensor(from array: [[Float]], shape: [NSNumber]) throws -> ORTValue {
        let flattened = array.flatMap { $0 }
        return try createTensor(from: flattened, shape: shape)
    }

    static func createTransposedTensor(from array2D: [[Float]], shape: [NSNumber]) throws -> ORTValue {
        let transposed = TensorLayoutHelper.transposeToChannelFirst(array2D: array2D)
        return try createTensor(from: transposed, shape: shape)
    }

    static func extractTransposed2D(from value: ORTValue, frames: Int, dim: Int) throws -> [[Float]] {
        let flat = try extractFloatArray(from: value)
        return TensorLayoutHelper.transposeFromChannelFirst(flat: flat, frames: frames, dim: dim)
    }

    static func createTensor(from array: [Int64], shape: [NSNumber]) throws -> ORTValue {
        var mutableArray = array
        let data = NSMutableData(bytes: &mutableArray, length: array.count * MemoryLayout<Int64>.size)
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape)
    }

    static func createTensor(from array: [Bool], shape: [NSNumber]) throws -> ORTValue {
        var boolArray = array.map { $0 ? UInt8(1) : UInt8(0) }
        let data = NSMutableData(bytes: &boolArray, length: boolArray.count * MemoryLayout<UInt8>.size)
        return try ORTValue(tensorData: data, elementType: .bool, shape: shape)
    }

    static func extractFloatArray(from value: ORTValue) throws -> [Float] {
        let data = try value.tensorData()
        let count = data.length / MemoryLayout<Float>.size
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBytes { ptr in
            data.getBytes(ptr.baseAddress!, length: data.length)
        }
        return result
    }

    static func extractInt64Array(from value: ORTValue) throws -> [Int64] {
        let data = try value.tensorData()
        let count = data.length / MemoryLayout<Int64>.size
        var result = [Int64](repeating: 0, count: count)
        result.withUnsafeMutableBytes { ptr in
            data.getBytes(ptr.baseAddress!, length: data.length)
        }
        return result
    }
}
#endif
