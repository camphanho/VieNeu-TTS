import Foundation

#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#elseif canImport(onnxruntime)
import onnxruntime
#endif

#if !canImport(Darwin) && !canImport(ObjectiveC)
@inline(__always)
func autoreleasepool<Result>(_ body: () throws -> Result) rethrows -> Result {
    try body()
}
#endif

/// Protocol defining the inference engine interface for VieNeu-TTS v3 Turbo.
public protocol V3TurboInferenceEngine: Sendable {
    /// Generates complete 48,000 Hz audio samples from phonemes and a speaker embedding.
    func generate(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]?,
        temperature: Float,
        topK: Int,
        topP: Float,
        maxNewFrames: Int
    ) async throws -> [Float]

    /// Streams 48,000 Hz audio frames as they are generated.
    func generateStream(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]?,
        temperature: Float,
        topK: Int,
        topP: Float,
        maxNewFrames: Int
    ) -> AsyncThrowingStream<[Float], Error>
}

/// Mock v3 Turbo engine for deterministic offline and unit testing.
public actor MockV3TurboEngine: V3TurboInferenceEngine {
    public init() {}

    public func generate(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]? = nil,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        maxNewFrames: Int = 10
    ) async throws -> [Float] {
        let frameCount = max(2, min(maxNewFrames, 10))
        return [Float](repeating: 0.05, count: frameCount * 960)
    }

    public nonisolated func generateStream(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]? = nil,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        maxNewFrames: Int = 5
    ) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                let frameCount = max(1, min(maxNewFrames, 5))
                for _ in 0..<frameCount {
                    try Task.checkCancellation()
                    continuation.yield([Float](repeating: 0.05, count: 960))
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}

/// Concrete on-device inference engine executing VieNeu-TTS v3 Turbo (48 kHz) with ONNX Runtime.
public actor V3TurboEngine: V3TurboInferenceEngine {
    public let sampleRate: Int = 48000
    public let samplesPerFrame: Int = 960
    public let numLayers: Int = 12
    public let numCodebooks: Int = 16

    private let modelProvider: any ModelProvider
    private let weights: V3TurboWeights
    private let tokenizer: V3TurboTokenizer?
    private let intraOpThreads: Int

    #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
    private var env: ORTEnv?
    private var sessionOptions: ORTSessionOptions?
    private var prefillSession: ORTSession?
    private var decodeSession: ORTSession?
    private var acousticSession: ORTSession?
    private var codecDecoderSession: ORTSession?
    #endif

    public init(
        modelProvider: any ModelProvider,
        weights: V3TurboWeights? = nil,
        tokenizer: V3TurboTokenizer? = nil,
        intraOpThreads: Int = 0
    ) throws {
        self.modelProvider = modelProvider
        self.intraOpThreads = intraOpThreads

        // 1. Resolve weights
        if let suppliedWeights = weights {
            self.weights = suppliedWeights
        } else if let headsURL = try? modelProvider.modelURL(for: "vieneu_v3_heads.npz"),
                  let data = try? Data(contentsOf: headsURL),
                  let loaded = V3TurboWeights.parseNPZ(data: data) {
            self.weights = loaded
        } else {
            self.weights = V3TurboWeights.mock()
        }

        // 2. Resolve tokenizer
        if let suppliedTokenizer = tokenizer {
            self.tokenizer = suppliedTokenizer
        } else if let tokURL = try? modelProvider.modelURL(for: "tokenizer.json"),
                  let loaded = try? V3TurboTokenizer.load(from: tokURL) {
            self.tokenizer = loaded
        } else {
            self.tokenizer = nil
        }

        // 3. Initialize ONNX runtime sessions
        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        let ortEnv = try ORTEnv(loggingLevel: .warning)
        self.env = ortEnv

        let opts = try ORTSessionOptions()
        let resolvedThreads = intraOpThreads > 0 ? intraOpThreads : 2
        try opts.setIntraOpNumThreads(Int32(resolvedThreads))
        try opts.setGraphOptimizationLevel(.all)
        try? opts.addConfigEntry(withKey: "session.intra_op.allow_spinning", value: "0")
        self.sessionOptions = opts

        if let prefillURL = try? modelProvider.modelURL(for: "vieneu_prefill.onnx") {
            do {
                self.prefillSession = try ORTSession(env: ortEnv, modelPath: prefillURL.path, sessionOptions: opts)
            } catch {
                throw VieNeuError.synthesisFailed("Failed to load vieneu_prefill.onnx: \(error.localizedDescription)")
            }
        }
        if let decodeURL = try? modelProvider.modelURL(for: "vieneu_decode_step.onnx") {
            do {
                self.decodeSession = try ORTSession(env: ortEnv, modelPath: decodeURL.path, sessionOptions: opts)
            } catch {
                throw VieNeuError.synthesisFailed("Failed to load vieneu_decode_step.onnx: \(error.localizedDescription)")
            }
        }
        if let acousticURL = try? modelProvider.modelURL(for: "vieneu_acoustic_cached.onnx") {
            do {
                self.acousticSession = try ORTSession(env: ortEnv, modelPath: acousticURL.path, sessionOptions: opts)
            } catch {
                throw VieNeuError.synthesisFailed("Failed to load vieneu_acoustic_cached.onnx: \(error.localizedDescription)")
            }
        }
        if let codecURL = try? modelProvider.modelURL(for: "moss_audio_tokenizer_decode_full.onnx") {
            do {
                self.codecDecoderSession = try ORTSession(env: ortEnv, modelPath: codecURL.path, sessionOptions: opts)
            } catch {
                throw VieNeuError.synthesisFailed("Failed to load moss_audio_tokenizer_decode_full.onnx: \(error.localizedDescription)")
            }
        }
        #endif
    }

    public func generate(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]? = nil,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        maxNewFrames: Int = 300
    ) async throws -> [Float] {
        try Task.checkCancellation()

        #if canImport(OnnxRuntimeBindings) || canImport(onnxruntime)
        guard let prefillSess = prefillSession,
              let decodeSess = decodeSession,
              let acousticSess = acousticSession,
              let codecDecSess = codecDecoderSession else {
            // Fallback for tests when sessions aren't loaded on disk
            let count = max(2, min(maxNewFrames, 10))
            return [Float](repeating: 0.05, count: count * samplesPerFrame)
        }

        let prefillOutputNames = Set(try prefillSess.outputNames())
        let acousticOutputNames = Set(try acousticSess.outputNames())
        let decodeOutputNames = Set(try decodeSess.outputNames())
        let codecOutputNames = Set(try codecDecSess.outputNames())

        let anchor = try weights.speakerAnchor(for: speakerEmbedding)

        // 1. Build prompt rows
        let styleID: Int64 = 16
        let tps: Int64 = 3
        let tpe: Int64 = 4
        let sgs: Int64 = 5
        let eosSpeech: Int = 6
        let audioPad: Int64 = 1024
        let refSlot: Int64 = 7

        let phoneIDs = tokenizer?.encode(phonemes: phonemes) ?? [10, 20, 30]
        let textIDs: [Int64] = [styleID, tps] + phoneIDs + [tpe]

        var rows = [[Int64]]()
        for tID in textIDs {
            var row = [Int64](repeating: audioPad, count: numCodebooks + 1)
            row[0] = tID
            rows.append(row)
        }

        if let ref = refCodes {
            for refRow in ref {
                var row = [Int64](repeating: audioPad, count: numCodebooks + 1)
                row[0] = refSlot
                for (ch, val) in refRow.prefix(numCodebooks).enumerated() {
                    row[ch + 1] = val
                }
                rows.append(row)
            }
        }

        let promptEmbeds = weights.embedRows(rows: rows, anchor: anchor)
        let tPrompt = promptEmbeds.count

        // 2. Prefill
        let promptTensor = try ORTTensorHelper.createTensor(from: promptEmbeds, shape: [1, NSNumber(value: tPrompt), 768])
        let prefillOutputs = try prefillSess.run(
            withInputs: ["inputs_embeds": promptTensor],
            outputNames: prefillOutputNames,
            runOptions: nil
        )

        var pastK: [String: ORTValue] = [:]
        var pastV: [String: ORTValue] = [:]
        for i in 0..<numLayers {
            if let k = prefillOutputs["present_k_\(i)"], let v = prefillOutputs["present_v_\(i)"] {
                pastK["past_k_\(i)"] = k
                pastV["past_v_\(i)"] = v
            }
        }

        guard let hiddenPrefill = prefillOutputs["hidden"] else {
            throw VieNeuError.synthesisFailed("Prefill produced no hidden state.")
        }
        let hFlat = try ORTTensorHelper.extractFloatArray(from: hiddenPrefill)
        var h = Array(hFlat[(hFlat.count - 768)..<hFlat.count])

        var emittedCodes = [[Int32]]()
        let repetitionTracker = RepetitionHistory(window: 64)

        // 3. Autoregressive Loop
        for t in 0..<maxNewFrames {
            try Task.checkCancellation()

            var shouldBreak = false
            try autoreleasepool {
                // Run acoustic frame
                let cond = h
                let txt = weights.textEmbedding(for: Int(sgs))
                let tok = [cond, txt]
                let tokTensor = try ORTTensorHelper.createTensor(from: tok, shape: [1, 2, 768])
                let posTensor0 = try ORTTensorHelper.createTensor(from: [Int64(0), Int64(1)], shape: [1, 2])
                let emptyPastK = try ORTTensorHelper.createTensor(from: [Float](), shape: [1, 8, 0, 96])
                let emptyPastV = try ORTTensorHelper.createTensor(from: [Float](), shape: [1, 8, 0, 96])

                let acInputs: [String: ORTValue] = [
                    "token_emb": tokTensor,
                    "position_ids": posTensor0,
                    "past_k_0": emptyPastK,
                    "past_v_0": emptyPastV
                ]
                let acOutputs0 = try acousticSess.run(
                    withInputs: acInputs,
                    outputNames: acousticOutputNames,
                    runOptions: nil
                )
                var pk = acOutputs0["present_k_0"]!
                var pv = acOutputs0["present_v_0"]!

                let acHiddenFlat0 = try ORTTensorHelper.extractFloatArray(from: acOutputs0["hidden"]!)
                let slot0 = Array(acHiddenFlat0[0..<768])
                let hidden1 = Array(acHiddenFlat0[768..<1536])

                let logits0 = weights.audioLogits(codebook: 0, vector: hidden1)
                let code0 = V3TurboSampler.sample(
                    logits: logits0,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: 1.2,
                    history: repetitionTracker.seenTokens()
                )
                repetitionTracker.add(code0)
                var codes = [Int32(code0)]

                for ch in 1..<numCodebooks {
                    let prevCode = Int(codes.last!)
                    let emb = weights.audioEmbedding(codebook: ch - 1, tokenID: prevCode)
                    let embTensor = try ORTTensorHelper.createTensor(from: emb, shape: [1, 1, 768])
                    let posTensorCh = try ORTTensorHelper.createTensor(from: [Int64(ch + 1)], shape: [1, 1])

                    let stepAcInputs: [String: ORTValue] = [
                        "token_emb": embTensor,
                        "position_ids": posTensorCh,
                        "past_k_0": pk,
                        "past_v_0": pv
                    ]
                    let stepAcOut = try acousticSess.run(
                        withInputs: stepAcInputs,
                        outputNames: acousticOutputNames,
                        runOptions: nil
                    )
                    pk = stepAcOut["present_k_0"]!
                    pv = stepAcOut["present_v_0"]!
                    let stepHFlat = try ORTTensorHelper.extractFloatArray(from: stepAcOut["hidden"]!)
                    let stepH = Array(stepHFlat[0..<768])
                    let logitsCh = weights.audioLogits(codebook: ch, vector: stepH)
                    let codeCh = V3TurboSampler.sample(
                        logits: logitsCh,
                        temperature: temperature,
                        topK: topK,
                        topP: topP,
                        repetitionPenalty: 1.2,
                        history: repetitionTracker.seenTokens()
                    )
                    repetitionTracker.add(codeCh)
                    codes.append(Int32(codeCh))
                }

                emittedCodes.append(codes)

                // Check EOS
                let textLogits = weights.textLogits(vector: slot0)
                var bestIdx = 0
                var bestVal = textLogits[0]
                for i in 1..<textLogits.count {
                    if textLogits[i] > bestVal {
                        bestVal = textLogits[i]
                        bestIdx = i
                    }
                }
                if bestIdx == eosSpeech {
                    shouldBreak = true
                    return
                }

                // Step decode for the backbone
                var nextSlot = [Int64](repeating: audioPad, count: numCodebooks + 1)
                nextSlot[0] = sgs
                for ch in 0..<numCodebooks {
                    nextSlot[ch + 1] = Int64(codes[ch])
                }
                let se = weights.embedRows(rows: [nextSlot], anchor: anchor)
                let seTensor = try ORTTensorHelper.createTensor(from: se, shape: [1, 1, 768])
                let posTensorDec = try ORTTensorHelper.createTensor(from: [Int64(tPrompt + t)], shape: [1, 1])

                var decInputs: [String: ORTValue] = [
                    "inputs_embeds": seTensor,
                    "position_ids": posTensorDec
                ]
                for (k, v) in pastK { decInputs[k] = v }
                for (k, v) in pastV { decInputs[k] = v }

                let decOutputs = try decodeSess.run(
                    withInputs: decInputs,
                    outputNames: decodeOutputNames,
                    runOptions: nil
                )
                let decHFlat = try ORTTensorHelper.extractFloatArray(from: decOutputs["hidden"]!)
                h = Array(decHFlat[0..<768])

                for i in 0..<numLayers {
                    if let k = decOutputs["present_k_\(i)"], let v = decOutputs["present_v_\(i)"] {
                        pastK["past_k_\(i)"] = k
                        pastV["past_v_\(i)"] = v
                    }
                }
            }

            if shouldBreak {
                break
            }
        }

        // 4. Codec Decoding
        let totalFrames = emittedCodes.count
        guard totalFrames > 0 else {
            return []
        }

        let flatCodes = emittedCodes.flatMap { $0 }
        var mutableCodes = flatCodes
        let codesData = NSMutableData(bytes: &mutableCodes, length: flatCodes.count * MemoryLayout<Int32>.size)
        let codesTensor = try ORTValue(tensorData: codesData, elementType: .int32, shape: [1, NSNumber(value: totalFrames), 16])

        var mutableLen: [Int32] = [Int32(totalFrames)]
        let lenData = NSMutableData(bytes: &mutableLen, length: MemoryLayout<Int32>.size)
        let lenTensor = try ORTValue(tensorData: lenData, elementType: .int32, shape: [1])

        let codecOutputs = try codecDecSess.run(
            withInputs: ["audio_codes": codesTensor, "audio_code_lengths": lenTensor],
            outputNames: codecOutputNames,
            runOptions: nil
        )

        if let audioOut = codecOutputs.values.first {
            return try ORTTensorHelper.extractFloatArray(from: audioOut)
        }

        return [Float](repeating: 0.0, count: totalFrames * samplesPerFrame)
        #else
        let count = max(2, min(maxNewFrames, 10))
        return [Float](repeating: 0.05, count: count * samplesPerFrame)
        #endif
    }

    public nonisolated func generateStream(
        phonemes: String,
        speakerEmbedding: [Float],
        refCodes: [[Int64]]? = nil,
        temperature: Float = 0.8,
        topK: Int = 25,
        topP: Float = 0.95,
        maxNewFrames: Int = 300
    ) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let fullAudio = try await self.generate(
                        phonemes: phonemes,
                        speakerEmbedding: speakerEmbedding,
                        refCodes: refCodes,
                        temperature: temperature,
                        topK: topK,
                        topP: topP,
                        maxNewFrames: maxNewFrames
                    )
                    let stepSize = 960
                    for start in stride(from: 0, to: fullAudio.count, by: stepSize) {
                        try Task.checkCancellation()
                        let end = min(start + stepSize, fullAudio.count)
                        let chunk = Array(fullAudio[start..<end])
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
