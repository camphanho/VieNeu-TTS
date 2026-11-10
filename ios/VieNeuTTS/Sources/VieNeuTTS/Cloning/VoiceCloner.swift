import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

/// Actor executing the reference voice cloning pipeline to enroll speaker embeddings and style tokens.
public actor VoiceCloner {
    /// Neural inference engine executing speaker, codec, and reference encoders.
    public let onnxEngine: ONNXInferenceEngine

    /// Filterbank computing Kaldi-compatible 80-bin mel spectrograms.
    private let melFilterbank: MelFilterbank

    /// Initializes a new voice cloner instance.
    /// - Parameters:
    ///   - onnxEngine: The neural inference engine conforming to `ONNXInferenceEngine`.
    ///   - melFilterbank: Pre-initialized `MelFilterbank` (defaults to a new instance).
    public init(
        onnxEngine: ONNXInferenceEngine,
        melFilterbank: MelFilterbank = MelFilterbank()
    ) {
        self.onnxEngine = onnxEngine
        self.melFilterbank = melFilterbank
    }

    /// Enrolls a new voice profile from a reference audio file URL.
    ///
    /// The audio file is converted to mono 24 kHz float samples and trimmed of leading and trailing silence
    /// using RMS energy gating. If the post-trim audio is shorter than 1.0 second or has an overall RMS
    /// energy below 1e-4, this method throws `VieNeuError.referenceAudioTooShortOrSilent`.
    ///
    /// - Parameters:
    ///   - audioURL: File URL to reference audio (e.g. WAV or system-supported audio format).
    ///   - name: Name or identifier to assign to the enrolled voice profile.
    /// - Returns: A populated `VoiceProfile` ready for speech synthesis.
    /// - Throws: `VieNeuError.referenceAudioTooShortOrSilent` or neural engine inference errors.
    public func clone(from audioURL: URL, name: String) async throws -> VoiceProfile {
        let rawSamples = try MelFilterbank.convertToMono24kHz(from: audioURL)
        guard !rawSamples.isEmpty else {
            throw VieNeuError.referenceAudioTooShortOrSilent
        }

        let trimmedSamples = Self.trimSilence(samples: rawSamples, sampleRate: 24000)
        let duration = Float(trimmedSamples.count) / 24000.0
        let rms = Self.calculateRMS(trimmedSamples)

        guard duration >= 1.0, rms >= 1e-4 else {
            throw VieNeuError.referenceAudioTooShortOrSilent
        }

        // Cap reference audio to 30 seconds for stability and memory safety
        let maxSamples = 30 * 24000
        let referenceSamples: [Float]
        if trimmedSamples.count > maxSamples {
            referenceSamples = Array(trimmedSamples.prefix(maxSamples))
        } else {
            referenceSamples = trimmedSamples
        }

        // 1. Extract 80-bin Mel spectrogram from reference waveform
        let mel = try melFilterbank.compute(samples: referenceSamples, sampleRate: 24000)

        // 2. Run speaker encoder to extract 192-d x-vector
        let speakerEmbedding = try await onnxEngine.runSpeakerEncoder(mel: mel)

        // 3. Run acoustic codec encoder to extract reference latents
        let latents = try await onnxEngine.runCodecEncoder(samples: referenceSamples)

        // 4. Run reference style encoder to extract style tokens (shape: [50, 256])
        let styleTokens = try await onnxEngine.runReferenceEncoder(latents: latents)

        return VoiceProfile(
            name: name,
            speakerEmbedding: speakerEmbedding,
            styleTokens: styleTokens
        )
    }

    /// Trims leading and trailing silence from audio samples using frame-based RMS energy.
    ///
    /// - Parameters:
    ///   - samples: Raw mono audio samples.
    ///   - sampleRate: Audio sample rate in Hz (default: 24,000).
    ///   - frameDuration: Frame window duration in seconds (default: 20ms).
    ///   - hopDuration: Hop step duration in seconds (default: 10ms).
    ///   - rmsThreshold: RMS energy threshold below which a frame is considered silent (default: 1e-4).
    /// - Returns: Trimmed audio samples retaining active speech.
    public static func trimSilence(
        samples: [Float],
        sampleRate: Int = 24000,
        frameDuration: Float = 0.02,
        hopDuration: Float = 0.01,
        rmsThreshold: Float = 1e-4
    ) -> [Float] {
        guard !samples.isEmpty else { return [] }

        let frameLength = max(1, Int(Float(sampleRate) * frameDuration))
        let hopLength = max(1, Int(Float(sampleRate) * hopDuration))

        guard samples.count >= frameLength else {
            let rms = calculateRMS(samples)
            return rms >= rmsThreshold ? samples : []
        }

        let numFrames = (samples.count - frameLength) / hopLength + 1
        var firstFrame: Int?
        var lastFrame: Int?

        for i in 0..<numFrames {
            let start = i * hopLength
            let end = start + frameLength

            var sumSq: Float = 0.0
            for j in start..<end {
                let s = samples[j]
                sumSq += s * s
            }
            let frameRMS = sqrt(sumSq / Float(frameLength))

            if frameRMS >= rmsThreshold {
                if firstFrame == nil {
                    firstFrame = i
                }
                lastFrame = i
            }
        }

        guard let first = firstFrame, let last = lastFrame else {
            return []
        }

        let sampleStart = first * hopLength
        let sampleEnd = (last == numFrames - 1) ? samples.count : min(samples.count, last * hopLength + frameLength)
        guard sampleStart < sampleEnd else { return [] }

        return Array(samples[sampleStart..<sampleEnd])
    }

    /// Calculates the Root Mean Square (RMS) energy of an array of audio samples.
    ///
    /// - Parameter samples: Array of Float32 audio samples.
    /// - Returns: RMS energy value.
    public static func calculateRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0.0 }
        #if canImport(Accelerate)
        var meanSquare: Float = 0.0
        vDSP_measqv(samples, 1, &meanSquare, vDSP_Length(samples.count))
        return sqrt(meanSquare)
        #else
        var sumSquares: Float = 0.0
        for s in samples {
            sumSquares += s * s
        }
        return sqrt(sumSquares / Float(samples.count))
        #endif
    }
}
