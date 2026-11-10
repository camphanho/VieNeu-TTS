import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

/// Container and projection engine for VieNeu-TTS v3 Turbo head weights and embeddings.
public final class V3TurboWeights: Sendable {
    public let hiddenSize: Int = 768
    public let textVocabSize: Int = 419
    public let audioVocabSize: Int = 1024
    public let numCodebooks: Int = 16
    public let speakerDim: Int = 192

    // Flat memory buffers
    private let textEmb: [Float]
    private let audioEmb: [Float]
    private let xvecW: [Float]
    private let xvecB: [Float]
    private let xvecLnW: [Float]
    private let xvecLnB: [Float]
    private let xvecLnEps: Float

    public init(
        textEmb: [Float],
        audioEmb: [Float],
        xvecW: [Float],
        xvecB: [Float],
        xvecLnW: [Float],
        xvecLnB: [Float],
        xvecLnEps: Float = 1e-5
    ) {
        self.textEmb = textEmb
        self.audioEmb = audioEmb
        self.xvecW = xvecW
        self.xvecB = xvecB
        self.xvecLnW = xvecLnW
        self.xvecLnB = xvecLnB
        self.xvecLnEps = xvecLnEps
    }

    /// Looks up the 768-dimensional text embedding for a token ID.
    public func textEmbedding(for tokenID: Int) -> [Float] {
        guard tokenID >= 0 && tokenID < textVocabSize else {
            return [Float](repeating: 0.0, count: hiddenSize)
        }
        let start = tokenID * hiddenSize
        let end = start + hiddenSize
        return Array(textEmb[start..<end])
    }

    /// Looks up the 768-dimensional audio codebook embedding for `(codebook, tokenID)`.
    public func audioEmbedding(codebook: Int, tokenID: Int) -> [Float] {
        guard codebook >= 0 && codebook < numCodebooks,
              tokenID >= 0 && tokenID < audioVocabSize else {
            return [Float](repeating: 0.0, count: hiddenSize)
        }
        let start = (codebook * audioVocabSize + tokenID) * hiddenSize
        let end = start + hiddenSize
        return Array(audioEmb[start..<end])
    }

    /// Computes the 768-dimensional global speaker anchor from a 192-dimensional x-vector.
    public func speakerAnchor(for speakerEmb: [Float]) throws -> [Float] {
        guard speakerEmb.count == speakerDim, !speakerEmb.allSatisfy({ $0 == 0 }) else {
            throw VieNeuError.synthesisFailed("Invalid speaker embedding: must be 192-d non-zero vector.")
        }

        var v = [Float](repeating: 0.0, count: hiddenSize)

        #if canImport(Accelerate)
        // Matrix-vector multiplication: v = xvecW * speakerEmb + xvecB
        // xvecW has shape (768, 192), speakerEmb has shape (192, 1) -> v has shape (768, 1)
        vDSP_mmul(
            xvecW, 1,
            speakerEmb, 1,
            &v, 1,
            vDSP_Length(hiddenSize),
            1,
            vDSP_Length(speakerDim)
        )
        // Add bias: v = v + xvecB
        vDSP_vadd(v, 1, xvecB, 1, &v, 1, vDSP_Length(hiddenSize))

        // LayerNorm: v = (v - mean) / sqrt(variance + eps) * xvecLnW + xvecLnB
        var mean: Float = 0.0
        var stdDev: Float = 0.0
        vDSP_normalize(v, 1, &v, 1, &mean, &stdDev, vDSP_Length(hiddenSize))
        // vDSP_normalize divides by population standard deviation
        vDSP_vmul(v, 1, xvecLnW, 1, &v, 1, vDSP_Length(hiddenSize))
        vDSP_vadd(v, 1, xvecLnB, 1, &v, 1, vDSP_Length(hiddenSize))
        #else
        for i in 0..<hiddenSize {
            var sum = xvecB[i]
            let rowOffset = i * speakerDim
            for j in 0..<speakerDim {
                sum += xvecW[rowOffset + j] * speakerEmb[j]
            }
            v[i] = sum
        }

        let mean = v.reduce(0.0, +) / Float(hiddenSize)
        let variance = v.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Float(hiddenSize)
        let invStd = 1.0 / sqrt(variance + xvecLnEps)

        for i in 0..<hiddenSize {
            v[i] = (v[i] - mean) * invStd * xvecLnW[i] + xvecLnB[i]
        }
        #endif

        return v
    }

    /// Embeds a 2D rows matrix `[T, 17]` into prompt embeddings `[T, 768]`.
    public func embedRows(rows: [[Int64]], anchor: [Float]? = nil, audioPadID: Int64 = 1024) -> [[Float]] {
        let T = rows.count
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: hiddenSize), count: T)

        for t in 0..<T {
            let row = rows[t]
            let textID = Int(row[0])
            var emb = textEmbedding(for: textID)

            for ch in 0..<numCodebooks {
                if ch + 1 < row.count {
                    let audioID = row[ch + 1]
                    if audioID != audioPadID {
                        let aEmb = audioEmbedding(codebook: ch, tokenID: Int(audioID))
                        for h in 0..<hiddenSize {
                            emb[h] += aEmb[h]
                        }
                    }
                }
            }

            if let anchor = anchor {
                for h in 0..<hiddenSize {
                    emb[h] += anchor[h]
                }
            }

            result[t] = emb
        }

        return result
    }

    /// Creates a mock weights container for deterministic testing.
    public static func mock() -> V3TurboWeights {
        let h = 768
        let textCount = 419 * h
        let audioCount = 16 * 1024 * h
        let xvecCount = 768 * 192

        return V3TurboWeights(
            textEmb: [Float](repeating: 0.05, count: textCount),
            audioEmb: [Float](repeating: 0.02, count: audioCount),
            xvecW: [Float](repeating: 0.01, count: xvecCount),
            xvecB: [Float](repeating: 0.0, count: h),
            xvecLnW: [Float](repeating: 1.0, count: h),
            xvecLnB: [Float](repeating: 0.0, count: h),
            xvecLnEps: 1e-5
        )
    }
}
