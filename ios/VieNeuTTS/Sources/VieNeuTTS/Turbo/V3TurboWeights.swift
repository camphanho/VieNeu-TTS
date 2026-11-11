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
    public let textEmb: [Float]
    public let audioEmb: [Float]
    public let xvecW: [Float]
    public let xvecB: [Float]
    public let xvecLnW: [Float]
    public let xvecLnB: [Float]
    public let xvecLnEps: Float

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

    /// Computes acoustic logits for codebook `ch` given hidden representation `vec`:
    /// `logits = vec @ audio_emb[ch].T` (1024 floats)
    public func audioLogits(codebook ch: Int, vector vec: [Float]) -> [Float] {
        guard ch >= 0 && ch < numCodebooks, vec.count == hiddenSize else {
            return [Float](repeating: 0.0, count: audioVocabSize)
        }
        var logits = [Float](repeating: 0.0, count: audioVocabSize)
        let chOffset = ch * audioVocabSize * hiddenSize

        #if canImport(Accelerate)
        audioEmb.withUnsafeBufferPointer { aPtr in
            vec.withUnsafeBufferPointer { vPtr in
                logits.withUnsafeMutableBufferPointer { lPtr in
                    let matrixA = aPtr.baseAddress!.advanced(by: chOffset)
                    vDSP_mmul(
                        matrixA, 1,
                        vPtr.baseAddress!, 1,
                        lPtr.baseAddress!, 1,
                        vDSP_Length(audioVocabSize),
                        1,
                        vDSP_Length(hiddenSize)
                    )
                }
            }
        }
        #else
        for i in 0..<audioVocabSize {
            var sum: Float = 0.0
            let rowOffset = chOffset + i * hiddenSize
            for j in 0..<hiddenSize {
                sum += audioEmb[rowOffset + j] * vec[j]
            }
            logits[i] = sum
        }
        #endif

        return logits
    }

    /// Computes text logits for speech generation end (EOS) check:
    /// `text_logits = slot0 @ text_emb.T` (419 floats)
    public func textLogits(vector slot0: [Float]) -> [Float] {
        guard slot0.count == hiddenSize else {
            return [Float](repeating: 0.0, count: textVocabSize)
        }
        var logits = [Float](repeating: 0.0, count: textVocabSize)

        #if canImport(Accelerate)
        textEmb.withUnsafeBufferPointer { tPtr in
            slot0.withUnsafeBufferPointer { sPtr in
                logits.withUnsafeMutableBufferPointer { lPtr in
                    vDSP_mmul(
                        tPtr.baseAddress!, 1,
                        sPtr.baseAddress!, 1,
                        lPtr.baseAddress!, 1,
                        vDSP_Length(textVocabSize),
                        1,
                        vDSP_Length(hiddenSize)
                    )
                }
            }
        }
        #else
        for i in 0..<textVocabSize {
            var sum: Float = 0.0
            let rowOffset = i * hiddenSize
            for j in 0..<hiddenSize {
                sum += textEmb[rowOffset + j] * slot0[j]
            }
            logits[i] = sum
        }
        #endif

        return logits
    }

    /// Computes the 768-dimensional global speaker anchor from a 192-dimensional x-vector.
    public func speakerAnchor(for speakerEmb: [Float]) throws -> [Float] {
        guard speakerEmb.count == speakerDim, !speakerEmb.allSatisfy({ $0 == 0 }) else {
            throw VieNeuError.synthesisFailed("Invalid speaker embedding: must be 192-d non-zero vector.")
        }

        var v = [Float](repeating: 0.0, count: hiddenSize)

        #if canImport(Accelerate)
        // Matrix-vector multiplication: v = xvecW * speakerEmb + xvecB
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

    /// Parses uncompressed NumPy arrays directly from `vieneu_v3_heads.npz` (ZIP format).
    public static func parseNPZ(data: Data) -> V3TurboWeights? {
        guard data.count > 1000 else { return nil }

        var arrays: [String: [Float]] = [:]
        let total = data.count

        data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            guard let bytes = rawPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            var pos = 0
            while pos + 46 <= total {
                if bytes[pos] == 0x50 && bytes[pos + 1] == 0x4b && bytes[pos + 2] == 0x01 && bytes[pos + 3] == 0x02 {
                    let compSize = Int(readUInt32(ptr: bytes, offset: pos + 20))
                    let fnLen = Int(readUInt16(ptr: bytes, offset: pos + 28))
                    let extraLen = Int(readUInt16(ptr: bytes, offset: pos + 30))
                    let commentLen = Int(readUInt16(ptr: bytes, offset: pos + 32))
                    let localHdrOffset = Int(readUInt32(ptr: bytes, offset: pos + 42))

                    if pos + 46 + fnLen <= total {
                        let nameString = String(decoding: UnsafeBufferPointer(start: bytes.advanced(by: pos + 46), count: fnLen), as: UTF8.self).replacingOccurrences(of: ".npy", with: "")

                        if localHdrOffset + 30 <= total {
                            let locFnLen = Int(readUInt16(ptr: bytes, offset: localHdrOffset + 26))
                            let locExtraLen = Int(readUInt16(ptr: bytes, offset: localHdrOffset + 28))
                            let dataStart = localHdrOffset + 30 + locFnLen + locExtraLen

                            if dataStart + compSize <= total && compSize >= 16 {
                                // \x93NUMPY header check
                                if bytes[dataStart] == 0x93 && bytes[dataStart + 1] == 0x4e && bytes[dataStart + 2] == 0x55 &&
                                   bytes[dataStart + 3] == 0x4d && bytes[dataStart + 4] == 0x50 && bytes[dataStart + 5] == 0x59 {
                                    let hdrLen = Int(readUInt16(ptr: bytes, offset: dataStart + 8))
                                    let floatStart = dataStart + 10 + hdrLen
                                    let floatBytesCount = compSize - (10 + hdrLen)
                                    if floatBytesCount > 0 && floatStart + floatBytesCount <= total {
                                        let floatCount = floatBytesCount / 4
                                        var floats = [Float](repeating: 0.0, count: floatCount)
                                        let src = rawPtr.baseAddress!.advanced(by: floatStart)
                                        floats.withUnsafeMutableBytes { dst in
                                            dst.baseAddress!.copyMemory(from: src, byteCount: floatCount * 4)
                                        }
                                        arrays[nameString] = floats
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
        }

        guard let textEmb = arrays["text_emb"], textEmb.count == 419 * 768,
              let audioEmb = arrays["audio_emb"], audioEmb.count == 16 * 1024 * 768,
              let xvecW = arrays["xvec_w"], xvecW.count == 768 * 192,
              let xvecB = arrays["xvec_b"], xvecB.count == 768,
              let xvecLnW = arrays["xvec_ln_w"], xvecLnW.count == 768,
              let xvecLnB = arrays["xvec_ln_b"], xvecLnB.count == 768 else {
            return nil
        }

        let eps = arrays["xvec_ln_eps"]?.first ?? 1e-5

        return V3TurboWeights(
            textEmb: textEmb,
            audioEmb: audioEmb,
            xvecW: xvecW,
            xvecB: xvecB,
            xvecLnW: xvecLnW,
            xvecLnB: xvecLnB,
            xvecLnEps: eps
        )
    }

    private static func readUInt16(ptr: UnsafePointer<UInt8>, offset: Int) -> UInt16 {
        UInt16(ptr[offset]) | (UInt16(ptr[offset + 1]) << 8)
    }

    private static func readUInt32(ptr: UnsafePointer<UInt8>, offset: Int) -> UInt32 {
        UInt32(ptr[offset]) |
        (UInt32(ptr[offset + 1]) << 8) |
        (UInt32(ptr[offset + 2]) << 16) |
        (UInt32(ptr[offset + 3]) << 24)
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
