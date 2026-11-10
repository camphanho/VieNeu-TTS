import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

#if canImport(AVFoundation)
import AVFoundation
#endif

/// Computes Kaldi-compatible 80-bin log-mel filterbanks for speaker encoder.
public final class MelFilterbank: Sendable {
    /// Structure holding a single triangular mel filter's sparse weights.
    struct TriangularFilter: Sendable {
        let firstBin: Int
        let weights: [Float]
    }

    private let poveyWindow: [Float]
    private let filters: [TriangularFilter]
    private let fft: FFT512

    public init() {
        let frameLength = 400
        let fftSize = 512
        let numBins = 80
        let lowFreq: Float = 20.0
        let highFreq: Float = 8000.0
        let sampleRate: Float = 16000.0

        // 1. Precompute Povey window: (0.5 - 0.5 * cos(2 * pi * i / (N - 1)))^0.85
        let a = 2.0 * Float.pi / Float(frameLength - 1)
        var window = [Float](repeating: 0, count: frameLength)
        for i in 0..<frameLength {
            let base = 0.5 - 0.5 * cos(a * Float(i))
            window[i] = pow(base, 0.85)
        }
        self.poveyWindow = window

        // 2. Precompute 80 Kaldi mel filter weights
        func melScale(_ freq: Float) -> Float {
            1127.0 * log(1.0 + freq / 700.0)
        }

        let melLow = melScale(lowFreq)
        let melHigh = melScale(highFreq)
        let melDelta = (melHigh - melLow) / Float(numBins + 1)
        let fftBinWidth = sampleRate / Float(fftSize) // 31.25 Hz

        var precomputedFilters = [TriangularFilter]()
        precomputedFilters.reserveCapacity(numBins)

        for i in 0..<numBins {
            let leftMel = melLow + Float(i) * melDelta
            let centerMel = leftMel + melDelta
            let rightMel = centerMel + melDelta

            var firstBin: Int?
            var weights = [Float]()

            for k in 0...256 {
                let freq = Float(k) * fftBinWidth
                let mel = melScale(freq)
                var weight: Float = 0
                if mel >= leftMel && mel <= centerMel {
                    weight = (mel - leftMel) / melDelta
                } else if mel > centerMel && mel <= rightMel {
                    weight = (rightMel - mel) / melDelta
                }

                if weight > 0 {
                    if firstBin == nil {
                        firstBin = k
                    }
                    weights.append(weight)
                }
            }

            precomputedFilters.append(
                TriangularFilter(
                    firstBin: firstBin ?? 0,
                    weights: weights
                )
            )
        }
        self.filters = precomputedFilters
        self.fft = FFT512()
    }

    /// Computes 80-channel log-mel filterbanks with Kaldi compliance.
    /// - Parameters:
    ///   - samples: 1D mono audio samples.
    ///   - sampleRate: Input sample rate in Hz. If not 16000, automatically resampled to 16000 Hz.
    /// - Returns: Array of frames with shape `[numFrames, 80]`, mean-normalized across time.
    public func compute(samples: [Float], sampleRate: Int) throws -> [[Float]] {
        let inputSamples: [Float]
        if sampleRate != 16000 {
            inputSamples = Self.resample(samples: samples, from: sampleRate, to: 16000)
        } else {
            inputSamples = samples
        }

        let frameLength = 400 // 25ms @ 16kHz
        let frameShift = 160  // 10ms @ 16kHz

        guard inputSamples.count >= frameLength else {
            throw AudioError.audioTooShort(samples: inputSamples.count, minimum: frameLength)
        }

        let numFrames = (inputSamples.count - frameLength) / frameShift + 1
        guard numFrames > 0 else {
            throw AudioError.audioTooShort(samples: inputSamples.count, minimum: frameLength)
        }

        var melFrames = [[Float]]()
        melFrames.reserveCapacity(numFrames)

        var realBuffer = [Float](repeating: 0, count: 512)
        var imagBuffer = [Float](repeating: 0, count: 512)
        var powerSpectrum = [Float](repeating: 0, count: 257)

        for frameIdx in 0..<numFrames {
            let offset = frameIdx * frameShift
            var frame = Array(inputSamples[offset..<offset + frameLength])

            // 1. Remove DC offset
            #if canImport(Accelerate)
            var mean: Float = 0
            vDSP_meanv(frame, 1, &mean, vDSP_Length(frameLength))
            var negMean = -mean
            vDSP_vsadd(frame, 1, &negMean, &frame, 1, vDSP_Length(frameLength))
            #else
            let mean = frame.reduce(0, +) / Float(frameLength)
            for i in 0..<frameLength {
                frame[i] -= mean
            }
            #endif

            // 2. Pre-emphasis (coeff = 0.97)
            for i in (1..<frameLength).reversed() {
                frame[i] -= 0.97 * frame[i - 1]
            }
            frame[0] -= 0.97 * frame[0]

            // 3. Windowing (Povey window)
            #if canImport(Accelerate)
            var windowed = [Float](repeating: 0, count: frameLength)
            vDSP_vmul(frame, 1, poveyWindow, 1, &windowed, 1, vDSP_Length(frameLength))
            #else
            var windowed = [Float](repeating: 0, count: frameLength)
            for i in 0..<frameLength {
                windowed[i] = frame[i] * poveyWindow[i]
            }
            #endif

            // 4. Pad to 512 and Real FFT
            realBuffer.withUnsafeMutableBufferPointer { rPtr in
                rPtr.initialize(repeating: 0)
                windowed.withUnsafeBufferPointer { wPtr in
                    rPtr.baseAddress!.initialize(from: wPtr.baseAddress!, count: frameLength)
                }
            }
            imagBuffer.withUnsafeMutableBufferPointer { iPtr in
                iPtr.initialize(repeating: 0)
            }

            fft.forward(real: &realBuffer, imag: &imagBuffer)

            // 5. Power spectrum for bins 0...256
            #if canImport(Accelerate)
            realBuffer.withUnsafeMutableBufferPointer { rPtr in
                imagBuffer.withUnsafeMutableBufferPointer { iPtr in
                    var splitComplex = DSPSplitComplex(
                        realp: rPtr.baseAddress!,
                        imagp: iPtr.baseAddress!
                    )
                    var magnitudes = [Float](repeating: 0, count: 257)
                    magnitudes.withUnsafeMutableBufferPointer { mPtr in
                        powerSpectrum.withUnsafeMutableBufferPointer { pPtr in
                            vDSP_zvabs(&splitComplex, 1, mPtr.baseAddress!, 1, 257)
                            vDSP_vsq(mPtr.baseAddress!, 1, pPtr.baseAddress!, 1, 257)
                        }
                    }
                }
            }
            #else
            for k in 0...256 {
                let r = realBuffer[k]
                let im = imagBuffer[k]
                powerSpectrum[k] = r * r + im * im
            }
            #endif

            // 6. Mel filterbank energies
            var logMel = [Float](repeating: 0, count: 80)
            let eps: Float = 1.1920929e-07

            for (i, filter) in filters.enumerated() {
                var energy: Float = 0
                #if canImport(Accelerate)
                powerSpectrum.withUnsafeBufferPointer { pPtr in
                    let start = pPtr.baseAddress! + filter.firstBin
                    vDSP_dotpr(start, 1, filter.weights, 1, &energy, vDSP_Length(filter.weights.count))
                }
                #else
                for (j, w) in filter.weights.enumerated() {
                    energy += w * powerSpectrum[filter.firstBin + j]
                }
                #endif
                logMel[i] = log(max(energy, eps))
            }

            melFrames.append(logMel)
        }

        // 7. Mean normalization across time
        for bin in 0..<80 {
            var sum: Float = 0
            for t in 0..<numFrames {
                sum += melFrames[t][bin]
            }
            let mean = sum / Float(numFrames)
            for t in 0..<numFrames {
                melFrames[t][bin] -= mean
            }
        }

        return melFrames
    }

    /// Converts an audio file at the given URL to mono 24,000 Hz Float32 samples.
    public static func convertToMono24kHz(from url: URL) throws -> [Float] {
        #if canImport(AVFoundation)
        if let samples = try? convertViaAVFoundation(from: url) {
            return samples
        }
        #endif
        return try convertViaWAVReader(from: url)
    }

    #if canImport(AVFoundation)
    private static func convertViaAVFoundation(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 24000.0,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioError.invalidFormat("Failed to create target audio format")
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw AudioError.invalidFormat("Failed to create AVAudioConverter")
        }

        let ratio = 24000.0 / sourceFormat.sampleRate
        let targetFrames = AVAudioFrameCount(ceil(Double(file.length) * ratio)) + 1024
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: targetFrames) else {
            throw AudioError.invalidFormat("Failed to allocate AVAudioPCMBuffer")
        }

        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            guard let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: inNumPackets) else {
                outStatus.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: buffer)
                if buffer.frameLength == 0 {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return buffer
            } catch {
                outStatus.pointee = .endOfStream
                return nil
            }
        }

        converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        if let error = error {
            throw error
        }

        guard let channelData = outputBuffer.floatChannelData else {
            throw AudioError.invalidFormat("No channel data produced by AVAudioConverter")
        }

        let count = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: count))
    }
    #endif

    private static func convertViaWAVReader(from url: URL) throws -> [Float] {
        let (rawSamples, sampleRate, channels) = try WAVWriter.decode(from: url)
        guard channels > 0, !rawSamples.isEmpty else {
            return []
        }

        let frameCount = rawSamples.count / channels
        var mono = [Float](repeating: 0, count: frameCount)

        if channels == 1 {
            mono = rawSamples
        } else {
            let invChannels = 1.0 / Float(channels)
            for i in 0..<frameCount {
                var sum: Float = 0
                for c in 0..<channels {
                    sum += rawSamples[i * channels + c]
                }
                mono[i] = sum * invChannels
            }
        }

        if sampleRate == 24000 {
            return mono
        } else {
            return resample(samples: mono, from: sampleRate, to: 24000)
        }
    }

    /// Resamples mono float audio from `inRate` to `outRate`.
    /// When downsampling (`inRate > outRate`), applies an anti-aliasing low-pass FIR filter
    /// before decimation to prevent high frequencies from folding over into the Nyquist band.
    public static func resample(samples: [Float], from inRate: Int, to outRate: Int) -> [Float] {
        guard inRate != outRate, !samples.isEmpty else { return samples }

        #if canImport(AVFoundation)
        if let converted = resampleViaAVAudioConverter(samples: samples, from: inRate, to: outRate) {
            return converted
        }
        #endif

        let filtered: [Float]
        if inRate > outRate {
            filtered = applyAntiAliasingFilter(samples: samples, inRate: inRate, outRate: outRate)
        } else {
            filtered = samples
        }

        let ratio = Double(inRate) / Double(outRate)
        let outCount = Int(round(Double(samples.count) / ratio))
        guard outCount > 0 else { return [] }

        var output = [Float](repeating: 0, count: outCount)
        let maxInIndex = filtered.count - 1

        for i in 0..<outCount {
            let inPos = Double(i) * ratio
            let idx = Int(inPos)
            let frac = Float(inPos - Double(idx))
            if idx >= maxInIndex {
                output[i] = filtered[maxInIndex]
            } else {
                output[i] = filtered[idx] * (1.0 - frac) + filtered[idx + 1] * frac
            }
        }
        return output
    }

    #if canImport(AVFoundation)
    private static func resampleViaAVAudioConverter(samples: [Float], from inRate: Int, to outRate: Int) -> [Float]? {
        guard let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(inRate), channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(outRate), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            return nil
        }

        let ratio = Double(outRate) / Double(inRate)
        let outFrameCapacity = AVAudioFrameCount(ceil(Double(samples.count) * ratio)) + 64
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outFrameCapacity) else {
            return nil
        }

        var hasProvidedData = false
        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            if hasProvidedData {
                outStatus.pointee = .endOfStream
                return nil
            }
            guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(samples.count)) else {
                outStatus.pointee = .endOfStream
                return nil
            }
            inBuffer.frameLength = AVAudioFrameCount(samples.count)
            if let channelData = inBuffer.floatChannelData {
                samples.withUnsafeBufferPointer { ptr in
                    guard let base = ptr.baseAddress else { return }
                    channelData[0].initialize(from: base, count: samples.count)
                }
            }
            hasProvidedData = true
            outStatus.pointee = .haveData
            return inBuffer
        }

        converter.convert(to: outBuffer, error: &error, withInputFrom: inputBlock)
        if error != nil { return nil }
        guard let channelData = outBuffer.floatChannelData else { return nil }
        let outCount = Int(outBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: outCount))
    }
    #endif

    /// Symmetric 31-tap Type-I linear-phase FIR low-pass filter with Hamming window.
    private static func applyAntiAliasingFilter(samples: [Float], inRate: Int, outRate: Int) -> [Float] {
        guard inRate > outRate, !samples.isEmpty else { return samples }
        let numTaps = 31
        let m = numTaps - 1
        let halfM = Float(m) / 2.0
        let fc = Float(outRate) * 0.475 / Float(inRate)

        var h = [Float](repeating: 0, count: numTaps)
        var sum: Float = 0
        for k in 0..<numTaps {
            let diff = Float(k) - halfM
            let sinc: Float
            if abs(diff) < 1e-6 {
                sinc = 2.0 * fc
            } else {
                sinc = sin(2.0 * Float.pi * fc * diff) / (Float.pi * diff)
            }
            let win = 0.54 - 0.46 * cos(2.0 * Float.pi * Float(k) / Float(m))
            let tap = sinc * win
            h[k] = tap
            sum += tap
        }
        guard sum > 0 else { return samples }
        for k in 0..<numTaps { h[k] /= sum }

        let count = samples.count
        var filtered = [Float](repeating: 0, count: count)
        let halfInt = numTaps / 2

        for n in 0..<count {
            var acc: Float = 0
            for k in 0..<numTaps {
                let idx = min(max(n + k - halfInt, 0), count - 1)
                acc += h[k] * samples[idx]
            }
            filtered[n] = acc
        }
        return filtered
    }
}

// MARK: - Internal 512-Point FFT Engine

final class FFT512: @unchecked Sendable {
    private let bitReversal: [Int]
    private let twiddleReal: [[Float]]
    private let twiddleImag: [[Float]]

    init() {
        var rev = [Int](repeating: 0, count: 512)
        var j = 0
        for i in 0..<511 {
            rev[i] = j
            var k = 256
            while k <= j {
                j -= k
                k >>= 1
            }
            j += k
        }
        rev[511] = 511
        self.bitReversal = rev

        var tR = [[Float]]()
        var tI = [[Float]]()
        var len = 2
        while len <= 512 {
            let half = len / 2
            var r = [Float](repeating: 0, count: half)
            var im = [Float](repeating: 0, count: half)
            for k in 0..<half {
                let angle = -2.0 * Float.pi * Float(k) / Float(len)
                r[k] = cos(angle)
                im[k] = sin(angle)
            }
            tR.append(r)
            tI.append(im)
            len <<= 1
        }
        self.twiddleReal = tR
        self.twiddleImag = tI
    }

    func forward(real: inout [Float], imag: inout [Float]) {
        for i in 0..<512 {
            let j = bitReversal[i]
            if i < j {
                real.swapAt(i, j)
                imag.swapAt(i, j)
            }
        }

        var stage = 0
        var len = 2
        while len <= 512 {
            let half = len / 2
            let tr = twiddleReal[stage]
            let ti = twiddleImag[stage]
            var i = 0
            while i < 512 {
                for k in 0..<half {
                    let wReal = tr[k]
                    let wImag = ti[k]
                    let uReal = real[i + k]
                    let uImag = imag[i + k]
                    let vReal = real[i + k + half] * wReal - imag[i + k + half] * wImag
                    let vImag = real[i + k + half] * wImag + imag[i + k + half] * wReal

                    real[i + k] = uReal + vReal
                    imag[i + k] = uImag + vImag
                    real[i + k + half] = uReal - vReal
                    imag[i + k + half] = uImag - vImag
                }
                i += len
            }
            len <<= 1
            stage += 1
        }
    }
}
