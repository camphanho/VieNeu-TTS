import Foundation

public enum AudioError: Error, LocalizedError, Equatable {
    case invalidWAVHeader(String)
    case unsupportedWAVFormat(String)
    case truncatedWAVData
    case fileReadError(String)
    case audioTooShort(samples: Int, minimum: Int)
    case resamplingError(String)
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .invalidWAVHeader(let msg):
            return "Invalid WAV header: \(msg)"
        case .unsupportedWAVFormat(let msg):
            return "Unsupported WAV format: \(msg)"
        case .truncatedWAVData:
            return "Truncated WAV data"
        case .fileReadError(let msg):
            return "File read error: \(msg)"
        case .audioTooShort(let samples, let minimum):
            return "Audio too short: \(samples) samples < minimum \(minimum)"
        case .resamplingError(let msg):
            return "Resampling error: \(msg)"
        case .invalidFormat(let msg):
            return "Invalid audio format: \(msg)"
        }
    }
}

public enum WAVWriter {
    /// Encodes float audio samples into a 16-bit PCM RIFF WAV container.
    public static func encode(
        samples: [Float],
        sampleRate: Int = 24000,
        channels: Int = 1
    ) -> Data {
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample / 8)
        let dataSize = UInt32(samples.count * bytesPerSample)
        let chunkSize = 36 + dataSize
        let byteRate = UInt32(sampleRate * channels * bytesPerSample)
        let blockAlign = UInt16(channels * bytesPerSample)

        var data = Data(capacity: 44 + Int(dataSize))

        // RIFF chunk descriptor
        data.append(contentsOf: "RIFF".utf8)
        data.append(UInt32(chunkSize).littleEndianData)
        data.append(contentsOf: "WAVE".utf8)

        // "fmt " sub-chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(UInt32(16).littleEndianData) // Subchunk1Size for PCM
        data.append(UInt16(1).littleEndianData)  // AudioFormat 1 = PCM
        data.append(UInt16(channels).littleEndianData)
        data.append(UInt32(sampleRate).littleEndianData)
        data.append(byteRate.littleEndianData)
        data.append(blockAlign.littleEndianData)
        data.append(bitsPerSample.littleEndianData)

        // "data" sub-chunk
        data.append(contentsOf: "data".utf8)
        data.append(dataSize.littleEndianData)

        // PCM 16-bit sample data
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample: Int16
            if clamped >= 0 {
                intSample = Int16(round(clamped * 32767.0))
            } else {
                intSample = Int16(round(clamped * 32768.0))
            }
            data.append(intSample.littleEndianData)
        }

        return data
    }

    /// Writes 16-bit PCM WAV to a file URL.
    public static func write(
        samples: [Float],
        sampleRate: Int = 24000,
        channels: Int = 1,
        to url: URL
    ) throws {
        let data = encode(samples: samples, sampleRate: sampleRate, channels: channels)
        try data.write(to: url, options: .atomic)
    }

    /// Decodes a WAV file from raw data into float samples.
    public static func decode(data: Data) throws -> (samples: [Float], sampleRate: Int, channels: Int) {
        guard data.count >= 44 else {
            throw AudioError.truncatedWAVData
        }

        guard let riff = String(data: data.subdata(in: 0..<4), encoding: .ascii), riff == "RIFF",
              let wave = String(data: data.subdata(in: 8..<12), encoding: .ascii), wave == "WAVE" else {
            throw AudioError.invalidWAVHeader("Missing RIFF or WAVE identifier")
        }

        var offset = 12
        var formatFound = false
        var audioFormat: UInt16 = 0
        var channels: UInt16 = 0
        var sampleRate: UInt32 = 0
        var bitsPerSample: UInt16 = 0

        var audioSamples: [Float]?

        while offset + 8 <= data.count {
            guard let chunkID = String(data: data.subdata(in: offset..<offset+4), encoding: .ascii) else {
                break
            }
            let chunkSize = Int(data.loadUInt32LE(at: offset + 4))
            offset += 8

            if chunkID == "fmt " {
                guard chunkSize >= 16, offset + 16 <= data.count else {
                    throw AudioError.invalidWAVHeader("Malformed fmt chunk")
                }
                audioFormat = data.loadUInt16LE(at: offset)
                channels = data.loadUInt16LE(at: offset + 2)
                sampleRate = data.loadUInt32LE(at: offset + 4)
                bitsPerSample = data.loadUInt16LE(at: offset + 14)
                formatFound = true
            } else if chunkID == "data" {
                guard formatFound else {
                    throw AudioError.invalidWAVHeader("data chunk appeared before fmt chunk")
                }
                let dataEnd = min(offset + chunkSize, data.count)
                let dataBytes = dataEnd - offset

                if audioFormat == 1 { // PCM
                    if bitsPerSample == 16 {
                        let sampleCount = dataBytes / 2
                        var samples = [Float](repeating: 0, count: sampleCount)
                        for i in 0..<sampleCount {
                            let raw = Int16(bitPattern: data.loadUInt16LE(at: offset + i * 2))
                            samples[i] = Float(raw) / 32768.0
                        }
                        audioSamples = samples
                    } else if bitsPerSample == 8 {
                        let sampleCount = dataBytes
                        var samples = [Float](repeating: 0, count: sampleCount)
                        for i in 0..<sampleCount {
                            let raw = data[offset + i]
                            samples[i] = (Float(raw) - 128.0) / 128.0
                        }
                        audioSamples = samples
                    } else if bitsPerSample == 32 {
                        let sampleCount = dataBytes / 4
                        var samples = [Float](repeating: 0, count: sampleCount)
                        for i in 0..<sampleCount {
                            let raw = Int32(bitPattern: data.loadUInt32LE(at: offset + i * 4))
                            samples[i] = Float(raw) / 2147483648.0
                        }
                        audioSamples = samples
                    } else {
                        throw AudioError.unsupportedWAVFormat("Unsupported PCM bit depth: \(bitsPerSample)")
                    }
                } else if audioFormat == 3 { // IEEE Float
                    if bitsPerSample == 32 {
                        let sampleCount = dataBytes / 4
                        var samples = [Float](repeating: 0, count: sampleCount)
                        for i in 0..<sampleCount {
                            let bitPattern = data.loadUInt32LE(at: offset + i * 4)
                            samples[i] = Float(bitPattern: bitPattern)
                        }
                        audioSamples = samples
                    } else {
                        throw AudioError.unsupportedWAVFormat("Unsupported float bit depth: \(bitsPerSample)")
                    }
                } else {
                    throw AudioError.unsupportedWAVFormat("Unsupported audio format code: \(audioFormat)")
                }
                break
            }

            offset += chunkSize
            // Word alignment (2-byte boundary)
            if chunkSize % 2 != 0 {
                offset += 1
            }
        }

        guard let samples = audioSamples, formatFound else {
            throw AudioError.invalidWAVHeader("No audio data chunk found in WAV")
        }

        return (samples, Int(sampleRate), Int(channels))
    }

    /// Reads and decodes a WAV file from a URL.
    public static func decode(from url: URL) throws -> (samples: [Float], sampleRate: Int, channels: Int) {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AudioError.fileReadError(error.localizedDescription)
        }
        return try decode(data: data)
    }
}

// MARK: - Binary Helper Extensions

private extension Data {
    func loadUInt16LE(at offset: Int) -> UInt16 {
        let b0 = UInt16(self[offset])
        let b1 = UInt16(self[offset + 1])
        return b0 | (b1 << 8)
    }

    func loadUInt32LE(at offset: Int) -> UInt32 {
        let b0 = UInt32(self[offset])
        let b1 = UInt32(self[offset + 1])
        let b2 = UInt32(self[offset + 2])
        let b3 = UInt32(self[offset + 3])
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }
}

private extension UInt16 {
    var littleEndianData: Data {
        var val = self.littleEndian
        return withUnsafeBytes(of: &val) { Data($0) }
    }
}

private extension UInt32 {
    var littleEndianData: Data {
        var val = self.littleEndian
        return withUnsafeBytes(of: &val) { Data($0) }
    }
}

private extension Int16 {
    var littleEndianData: Data {
        var val = self.littleEndian
        return withUnsafeBytes(of: &val) { Data($0) }
    }
}
