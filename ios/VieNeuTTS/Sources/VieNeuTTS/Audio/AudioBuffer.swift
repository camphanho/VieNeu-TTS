import Foundation

#if canImport(AVFoundation)
import AVFoundation
#endif

/// Represents an audio buffer containing mono 32-bit float samples.
public struct AudioBuffer: Sendable, Equatable {
    /// Mono 32-bit floating point audio samples in the range `[-1.0, 1.0]`.
    public let samples: [Float]

    /// Sample rate in Hertz (default 24,000 Hz for VieNeu-TTS v3 Nano).
    public let sampleRate: Int

    /// Duration in seconds.
    public var duration: TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }

    /// Creates a new audio buffer.
    public init(samples: [Float], sampleRate: Int = 24000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    /// Encodes the buffer as a 16-bit PCM WAV Data container.
    public func wavData() -> Data {
        WAVWriter.encode(samples: samples, sampleRate: sampleRate, channels: 1)
    }

    /// Writes the buffer to a WAV file at the specified URL.
    public func writeWAV(to url: URL) throws {
        try WAVWriter.write(samples: samples, sampleRate: sampleRate, channels: 1, to: url)
    }
}

#if canImport(AVFoundation)
public extension AudioBuffer {
    /// Creates an `AVAudioPCMBuffer` matching the sample rate and float samples.
    func makeAVAudioPCMBuffer() -> AVAudioPCMBuffer? {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else { return nil }

        let frameCount = AVAudioFrameCount(samples.count)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(1, frameCount)) else {
            return nil
        }
        pcmBuffer.frameLength = frameCount
        if frameCount > 0, let channelData = pcmBuffer.floatChannelData {
            samples.withUnsafeBufferPointer { ptr in
                guard let baseAddress = ptr.baseAddress else { return }
                channelData[0].initialize(from: baseAddress, count: samples.count)
            }
        }
        return pcmBuffer
    }
}
#endif

/// Represents an audio chunk emitted during streaming speech synthesis.
public struct AudioChunk: Sendable, Equatable {
    public let index: Int
    public let text: String
    public let audio: AudioBuffer
    public let pauseAfter: Float

    public init(index: Int, text: String, audio: AudioBuffer, pauseAfter: Float = 0.2) {
        self.index = index
        self.text = text
        self.audio = audio
        self.pauseAfter = pauseAfter
    }
}
