import XCTest
@testable import VieNeuTTS

final class AudioBufferTests: XCTestCase {
    func testAudioBufferInitialization() {
        let samples: [Float] = [0.1, -0.2, 0.5, 0.9]
        let buffer = AudioBuffer(samples: samples, sampleRate: 24000)

        XCTAssertEqual(buffer.samples, samples)
        XCTAssertEqual(buffer.sampleRate, 24000)
        XCTAssertEqual(buffer.duration, 4.0 / 24000.0, accuracy: 1e-6)
    }

    func testAudioBufferDefaultSampleRateIs24kHz() {
        let buffer = AudioBuffer(samples: [0.0, 0.1])
        XCTAssertEqual(buffer.sampleRate, 24000)
    }

    func testAudioBufferDurationZeroWhenEmpty() {
        let buffer = AudioBuffer(samples: [], sampleRate: 24000)
        XCTAssertEqual(buffer.duration, 0.0)
    }

    func testWriteAndReadWAVRoundTrip() throws {
        let sampleRate = 24000
        let count = 2400 // 0.1s
        let samples = (0..<count).map { Float(sin(2.0 * Double.pi * 440.0 * Double($0) / Double(sampleRate))) }
        let buffer = AudioBuffer(samples: samples, sampleRate: sampleRate)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_audio_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try buffer.writeWAV(to: tempURL)

        let decoded = try WAVWriter.decode(from: tempURL)
        XCTAssertEqual(decoded.sampleRate, sampleRate)
        XCTAssertEqual(decoded.channels, 1)
        XCTAssertEqual(decoded.samples.count, samples.count)

        // Verify quantization roundtrip within 16-bit PCM precision (~1/32767)
        for i in 0..<samples.count {
            XCTAssertEqual(decoded.samples[i], samples[i], accuracy: 0.001)
        }
    }

    func testWAVWriterStereoEncodingAndDecoding() throws {
        let sampleRate = 48000
        let frameCount = 1000
        var interleaved = [Float](repeating: 0, count: frameCount * 2)
        for i in 0..<frameCount {
            interleaved[i * 2] = Float(i) / Float(frameCount)
            interleaved[i * 2 + 1] = -Float(i) / Float(frameCount)
        }

        let wavData = WAVWriter.encode(samples: interleaved, sampleRate: sampleRate, channels: 2)
        let decoded = try WAVWriter.decode(data: wavData)

        XCTAssertEqual(decoded.sampleRate, sampleRate)
        XCTAssertEqual(decoded.channels, 2)
        XCTAssertEqual(decoded.samples.count, interleaved.count)

        for i in 0..<interleaved.count {
            XCTAssertEqual(decoded.samples[i], interleaved[i], accuracy: 0.001)
        }
    }

    func testWAVWriterThrowsOnInvalidData() {
        let invalidData = Data([0x00, 0x01, 0x02, 0x03])
        XCTAssertThrowsError(try WAVWriter.decode(data: invalidData)) { error in
            guard let audioError = error as? AudioError, case .truncatedWAVData = audioError else {
                XCTFail("Expected .truncatedWAVData error")
                return
            }
        }
    }

    func testAudioBufferEquality() {
        let buf1 = AudioBuffer(samples: [0.1, 0.2], sampleRate: 24000)
        let buf2 = AudioBuffer(samples: [0.1, 0.2], sampleRate: 24000)
        let buf3 = AudioBuffer(samples: [0.1, 0.3], sampleRate: 24000)
        let buf4 = AudioBuffer(samples: [0.1, 0.2], sampleRate: 16000)

        XCTAssertEqual(buf1, buf2)
        XCTAssertNotEqual(buf1, buf3)
        XCTAssertNotEqual(buf1, buf4)
    }

    func testWAVDataGeneration() {
        let buffer = AudioBuffer(samples: [0.5, -0.5], sampleRate: 24000)
        let data = buffer.wavData()
        XCTAssertGreaterThanOrEqual(data.count, 44)
    }

    func testWAVWriterDecodeNonExistentFileThrows() {
        let fakeURL = URL(fileURLWithPath: "/non/existent/path/audio.wav")
        XCTAssertThrowsError(try WAVWriter.decode(from: fakeURL)) { error in
            guard let audioError = error as? AudioError, case .fileReadError = audioError else {
                XCTFail("Expected .fileReadError")
                return
            }
        }
    }

    func testWAVWriterCorruptRIFFHeaderThrows() {
        var dummyData = Data(repeating: 0, count: 50)
        // Corrupt header without RIFF / WAVE
        dummyData.replaceSubrange(0..<4, with: "NOPE".utf8)
        dummyData.replaceSubrange(8..<12, with: "NOPE".utf8)
        XCTAssertThrowsError(try WAVWriter.decode(data: dummyData)) { error in
            guard let audioError = error as? AudioError, case .invalidWAVHeader = audioError else {
                XCTFail("Expected .invalidWAVHeader")
                return
            }
        }
    }

    #if canImport(AVFoundation)
    func testMakeAVAudioPCMBuffer() {
        let buffer = AudioBuffer(samples: [0.1, 0.2, 0.3], sampleRate: 24000)
        let pcm = buffer.makeAVAudioPCMBuffer()
        XCTAssertNotNil(pcm)
        XCTAssertEqual(pcm?.frameLength, 3)
        XCTAssertEqual(pcm?.format.sampleRate, 24000.0)
    }
    #endif
}
