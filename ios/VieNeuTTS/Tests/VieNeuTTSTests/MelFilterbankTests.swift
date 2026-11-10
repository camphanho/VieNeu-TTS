import XCTest
@testable import VieNeuTTS

final class MelFilterbankTests: XCTestCase {
    func testComputes80BinMelFilterbank() throws {
        let fbank = MelFilterbank()
        let sampleCount = 24000 // 1 second of audio
        let sineWave = (0..<sampleCount).map { sin(2.0 * Float.pi * 440.0 * Float($0) / 24000.0) }
        let mel = try fbank.compute(samples: sineWave, sampleRate: 24000)
        XCTAssertFalse(mel.isEmpty)
        XCTAssertEqual(mel.count, 98)
        XCTAssertEqual(mel[0].count, 80) // 80 filter channels
    }

    func testMeanNormalizationAcrossTime() throws {
        let fbank = MelFilterbank()
        let sampleCount = 24000
        let sineWave = (0..<sampleCount).map { sin(2.0 * Float.pi * 440.0 * Float($0) / 24000.0) }
        let mel = try fbank.compute(samples: sineWave, sampleRate: 24000)

        // Across frames, each bin should have mean ~ 0
        for bin in 0..<80 {
            var sum: Float = 0
            for frame in 0..<mel.count {
                sum += mel[frame][bin]
            }
            let mean = sum / Float(mel.count)
            XCTAssertEqual(mean, 0.0, accuracy: 1e-4)
        }
    }

    func testAudioAt16kHzDirectPassthrough() throws {
        let fbank = MelFilterbank()
        let sampleCount = 16000 // 1 second @ 16kHz
        let sineWave = (0..<sampleCount).map { sin(2.0 * Float.pi * 440.0 * Float($0) / 16000.0) }
        let mel = try fbank.compute(samples: sineWave, sampleRate: 16000)
        XCTAssertEqual(mel.count, 98)
        XCTAssertEqual(mel[0].count, 80)
    }

    func testDownmixStereoAndResample() throws {
        // Review Focus #4: Stereo high-sample-rate audio resampled cleanly to 24kHz mono Float32
        let dummyWavURL = try makeTestStereoWav(sampleRate: 48000, duration: 1.0)
        defer { try? FileManager.default.removeItem(at: dummyWavURL) }
        let mono24k = try MelFilterbank.convertToMono24kHz(from: dummyWavURL)
        XCTAssertEqual(mono24k.count, 24000, accuracy: 240)
    }

    func testAudioTooShortThrows() {
        let fbank = MelFilterbank()
        let shortAudio = [Float](repeating: 0.1, count: 100)
        XCTAssertThrowsError(try fbank.compute(samples: shortAudio, sampleRate: 16000)) { error in
            guard let audioError = error as? AudioError, case .audioTooShort = audioError else {
                XCTFail("Expected .audioTooShort error")
                return
            }
        }
    }

    func testExactFrameCountCalculation() throws {
        let fbank = MelFilterbank()
        // 400 samples @ 16kHz = 1 frame
        let samples400 = (0..<400).map { Float(sin(2.0 * Double.pi * 440.0 * Double($0) / 16000.0)) }
        let mel1 = try fbank.compute(samples: samples400, sampleRate: 16000)
        XCTAssertEqual(mel1.count, 1)

        // 560 samples @ 16kHz = (560 - 400) / 160 + 1 = 2 frames
        let samples560 = (0..<560).map { Float(sin(2.0 * Double.pi * 440.0 * Double($0) / 16000.0)) }
        let mel2 = try fbank.compute(samples: samples560, sampleRate: 16000)
        XCTAssertEqual(mel2.count, 2)
    }

    func testResampleEdgeCases() {
        let empty = MelFilterbank.resample(samples: [], from: 16000, to: 24000)
        XCTAssertTrue(empty.isEmpty)

        let original: [Float] = [0.1, 0.5, -0.3]
        let sameRate = MelFilterbank.resample(samples: original, from: 24000, to: 24000)
        XCTAssertEqual(sameRate, original)
    }

    func testDownsamplingAttenuatesFrequenciesAboveTargetNyquist() {
        // 10 kHz tone at 24 kHz sample rate (> 8 kHz target Nyquist for 16 kHz)
        let sampleRate = 24000
        let count = 2400 // 100ms
        let highFreqTone = (0..<count).map { sin(2.0 * Float.pi * 10000.0 * Float($0) / Float(sampleRate)) }

        let downsampled = MelFilterbank.resample(samples: highFreqTone, from: 24000, to: 16000)
        XCTAssertEqual(downsampled.count, 1600)

        // Check steady-state max amplitude (skipping FIR filter warm-up/cool-down edges)
        let steadyState = downsampled[50..<(downsampled.count - 50)]
        let maxAmp = steadyState.map { abs($0) }.max() ?? 1.0
        XCTAssertLessThan(maxAmp, 0.15, "10kHz tone should be significantly attenuated by anti-aliasing filter when downsampling to 16kHz")
    }

    func testDeterministicComputation() throws {
        let fbank = MelFilterbank()
        let sampleCount = 8000 // 0.5s @ 16kHz
        let sineWave = (0..<sampleCount).map { sin(2.0 * Float.pi * 440.0 * Float($0) / 16000.0) }
        let mel1 = try fbank.compute(samples: sineWave, sampleRate: 16000)
        let mel2 = try fbank.compute(samples: sineWave, sampleRate: 16000)

        XCTAssertEqual(mel1.count, mel2.count)
        for i in 0..<mel1.count {
            for j in 0..<80 {
                XCTAssertEqual(mel1[i][j], mel2[i][j], accuracy: 1e-6)
            }
        }
    }

    func testConvertToMonoFromMono24kHz() throws {
        let sampleRate = 24000
        let frameCount = 2400
        let original = (0..<frameCount).map { Float(sin(2.0 * Double.pi * 440.0 * Double($0) / Double(sampleRate))) }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_mono_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try WAVWriter.write(samples: original, sampleRate: sampleRate, channels: 1, to: tempURL)

        let converted = try MelFilterbank.convertToMono24kHz(from: tempURL)
        XCTAssertEqual(converted.count, frameCount)
        for i in 0..<frameCount {
            XCTAssertEqual(converted[i], original[i], accuracy: 0.001)
        }
    }

    func testAudioErrorDescriptions() {
        let errors: [AudioError] = [
            .invalidWAVHeader("bad"),
            .unsupportedWAVFormat("raw"),
            .truncatedWAVData,
            .fileReadError("missing"),
            .audioTooShort(samples: 10, minimum: 400),
            .resamplingError("failed"),
            .invalidFormat("invalid")
        ]
        for err in errors {
            XCTAssertFalse(err.localizedDescription.isEmpty)
        }
        XCTAssertTrue(AudioError.invalidFormat("bad format").localizedDescription.contains("Invalid audio format"))
    }

    private func makeTestStereoWav(sampleRate: Int, duration: Double) throws -> URL {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_stereo_\(UUID().uuidString).wav")
        let totalFrames = Int(Double(sampleRate) * duration)
        var interleaved = [Float](repeating: 0, count: totalFrames * 2)
        for i in 0..<totalFrames {
            let t = Float(i) / Float(sampleRate)
            let ch0 = sin(2.0 * Float.pi * 440.0 * t)
            let ch1 = cos(2.0 * Float.pi * 440.0 * t)
            interleaved[i * 2] = ch0
            interleaved[i * 2 + 1] = ch1
        }
        let data = WAVWriter.encode(samples: interleaved, sampleRate: sampleRate, channels: 2)
        try data.write(to: tempURL)
        return tempURL
    }
}
