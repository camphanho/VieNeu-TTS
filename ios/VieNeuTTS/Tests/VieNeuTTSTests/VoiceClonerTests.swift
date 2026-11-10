import XCTest
@testable import VieNeuTTS

final class VoiceClonerTests: XCTestCase {
    private func makeSilentWav(duration: Float, sampleRate: Int = 24000) throws -> URL {
        let sampleCount = Int(Float(sampleRate) * duration)
        let samples = [Float](repeating: 0.0, count: sampleCount)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try WAVWriter.write(samples: samples, sampleRate: sampleRate, channels: 1, to: tempURL)
        return tempURL
    }

    private func makeToneWav(duration: Float, frequency: Float = 440.0, amplitude: Float = 0.5, sampleRate: Int = 24000, padSilence: Float = 0.0) throws -> URL {
        let totalSamples = Int(Float(sampleRate) * (duration + padSilence * 2.0))
        var samples = [Float](repeating: 0.0, count: totalSamples)
        let startSample = Int(Float(sampleRate) * padSilence)
        let toneSampleCount = Int(Float(sampleRate) * duration)
        for i in 0..<toneSampleCount {
            let t = Float(i) / Float(sampleRate)
            samples[startSample + i] = amplitude * sin(2.0 * .pi * frequency * t)
        }
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try WAVWriter.write(samples: samples, sampleRate: sampleRate, channels: 1, to: tempURL)
        return tempURL
    }

    func testVoiceProfileSerialization() throws {
        let profile = VoiceProfile(
            name: "TestSpeaker",
            speakerEmbedding: Array(repeating: 0.1, count: 192),
            styleTokens: Array(repeating: Array(repeating: 0.2, count: 256), count: 50)
        )
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("profile.json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try profile.export(to: tempURL)
        let loaded = try VoiceProfile.load(from: tempURL)
        XCTAssertEqual(loaded.name, "TestSpeaker")
        XCTAssertEqual(loaded.speakerEmbedding.count, 192)
        XCTAssertEqual(loaded.styleTokens.count, 50)
        XCTAssertEqual(loaded, profile)
    }

    func testTooShortOrSilentAudioThrows() async throws {
        // Review Focus #2: Reference audio < 1s or silent throws referenceAudioTooShortOrSilent
        let silentURL = try makeSilentWav(duration: 0.5)
        defer { try? FileManager.default.removeItem(at: silentURL) }

        let cloner = VoiceCloner(onnxEngine: MockONNXEngine())
        do {
            _ = try await cloner.clone(from: silentURL, name: "BadClip")
            XCTFail("Expected error on silent short audio")
        } catch let error as VieNeuError {
            guard case .referenceAudioTooShortOrSilent = error else {
                XCTFail("Expected .referenceAudioTooShortOrSilent error but got \(error)")
                return
            }
        }
    }

    func testLongSilentAudioThrows() async throws {
        // 2 seconds of pure silence should be trimmed down to 0 length and throw
        let silentURL = try makeSilentWav(duration: 2.0)
        defer { try? FileManager.default.removeItem(at: silentURL) }

        let cloner = VoiceCloner(onnxEngine: MockONNXEngine())
        do {
            _ = try await cloner.clone(from: silentURL, name: "LongSilent")
            XCTFail("Expected error on silent audio")
        } catch let error as VieNeuError {
            guard case .referenceAudioTooShortOrSilent = error else {
                XCTFail("Expected .referenceAudioTooShortOrSilent error but got \(error)")
                return
            }
        }
    }

    func testShortToneAudioThrows() async throws {
        // 0.5 seconds of non-silent tone is < 1.0 second, must throw
        let shortToneURL = try makeToneWav(duration: 0.5)
        defer { try? FileManager.default.removeItem(at: shortToneURL) }

        let cloner = VoiceCloner(onnxEngine: MockONNXEngine())
        do {
            _ = try await cloner.clone(from: shortToneURL, name: "ShortTone")
            XCTFail("Expected error on <1.0s audio")
        } catch let error as VieNeuError {
            guard case .referenceAudioTooShortOrSilent = error else {
                XCTFail("Expected .referenceAudioTooShortOrSilent error but got \(error)")
                return
            }
        }
    }

    func testSuccessfulVoiceCloning() async throws {
        // 2.0 seconds tone with 0.3s leading and trailing silence
        let toneURL = try makeToneWav(duration: 2.0, padSilence: 0.3)
        defer { try? FileManager.default.removeItem(at: toneURL) }

        let cloner = VoiceCloner(onnxEngine: MockONNXEngine())
        let profile = try await cloner.clone(from: toneURL, name: "ClonedSpeaker")
        XCTAssertEqual(profile.name, "ClonedSpeaker")
        XCTAssertEqual(profile.speakerEmbedding.count, 192)
        XCTAssertEqual(profile.styleTokens.count, 50)
        XCTAssertEqual(profile.styleTokens.first?.count, 256)
    }

    func testVoiceProfileMock() {
        let mock = VoiceProfile.mock(name: "CustomMock")
        XCTAssertEqual(mock.name, "CustomMock")
        XCTAssertEqual(mock.speakerEmbedding.count, 192)
        XCTAssertEqual(mock.styleTokens.count, 50)
        XCTAssertEqual(mock.styleTokens[0].count, 256)
    }

    func testSilenceTrimmingDirect() {
        let sampleRate = 24000
        let leadingSilence = [Float](repeating: 0.0, count: 12000) // 0.5s
        let tone = (0..<36000).map { sin(2.0 * Float.pi * 440.0 * Float($0) / Float(sampleRate)) * 0.5 } // 1.5s
        let trailingSilence = [Float](repeating: 0.0, count: 12000) // 0.5s
        let full = leadingSilence + tone + trailingSilence

        let trimmed = VoiceCloner.trimSilence(samples: full, sampleRate: sampleRate)
        let trimmedDuration = Float(trimmed.count) / Float(sampleRate)
        XCTAssertGreaterThanOrEqual(trimmedDuration, 1.45)
        XCTAssertLessThanOrEqual(trimmedDuration, 1.6)

        let rms = VoiceCloner.calculateRMS(trimmed)
        XCTAssertGreaterThan(rms, 1e-4)
    }

    func testLoadNonExistentProfileThrows() {
        let missingURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        XCTAssertThrowsError(try VoiceProfile.load(from: missingURL))
    }
}
