import XCTest
@testable import VieNeuTTS

final class VieNeuTTSSynthesisTests: XCTestCase {
    func testEmptyTextInputThrows() async throws {
        // Review Focus #1: Empty or whitespace input throws emptyInput error
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        do {
            _ = try await tts.synthesize(text: "    ", voice: voice)
            XCTFail("Expected emptyInput error")
        } catch let error as VieNeuError {
            guard case .emptyInput = error else {
                XCTFail("Expected .emptyInput error, got \(error)")
                return
            }
        }

        // Test empty string as well
        do {
            _ = try await tts.synthesize(text: "", voice: voice)
            XCTFail("Expected emptyInput error for empty string")
        } catch let error as VieNeuError {
            guard case .emptyInput = error else {
                XCTFail("Expected .emptyInput error, got \(error)")
                return
            }
        }

        // Test whitespace in streaming API
        do {
            for try await _ in tts.synthesizeStream(text: "   \t\n  ", voice: voice) {
                XCTFail("Stream should not yield any chunks on whitespace input")
            }
            XCTFail("Expected emptyInput error in stream")
        } catch let error as VieNeuError {
            guard case .emptyInput = error else {
                XCTFail("Expected .emptyInput error, got \(error)")
                return
            }
        }
    }

    func testCancellationStopsStreamExecution() async throws {
        // Review Focus #5: Mid-stream task cancellation immediately stops computation
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        let task = Task {
            for try await _ in tts.synthesizeStream(text: "Một đoạn văn dài gồm rất nhiều câu.", voice: voice) {
                // cancel immediately
                withUnsafeCurrentTask { $0?.cancel() }
                throw CancellationError()
            }
        }
        task.cancel()
        _ = await task.result
        XCTAssertTrue(task.isCancelled)
    }

    func testFullSynthesisConcatenatesAudio() async throws {
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        let text = "Xin chào Việt Nam. Đây là câu thứ hai."
        let audio = try await tts.synthesize(text: text, voice: voice)

        XCTAssertGreaterThan(audio.samples.count, 0)
        XCTAssertEqual(audio.sampleRate, 24000)
        XCTAssertGreaterThan(audio.duration, 0.0)
    }

    func testSynthesizeStreamYieldsChunks() async throws {
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())
        let voice = VoiceProfile.mock()
        let text = "Câu đầu tiên kết thúc ở đây.\nCâu thứ hai tiếp nối ngay sau đó."

        var chunks: [AudioChunk] = []
        for try await chunk in tts.synthesizeStream(text: text, voice: voice) {
            chunks.append(chunk)
        }

        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].index, 0)
        XCTAssertEqual(chunks[1].index, 1)
        XCTAssertGreaterThan(chunks[0].audio.samples.count, 0)
        XCTAssertGreaterThan(chunks[1].audio.samples.count, 0)
        XCTAssertGreaterThan(chunks[0].pauseAfter, 0)
    }

    func testLoadPresetVoice() async throws {
        let tts = try await VieNeuTTS(config: MockVieNeuConfig())

        let presetMinhQuan = try tts.loadPresetVoice(name: "Minh Quân")
        XCTAssertEqual(presetMinhQuan.name, "Minh Quân")
        XCTAssertEqual(presetMinhQuan.speakerEmbedding.count, 192)
        XCTAssertEqual(presetMinhQuan.styleTokens.count, 50)

        let mockVoice = try tts.loadPresetVoice(name: "MockVoice")
        XCTAssertEqual(mockVoice.name, "MockVoice")

        XCTAssertThrowsError(try tts.loadPresetVoice(name: "NonExistentVoiceName12345")) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
        }
    }

    func testV3TurboModeReports48kHzSampleRate() async throws {
        let config = VieNeuConfig.mock(mode: .v3turbo)
        let tts = try await VieNeuTTS(config: config)
        let audio = try await tts.synthesize(text: "Xin chào", voice: VoiceProfile.mock())
        XCTAssertEqual(audio.sampleRate, 48000)
    }
}
