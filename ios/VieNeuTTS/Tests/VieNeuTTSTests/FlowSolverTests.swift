import XCTest
@testable import VieNeuTTS

final class FlowSolverTests: XCTestCase {
    func testEulerIntegrationStepsShapeAndValues() throws {
        let steps = 4
        let dim = 8
        let frames = 10
        let initialNoise = (0..<frames).map { _ in (0..<dim).map { _ in Float.random(in: -1...1) } }

        // Test with mock vector field: dX/dt = -X
        let trajectory = try FlowSolver.solveEuler(
            initialLatents: initialNoise,
            steps: steps,
            cfgScale: 1.0,
            vectorEstimator: { x, t in x.map { row in row.map { -$0 } } }
        )
        XCTAssertEqual(trajectory.count, frames)
        XCTAssertEqual(trajectory[0].count, dim)
    }

    func testEulerDecayTowardsZero() throws {
        let steps = 10
        let dim = 4
        let frames = 4
        let initialVal: Float = 1.0
        let initial = Array(repeating: Array(repeating: initialVal, count: dim), count: frames)

        // dX/dt = -X
        let result = try FlowSolver.solveEuler(
            initialLatents: initial,
            steps: steps,
            cfgScale: 0.0,
            sway: 0.0,
            vectorEstimator: { x, _ in x.map { row in row.map { -$0 } } }
        )

        // (1 - 1/10)^10 ≈ 0.348678
        for row in result {
            for val in row {
                XCTAssertLessThan(val, 0.5)
                XCTAssertGreaterThan(val, 0.2)
            }
        }
    }

    func testCFGExtrapolation() throws {
        let steps = 1
        let initial = [[0.0 as Float]]

        // Cond gives +2.0, Uncond gives +1.0
        // v = uncond + cfg * (cond - uncond) = 1.0 + 3.0 * (2.0 - 1.0) = 4.0
        // dt = 1.0, x = 0 + 1.0 * 4.0 = 4.0
        let result = try FlowSolver.solveEuler(
            initialLatents: initial,
            steps: steps,
            cfgScale: 3.0,
            sway: 0.0,
            uncondEstimator: { _, _ in [[1.0 as Float]] },
            vectorEstimator: { _, _ in [[2.0 as Float]] }
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].count, 1)
        XCTAssertEqual(result[0][0], 4.0, accuracy: 1e-5)
    }

    func testSwayScheduleValues() throws {
        // Test sway formula across steps
        let steps = 4
        let initial = [[0.0 as Float]]
        var observedTimes: [Float] = []

        _ = try FlowSolver.solveEuler(
            initialLatents: initial,
            steps: steps,
            cfgScale: 0.0,
            sway: -1.0,
            vectorEstimator: { _, t in
                observedTimes.append(t)
                return [[1.0 as Float]]
            }
        )

        XCTAssertEqual(observedTimes.count, steps)
        // t = 0 at start
        XCTAssertEqual(observedTimes[0], 0.0, accuracy: 1e-5)

        // All times should be monotonically increasing in [0, 1)
        for i in 1..<observedTimes.count {
            XCTAssertGreaterThan(observedTimes[i], observedTimes[i - 1])
            XCTAssertLessThan(observedTimes[i], 1.0)
        }
    }

    func testTaskCancellationThrowsImmediately() async throws {
        let initial = [[1.0 as Float]]
        let task = Task {
            try await FlowSolver.solveEuler(
                initialLatents: initial,
                steps: 500,
                vectorEstimator: { _, _ in
                    try await Task.sleep(nanoseconds: 10_000_000)
                    return [[0.0 as Float]]
                }
            )
        }

        task.cancel()

        do {
            _ = try await task.value
            XCTFail("Expected CancellationError to be thrown")
        } catch is CancellationError {
            // Expected
        } catch {
            XCTFail("Expected CancellationError but got \(error)")
        }
    }

    func testEmptyLatentsReturnsEmpty() throws {
        let result = try FlowSolver.solveEuler(
            initialLatents: [],
            steps: 4,
            vectorEstimator: { x, _ in x }
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testMockONNXEngineOutputs() async throws {
        let engine = MockONNXEngine()
        let style = Array(repeating: Array(repeating: Float(0.2), count: 256), count: 50)
        let speaker = Array(repeating: Float(0.1), count: 192)
        let (ctx, mask) = try await engine.runTextEncoder(tokenIDs: [1, 2, 3], style: style)
        XCTAssertEqual(ctx.count, 3)
        XCTAssertEqual(ctx[0].count, 256)
        XCTAssertEqual(mask.count, 3)

        let frames = try await engine.runDurationPredictor(
            ctx: ctx,
            mask: mask,
            speakerEmbedding: speaker,
            speed: 1.0
        )
        XCTAssertGreaterThanOrEqual(frames, 2)

        let vector = try await engine.runVectorEstimator(
            x: [[1.0, 2.0]],
            t: 0.5,
            ctx: ctx,
            mask: mask,
            speakerEmbedding: speaker,
            style: style
        )
        XCTAssertEqual(vector.count, 1)
        XCTAssertEqual(vector[0].count, 2)

        let audio = try await engine.runCodecDecoder(latents: [[0.1, 0.2]])
        XCTAssertFalse(audio.isEmpty)

        let spk = try await engine.runSpeakerEncoder(melFilterbank: [[0.1]])
        XCTAssertEqual(spk.count, 192)

        let codec = try await engine.runCodecEncoder(audio: [0.1, 0.2])
        XCTAssertEqual(codec.count, 6)
        XCTAssertEqual(codec[0].count, 24)

        let encodedStyle = try await engine.runReferenceEncoder(refCodes: codec)
        XCTAssertEqual(encodedStyle.count, 50)
        XCTAssertEqual(encodedStyle[0].count, 256)
    }

    func testDurationFramesConvertsScalarLogSecondsUsingNanoFormula() {
        XCTAssertEqual(ONNXEngine.durationFrames(logSeconds: 0.0, speed: 1.0), 16)
        XCTAssertEqual(ONNXEngine.durationFrames(logSeconds: log(2.0), speed: 2.0), 16)
        XCTAssertEqual(ONNXEngine.durationFrames(logSeconds: log(100.0), speed: 1.0), 234)
        XCTAssertEqual(ONNXEngine.durationFrames(logSeconds: -100.0, speed: 1.0), 2)
    }

    func testTextEncoderAndVectorEstimatorPreserveVoiceConditioning() async throws {
        let engine = ConditioningRecordingEngine()
        let voice = VoiceProfile(
            name: "Recorded",
            speakerEmbedding: Array(repeating: 0.75, count: 192),
            styleTokens: Array(repeating: Array(repeating: 0.5, count: 256), count: 50)
        )
        let tts = try await VieNeuTTS(config: MockVieNeuConfig(eulerSteps: 1, cfgScale: 1.0, onnxEngineOverride: engine))

        _ = try await tts.synthesize(text: "xin chào", voice: voice)

        let recording = await engine.recording
        XCTAssertEqual(recording.textStyles.first, voice.styleTokens)
        XCTAssertEqual(recording.durationSpeakerEmbeddings.first, voice.speakerEmbedding)
        XCTAssertEqual(recording.vectorSpeakerEmbeddings.first, voice.speakerEmbedding)
        XCTAssertEqual(recording.vectorStyles.first, voice.styleTokens)
        XCTAssertEqual(recording.unconditionalCalls, 1)
    }

    private actor ConditioningRecordingEngine: ONNXInferenceEngine {
        struct Recording: Sendable {
            var textStyles = [[[Float]]]()
            var durationSpeakerEmbeddings = [[Float]]()
            var vectorSpeakerEmbeddings = [[Float]]()
            var vectorStyles = [[[Float]]]()
            var unconditionalCalls = 0
        }

        private(set) var recording = Recording()

        func runTextEncoder(tokenIDs: [Int64], style: [[Float]]) async throws -> (ctx: [[Float]], mask: [Bool]) {
            recording.textStyles.append(style)
            return (Array(repeating: Array(repeating: 0.1, count: 256), count: tokenIDs.count), Array(repeating: true, count: tokenIDs.count))
        }

        func runDurationPredictor(ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], speed: Float) async throws -> Int {
            recording.durationSpeakerEmbeddings.append(speakerEmbedding)
            return 2
        }

        func runVectorEstimator(x: [[Float]], t: Float, ctx: [[Float]], mask: [Bool], speakerEmbedding: [Float], style: [[Float]]) async throws -> [[Float]] {
            if ctx.allSatisfy({ $0.allSatisfy { $0 == 0 } }) {
                recording.unconditionalCalls += 1
            } else {
                recording.vectorSpeakerEmbeddings.append(speakerEmbedding)
                recording.vectorStyles.append(style)
            }
            return x
        }

        func runCodecDecoder(latents: [[Float]]) async throws -> [Float] { [0, 0] }
    }

    func testTensorLayoutHelperTranspositionRoundTrip() {
        let frames = 4
        let dim = 3
        var matrix = [[Float]]()
        var val: Float = 1.0
        for _ in 0..<frames {
            var row = [Float]()
            for _ in 0..<dim {
                row.append(val)
                val += 1.0
            }
            matrix.append(row)
        }

        // matrix:
        // [1, 2, 3]
        // [4, 5, 6]
        // [7, 8, 9]
        // [10, 11, 12]
        let channelFirst = TensorLayoutHelper.transposeToChannelFirst(array2D: matrix)
        XCTAssertEqual(channelFirst.count, frames * dim)
        // Channel 0: [1, 4, 7, 10]
        XCTAssertEqual(Array(channelFirst[0..<4]), [1, 4, 7, 10])
        // Channel 1: [2, 5, 8, 11]
        XCTAssertEqual(Array(channelFirst[4..<8]), [2, 5, 8, 11])
        // Channel 2: [3, 6, 9, 12]
        XCTAssertEqual(Array(channelFirst[8..<12]), [3, 6, 9, 12])

        // Roundtrip unpacking
        let unpacked = TensorLayoutHelper.transposeFromChannelFirst(flat: channelFirst, frames: frames, dim: dim)
        XCTAssertEqual(unpacked.count, frames)
        for r in 0..<frames {
            XCTAssertEqual(unpacked[r], matrix[r])
        }
    }
}
