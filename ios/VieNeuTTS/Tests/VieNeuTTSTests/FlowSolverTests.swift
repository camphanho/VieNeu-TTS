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
        let (mu, mask) = try await engine.runTextEncoder(tokenIDs: [1, 2, 3], speakerEmbedding: Array(repeating: 0.1, count: 192))
        XCTAssertEqual(mu.count, 3)
        XCTAssertEqual(mu[0].count, 256)
        XCTAssertEqual(mask.count, 3)

        let durations = try await engine.runDurationPredictor(mu: mu)
        XCTAssertEqual(durations.count, 3)

        let vector = try await engine.runVectorEstimator(
            x: [[1.0, 2.0]],
            t: 0.5,
            cond: [[0.1]],
            style: [[0.2]]
        )
        XCTAssertEqual(vector.count, 1)
        XCTAssertEqual(vector[0].count, 2)

        let audio = try await engine.runCodecDecoder(latents: [[0.1, 0.2]])
        XCTAssertFalse(audio.isEmpty)

        let spk = try await engine.runSpeakerEncoder(melFilterbank: [[0.1]])
        XCTAssertEqual(spk.count, 192)

        let codec = try await engine.runCodecEncoder(audio: [0.1, 0.2])
        XCTAssertEqual(codec.count, 50)
        XCTAssertEqual(codec[0].count, 256)

        let style = try await engine.runReferenceEncoder(refCodes: codec)
        XCTAssertEqual(style.count, 50)
        XCTAssertEqual(style[0].count, 256)
    }
}
