import XCTest
@testable import VieNeuTTS

final class V3TurboSamplerTests: XCTestCase {
    func testArgmaxWhenTemperatureZero() {
        let logits: [Float] = [1.0, 5.0, 2.0, 0.5]
        let token = V3TurboSampler.sample(
            logits: logits,
            temperature: 0.0,
            topK: 0,
            topP: 1.0,
            repetitionPenalty: 1.0,
            history: nil
        )
        XCTAssertEqual(token, 1)
    }

    func testRepetitionPenaltySuppressesSeenTokens() {
        // Token 0 has highest logit (5.0), but with repetition penalty 2.0,
        // logit becomes 5.0 / 2.0 = 2.5, making token 1 (4.0) win.
        let logits: [Float] = [5.0, 4.0, 1.0]
        let token = V3TurboSampler.sample(
            logits: logits,
            temperature: 0.0,
            topK: 0,
            topP: 1.0,
            repetitionPenalty: 2.0,
            history: [0]
        )
        XCTAssertEqual(token, 1)
    }

    func testTopKRestrictsCandidates() {
        let logits: [Float] = [10.0, 9.0, 1.0, 0.0]
        // topK = 2 should only choose between index 0 and 1
        var chosenIndices = Set<Int>()
        for _ in 0..<50 {
            let t = V3TurboSampler.sample(
                logits: logits,
                temperature: 1.0,
                topK: 2,
                topP: 1.0,
                repetitionPenalty: 1.0,
                history: nil
            )
            chosenIndices.insert(t)
        }
        XCTAssertTrue(chosenIndices.isSubset(of: [0, 1]))
    }

    func testRepetitionHistorySlidingWindow() {
        let history = RepetitionHistory(window: 3)
        history.add(10)
        history.add(20)
        history.add(30)
        XCTAssertEqual(Set(history.seenTokens()), [10, 20, 30])

        // Adding 4th element evicts 10
        history.add(40)
        XCTAssertEqual(Set(history.seenTokens()), [20, 30, 40])
        XCTAssertFalse(history.seenTokens().contains(10))
    }
}
