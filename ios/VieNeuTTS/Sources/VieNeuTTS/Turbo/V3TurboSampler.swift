import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

/// Sampling pipeline implementing temperature, repetition penalty, Top-K, and Top-P (nucleus).
///
/// Ports the reference sampling logic from `src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py`.
public struct V3TurboSampler: Sendable {
    /// Samples a single token from raw logits.
    ///
    /// - Parameters:
    ///   - logits: Model output logits vector (e.g. 1024 codebook candidates or text vocab).
    ///   - temperature: Softmax temperature (0.0 = greedy argmax, > 0 = stochastic).
    ///   - topK: Number of top logits to retain (0 = all).
    ///   - topP: Cumulative probability mass threshold for nucleus sampling (1.0 = disabled).
    ///   - repetitionPenalty: Multiplier penalizing previously seen tokens (1.0 = neutral).
    ///   - history: Tokens emitted in recent frames subject to penalty.
    /// - Returns: Sampled token ID.
    public static func sample(
        logits: [Float],
        temperature: Float,
        topK: Int = 0,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        history: [Int]? = nil
    ) -> Int {
        guard !logits.isEmpty else { return 0 }
        var modified = logits

        // 1. Repetition penalty
        if abs(repetitionPenalty - 1.0) > 1e-4, let history = history {
            for token in history {
                if token >= 0 && token < modified.count {
                    let val = modified[token]
                    if val < 0 {
                        modified[token] = val * repetitionPenalty
                    } else {
                        modified[token] = val / repetitionPenalty
                    }
                }
            }
        }

        // 2. Greedy argmax if temperature <= 0
        if temperature <= 1e-6 {
            var bestIdx = 0
            var bestVal = modified[0]
            for i in 1..<modified.count {
                if modified[i] > bestVal {
                    bestVal = modified[i]
                    bestIdx = i
                }
            }
            return bestIdx
        }

        // 3. Temperature scaling
        let invTemp = 1.0 / temperature
        for i in 0..<modified.count {
            modified[i] *= invTemp
        }

        let V = modified.count
        var candidates: [Int]

        // 4. Top-K filtering: gather and sort top-K candidate indices
        if topK > 0 && topK < V {
            candidates = Array(0..<V)
            candidates.sort { modified[$0] > modified[$1] }
            candidates = Array(candidates.prefix(topK))
        } else {
            candidates = Array(0..<V)
            candidates.sort { modified[$0] > modified[$1] }
        }

        // 5. Softmax over selected candidate logits
        let candidateLogits = candidates.map { modified[$0] }
        let maxLogit = candidateLogits.first ?? 0.0
        let exps = candidateLogits.map { exp($0 - maxLogit) }
        var sumExp = exps.reduce(0.0, +)
        if sumExp <= 0 { sumExp = 1.0 }
        var probs = exps.map { $0 / sumExp }

        // 6. Top-P (nucleus) filtering
        if topP < 1.0 && topP > 0.0 {
            var cumSum: Float = 0.0
            var cutoff = probs.count
            for i in 0..<probs.count {
                if cumSum >= topP {
                    cutoff = max(1, i)
                    break
                }
                cumSum += probs[i]
            }
            candidates = Array(candidates.prefix(cutoff))
            probs = Array(probs.prefix(cutoff))
            let pSum = probs.reduce(0.0, +)
            if pSum > 0 {
                probs = probs.map { $0 / pSum }
            }
        }

        // 7. Categorical random draw
        let r = Float.random(in: 0..<1.0)
        var cumulative: Float = 0.0
        for i in 0..<probs.count {
            cumulative += probs[i]
            if r < cumulative {
                return candidates[i]
            }
        }

        return candidates.last ?? 0
    }
}
