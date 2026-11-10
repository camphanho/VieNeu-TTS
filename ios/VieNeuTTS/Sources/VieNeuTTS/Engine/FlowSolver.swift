import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Numerical ODE solver using the Euler method with sway schedule and Classifier-Free Guidance (CFG).
///
/// Ports the flow matching trajectory integration from `src/vieneu/v3nano.py`.
public struct FlowSolver: Sendable {
    /// Solves the vector field ODE using the Euler integration scheme with synchronous closures.
    ///
    /// - Parameters:
    ///   - initialLatents: Initial noise latents tensor of shape `[frames, dim]`.
    ///   - steps: Number of integration steps (clamped to at least 1).
    ///   - cfgScale: Classifier-Free Guidance extrapolation scale ($\ge 0$).
    ///   - sway: Time-grid warping parameter (e.g. `0.0` for uniform, `-1.0` for front-heavy schedule).
    ///   - uncondEstimator: Optional unconditional vector field estimator closure for CFG.
    ///   - vectorEstimator: Conditional vector field estimator closure producing velocities for `(x, t)`.
    /// - Returns: Integrated latents tensor after `steps` integration steps.
    /// - Throws: `CancellationError` if the task is cancelled, or `VieNeuError.synthesisFailed` on dimension mismatch.
    public static func solveEuler(
        initialLatents: [[Float]],
        steps: Int,
        cfgScale: Float = 0.0,
        sway: Float = 0.0,
        uncondEstimator: ((_ x: [[Float]], _ t: Float) throws -> [[Float]])? = nil,
        vectorEstimator: (_ x: [[Float]], _ t: Float) throws -> [[Float]]
    ) throws -> [[Float]] {
        guard !initialLatents.isEmpty else { return [] }
        let rows = initialLatents.count
        let cols = initialLatents[0].count
        guard cols > 0 else { return initialLatents }

        let n = max(1, steps)
        let tg = computeTimeGrid(steps: n, sway: sway)

        var x = initialLatents

        for i in 0..<n {
            try Task.checkCancellation()

            let t = tg[i]
            let dt = tg[i + 1] - tg[i]

            let vCond = try vectorEstimator(x, t)
            guard vCond.count == rows, vCond[0].count == cols else {
                throw VieNeuError.synthesisFailed("Vector estimator returned invalid shape.")
            }

            let v: [[Float]]
            if cfgScale > 0, let uncond = uncondEstimator {
                let vUncond = try uncond(x, t)
                guard vUncond.count == rows, vUncond[0].count == cols else {
                    throw VieNeuError.synthesisFailed("Unconditional vector estimator returned invalid shape.")
                }
                v = applyCFG(vCond: vCond, vUncond: vUncond, cfgScale: cfgScale)
            } else {
                v = vCond
            }

            applyEulerStep(x: &x, v: v, dt: dt)
        }

        return x
    }

    /// Solves the vector field ODE using the Euler integration scheme with asynchronous closures.
    ///
    /// - Parameters:
    ///   - initialLatents: Initial noise latents tensor of shape `[frames, dim]`.
    ///   - steps: Number of integration steps (clamped to at least 1).
    ///   - cfgScale: Classifier-Free Guidance extrapolation scale ($\ge 0$).
    ///   - sway: Time-grid warping parameter (e.g. `0.0` for uniform, `-1.0` for front-heavy schedule).
    ///   - uncondEstimator: Optional unconditional vector field estimator closure for CFG.
    ///   - vectorEstimator: Conditional vector field estimator closure producing velocities for `(x, t)`.
    /// - Returns: Integrated latents tensor after `steps` integration steps.
    /// - Throws: `CancellationError` if the task is cancelled, or `VieNeuError.synthesisFailed` on dimension mismatch.
    public static func solveEuler(
        initialLatents: [[Float]],
        steps: Int,
        cfgScale: Float = 0.0,
        sway: Float = 0.0,
        uncondEstimator: ((_ x: [[Float]], _ t: Float) async throws -> [[Float]])? = nil,
        vectorEstimator: (_ x: [[Float]], _ t: Float) async throws -> [[Float]]
    ) async throws -> [[Float]] {
        guard !initialLatents.isEmpty else { return [] }
        let rows = initialLatents.count
        let cols = initialLatents[0].count
        guard cols > 0 else { return initialLatents }

        let n = max(1, steps)
        let tg = computeTimeGrid(steps: n, sway: sway)

        var x = initialLatents

        for i in 0..<n {
            try Task.checkCancellation()

            let t = tg[i]
            let dt = tg[i + 1] - tg[i]

            let vCond = try await vectorEstimator(x, t)
            guard vCond.count == rows, vCond[0].count == cols else {
                throw VieNeuError.synthesisFailed("Vector estimator returned invalid shape.")
            }

            let v: [[Float]]
            if cfgScale > 0, let uncond = uncondEstimator {
                let vUncond = try await uncond(x, t)
                guard vUncond.count == rows, vUncond[0].count == cols else {
                    throw VieNeuError.synthesisFailed("Unconditional vector estimator returned invalid shape.")
                }
                v = applyCFG(vCond: vCond, vUncond: vUncond, cfgScale: cfgScale)
            } else {
                v = vCond
            }

            applyEulerStep(x: &x, v: v, dt: dt)
        }

        return x
    }

    /// Convenience overload with `condEstimator` naming matching SDK conventions.
    public static func solveEuler(
        initialLatents: [[Float]],
        steps: Int,
        cfgScale: Float = 0.0,
        sway: Float = 0.0,
        condEstimator: (_ x: [[Float]], _ t: Float) throws -> [[Float]],
        uncondEstimator: ((_ x: [[Float]], _ t: Float) throws -> [[Float]])? = nil
    ) throws -> [[Float]] {
        try solveEuler(
            initialLatents: initialLatents,
            steps: steps,
            cfgScale: cfgScale,
            sway: sway,
            uncondEstimator: uncondEstimator,
            vectorEstimator: condEstimator
        )
    }

    /// Convenience overload with `condEstimator` async naming.
    public static func solveEuler(
        initialLatents: [[Float]],
        steps: Int,
        cfgScale: Float = 0.0,
        sway: Float = 0.0,
        condEstimator: (_ x: [[Float]], _ t: Float) async throws -> [[Float]],
        uncondEstimator: ((_ x: [[Float]], _ t: Float) async throws -> [[Float]])? = nil
    ) async throws -> [[Float]] {
        try await solveEuler(
            initialLatents: initialLatents,
            steps: steps,
            cfgScale: cfgScale,
            sway: sway,
            uncondEstimator: uncondEstimator,
            vectorEstimator: condEstimator
        )
    }

    // MARK: - Helper Methods

    /// Computes the time grid points: $t_g = u + \text{sway} \cdot (\cos(\pi / 2 \cdot u) - 1 + u)$.
    public static func computeTimeGrid(steps: Int, sway: Float) -> [Float] {
        let n = max(1, steps)
        var tg = [Float](repeating: 0.0, count: n + 1)
        let swayDouble = Double(sway)

        for i in 0...n {
            let u = Double(i) / Double(n)
            let val = u + swayDouble * (cos(Double.pi / 2.0 * u) - 1.0 + u)
            tg[i] = Float(val)
        }
        return tg
    }

    /// Extrapolates vector velocities using Classifier-Free Guidance:
    /// $v = v_{\text{uncond}} + \text{cfgScale} \cdot (v_{\text{cond}} - v_{\text{uncond}})$.
    private static func applyCFG(vCond: [[Float]], vUncond: [[Float]], cfgScale: Float) -> [[Float]] {
        let rows = vCond.count
        let cols = vCond[0].count

        #if canImport(Accelerate)
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: cols), count: rows)
        var scale = cfgScale
        for r in 0..<rows {
            vCond[r].withUnsafeBufferPointer { condPtr in
                vUncond[r].withUnsafeBufferPointer { uncondPtr in
                    result[r].withUnsafeMutableBufferPointer { resPtr in
                        var diff = [Float](repeating: 0.0, count: cols)
                        diff.withUnsafeMutableBufferPointer { diffPtr in
                            // diff = vCond - vUncond
                            vDSP_vsub(uncondPtr.baseAddress!, 1, condPtr.baseAddress!, 1, diffPtr.baseAddress!, 1, vDSP_Length(cols))
                            // res = diff * scale + vUncond
                            vDSP_vsma(diffPtr.baseAddress!, 1, &scale, uncondPtr.baseAddress!, 1, resPtr.baseAddress!, 1, vDSP_Length(cols))
                        }
                    }
                }
            }
        }
        return result
        #else
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: cols), count: rows)
        for r in 0..<rows {
            for c in 0..<cols {
                result[r][c] = vUncond[r][c] + cfgScale * (vCond[r][c] - vUncond[r][c])
            }
        }
        return result
        #endif
    }

    /// Updates latents with Euler integration step: $x \leftarrow x + dt \cdot v$.
    private static func applyEulerStep(x: inout [[Float]], v: [[Float]], dt: Float) {
        let rows = x.count
        let cols = x[0].count

        #if canImport(Accelerate)
        var dtCopy = dt
        for r in 0..<rows {
            v[r].withUnsafeBufferPointer { vPtr in
                x[r].withUnsafeMutableBufferPointer { xPtr in
                    // xPtr = vPtr * dt + xPtr
                    vDSP_vsma(vPtr.baseAddress!, 1, &dtCopy, xPtr.baseAddress!, 1, xPtr.baseAddress!, 1, vDSP_Length(cols))
                }
            }
        }
        #else
        for r in 0..<rows {
            for c in 0..<cols {
                x[r][c] += dt * v[r][c]
            }
        }
        #endif
    }
}
