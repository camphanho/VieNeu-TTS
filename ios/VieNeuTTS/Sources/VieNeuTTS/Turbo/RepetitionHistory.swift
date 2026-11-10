import Foundation

/// Sliding-window repetition tracker for a codebook channel.
///
/// Prevents local acoustic loops while allowing legitimate phonetic repetitions
/// beyond the sliding window.
public final class RepetitionHistory: @unchecked Sendable {
    public let window: Int
    private var seen: [Int: Int] = [:]
    private var order: [Int] = []
    private let lock = NSLock()

    public init(window: Int = 64) {
        self.window = window
    }

    /// Records an emitted token into the history, evicting the oldest if window is exceeded.
    public func add(_ token: Int) {
        lock.lock()
        defer { lock.unlock() }

        seen[token, default: 0] += 1
        if window > 0 {
            order.append(token)
            if order.count > window {
                let old = order.removeFirst()
                if let count = seen[old] {
                    if count <= 1 {
                        seen.removeValue(forKey: old)
                    } else {
                        seen[old] = count - 1
                    }
                }
            }
        }
    }

    /// Returns all distinct tokens currently present within the sliding window.
    public func seenTokens() -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        return Array(seen.keys)
    }
}
