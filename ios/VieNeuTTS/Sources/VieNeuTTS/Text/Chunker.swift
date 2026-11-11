// ios/VieNeuTTS/Sources/VieNeuTTS/Text/Chunker.swift
import Foundation

/// A sentence-level text chunk paired with a natural prosodic pause duration.
public struct SentenceChunk: Equatable, Sendable {
    /// Text content of the chunk.
    public let text: String

    /// Pause duration (in seconds) following this chunk during synthesis or playback.
    public let pauseDuration: Float

    /// Creates a sentence chunk.
    public init(text: String, pauseDuration: Float = 0.2) {
        self.text = text
        self.pauseDuration = pauseDuration
    }
}

/// Chunker splitting Vietnamese and multi-lingual text into sentence-level chunks with natural pause durations.
public struct Chunker {

    /// Standard sentence ending characters.
    private static let sentenceEndCharacters: Set<Character> = [".", "!", "?", "…"]

    /// Secondary punctuation marks suitable for breaking long clauses.
    private static let secondaryPunctuation: Set<Character> = [";", ",", ":", "-", "–", "—"]

    /// Trailing punctuation / quotation characters that stay attached to the sentence terminator.
    private static let trailingQuoteCharacters: Set<Character> = ["\"", "'", "”", "’", "»", "›", "』", "」", ")", "]", "}"]

    /// Splits an input text into bounded `SentenceChunk`s with natural pauses (0.15s – 0.3s).
    ///
    /// - Parameters:
    ///   - text: The input raw text.
    ///   - maxChars: Maximum character limit per chunk (default: 200).
    ///   - minChunkChars: Minimum character threshold below which short sentences are merged (default: 0).
    /// - Returns: An array of `SentenceChunk` structs. Returns empty array for empty or whitespace-only inputs.
    public static func splitToChunks(_ text: String, maxChars: Int = 200, minChunkChars: Int = 0) -> [SentenceChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return []
        }

        // 1. Split into paragraphs on newlines
        let rawParagraphs = text.components(separatedBy: CharacterSet.newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !rawParagraphs.isEmpty else {
            return []
        }

        var resultChunks: [SentenceChunk] = []

        for (paraIndex, paragraph) in rawParagraphs.enumerated() {
            let isLastParagraph = (paraIndex == rawParagraphs.count - 1)
            let sentences = splitParagraphIntoSentences(paragraph)

            var buffer = ""

            for sentence in sentences {
                if sentence.count > maxChars {
                    // Flush existing buffer if non-empty
                    if !buffer.isEmpty {
                        let pause = classifyPause(for: buffer, isParagraphEnd: false)
                        resultChunks.append(SentenceChunk(text: buffer, pauseDuration: pause))
                        buffer = ""
                    }

                    // Split overly long sentence into sub-clause pieces
                    let subPieces = splitLongSentence(sentence, maxChars: maxChars)
                    for (pieceIndex, piece) in subPieces.enumerated() {
                        let isLastPiece = (pieceIndex == subPieces.count - 1)
                        if isLastPiece {
                            buffer = piece
                        } else {
                            let pause = classifyPause(for: piece, isParagraphEnd: false)
                            resultChunks.append(SentenceChunk(text: piece, pauseDuration: pause))
                        }
                    }
                } else {
                    let candidate = buffer.isEmpty ? sentence : "\(buffer) \(sentence)"
                    if candidate.count <= maxChars {
                        buffer = candidate
                    } else {
                        if !buffer.isEmpty {
                            let pause = classifyPause(for: buffer, isParagraphEnd: false)
                            resultChunks.append(SentenceChunk(text: buffer, pauseDuration: pause))
                        }
                        buffer = sentence
                    }
                }
            }

            if !buffer.isEmpty {
                // If it's the end of a paragraph (and not the very last paragraph of the entire text),
                // assign paragraph pause (0.3s). Otherwise classify by ending punctuation.
                let isParagraphEnd = !isLastParagraph
                let pause = classifyPause(for: buffer, isParagraphEnd: isParagraphEnd)
                resultChunks.append(SentenceChunk(text: buffer, pauseDuration: pause))
                buffer = ""
            }
        }

        return mergeShortChunks(resultChunks, minChars: minChunkChars)
    }

    /// Merges overly short chunks (< minChars) into adjacent chunks to ensure the neural model
    /// has sufficient context, matching Python's `_merge_short_chunks`.
    private static func mergeShortChunks(_ chunks: [SentenceChunk], minChars: Int) -> [SentenceChunk] {
        guard chunks.count > 1, minChars > 0 else { return chunks }
        var list = chunks

        var iterations = 0
        while list.count > 1 && iterations < 50 {
            iterations += 1
            guard let shortIndex = list.indices.first(where: { list[$0].text.count < minChars }) else {
                break
            }

            if shortIndex < list.count - 1 {
                let curr = list[shortIndex]
                let next = list.remove(at: shortIndex + 1)
                list[shortIndex] = SentenceChunk(
                    text: "\(curr.text) \(next.text)",
                    pauseDuration: next.pauseDuration
                )
            } else if shortIndex > 0 {
                let prev = list[shortIndex - 1]
                let curr = list.remove(at: shortIndex)
                list[shortIndex - 1] = SentenceChunk(
                    text: "\(prev.text) \(curr.text)",
                    pauseDuration: curr.pauseDuration
                )
            } else {
                break
            }
        }

        return list
    }

    /// Splits a single paragraph into individual sentences while respecting quotes and decimal numbers.
    /// If quotes or parentheses are unbalanced, falls back to splitting without quote protection.
    private static func splitParagraphIntoSentences(_ paragraph: String) -> [String] {
        let (sentences, isBalanced) = scanSentences(paragraph, quoteAware: true)
        if !isBalanced {
            let (fallbackSentences, _) = scanSentences(paragraph, quoteAware: false)
            return fallbackSentences.isEmpty ? [cleanWhitespace(paragraph)] : fallbackSentences
        }
        return sentences.isEmpty ? [cleanWhitespace(paragraph)] : sentences
    }

    /// Scans a paragraph and splits it at sentence boundaries, optionally respecting quotes and brackets.
    private static func scanSentences(_ paragraph: String, quoteAware: Bool) -> (sentences: [String], isBalanced: Bool) {
        var sentences: [String] = []
        let chars = Array(paragraph)
        let n = chars.count
        var start = 0
        var i = 0

        var inQuote = false
        var parenDepth = 0

        while i < n {
            let ch = chars[i]

            if quoteAware {
                if ch == "\"" || ch == "“" || ch == "”" {
                    inQuote.toggle()
                } else if ch == "(" || ch == "[" || ch == "{" {
                    parenDepth += 1
                } else if ch == ")" || ch == "]" || ch == "}" {
                    if parenDepth > 0 { parenDepth -= 1 }
                }
            }

            let canBreak = !quoteAware || (!inQuote && parenDepth == 0)

            if sentenceEndCharacters.contains(ch) && canBreak {
                // Check if this might be a decimal number (e.g. 3.5 or 8.30)
                let isPrecededByDigit = (i > 0 && chars[i - 1].isNumber)
                let isFollowedByDigit = (i + 1 < n && chars[i + 1].isNumber)
                if ch == "." && isPrecededByDigit && isFollowedByDigit {
                    i += 1
                    continue
                }

                // Consume contiguous sentence end markers (e.g. "...", "?!", "!!")
                var j = i + 1
                while j < n && sentenceEndCharacters.contains(chars[j]) {
                    j += 1
                }
                // Consume trailing quotation marks or parentheses (e.g. '."', '!")')
                while j < n && trailingQuoteCharacters.contains(chars[j]) {
                    j += 1
                }

                // If followed by whitespace or end of string, it is a valid sentence break
                if j >= n || chars[j].isWhitespace {
                    let sentenceString = cleanWhitespace(String(chars[start..<j]))
                    if !sentenceString.isEmpty {
                        sentences.append(sentenceString)
                    }
                    // Skip following whitespace
                    while j < n && chars[j].isWhitespace {
                        j += 1
                    }
                    start = j
                    i = j
                    continue
                }
            }

            i += 1
        }

        if start < n {
            let remainder = cleanWhitespace(String(chars[start..<n]))
            if !remainder.isEmpty {
                sentences.append(remainder)
            }
        }

        let isBalanced = (!inQuote && parenDepth == 0)
        return (sentences, isBalanced)
    }

    /// Normalizes and collapses repeated whitespace characters into single spaces.
    private static func cleanWhitespace(_ text: String) -> String {
        text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Splits a sentence exceeding `maxChars` into sub-clauses based on secondary punctuation or word boundaries.
    private static func splitLongSentence(_ sentence: String, maxChars: Int) -> [String] {
        var pieces: [String] = []

        // First attempt: split by secondary punctuation like ';', ',', ':', '-'
        let words = sentence.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var currentBuffer = ""

        for word in words {
            let candidate = currentBuffer.isEmpty ? word : "\(currentBuffer) \(word)"
            if candidate.count <= maxChars {
                currentBuffer = candidate
                // If this word ends with a secondary punctuation and current buffer is sufficiently long, break
                if let lastChar = word.last, secondaryPunctuation.contains(lastChar), currentBuffer.count >= maxChars / 2 {
                    pieces.append(currentBuffer)
                    currentBuffer = ""
                }
            } else {
                if !currentBuffer.isEmpty {
                    pieces.append(currentBuffer)
                    currentBuffer = word
                } else {
                    // Single word itself is longer than maxChars: hard split by characters
                    var longWord = word
                    while longWord.count > maxChars {
                        let splitIndex = longWord.index(longWord.startIndex, offsetBy: maxChars)
                        pieces.append(String(longWord[..<splitIndex]))
                        longWord = String(longWord[splitIndex...])
                    }
                    currentBuffer = longWord
                }
            }
        }

        if !currentBuffer.isEmpty {
            pieces.append(currentBuffer)
        }

        return pieces.isEmpty ? [sentence] : pieces
    }

    /// Classifies the pause duration following a chunk based on paragraph boundary and ending punctuation.
    private static func classifyPause(for text: String, isParagraphEnd: Bool) -> Float {
        if isParagraphEnd {
            return 0.3
        }

        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard let lastChar = trimmed.last else {
            return 0.2
        }

        if sentenceEndCharacters.contains(lastChar) {
            return 0.2
        }

        // If the chunk ends with a closing quote, check the character before it
        if trailingQuoteCharacters.contains(lastChar), trimmed.count >= 2 {
            let secondLast = trimmed[trimmed.index(trimmed.endIndex, offsetBy: -2)]
            if sentenceEndCharacters.contains(secondLast) {
                return 0.2
            }
        }

        if secondaryPunctuation.contains(lastChar) {
            return 0.15
        }

        // Forcefully split mid-sentence
        return 0.15
    }
}
