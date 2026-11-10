// ios/VieNeuTTS/Sources/VieNeuTTS/Text/SeaG2P.swift
import Foundation

#if canImport(SeaG2P)
import SeaG2P
#elseif canImport(CSeaG2P)
import CSeaG2P
#endif

/// Errors thrown by SeaG2P grapheme-to-phoneme operations.
public enum SeaG2PError: Error, LocalizedError, Equatable, Sendable {
    case stringConversionFailed
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .stringConversionFailed:
            return "Failed to convert string between Swift and C representations."
        case .processingFailed(let details):
            return "SeaG2P processing error: \(details)"
        }
    }
}

/// Swift wrapper for the sea-g2p Vietnamese text normalization and phonemization engine.
public final class SeaG2P: Sendable {

    /// Initializes a SeaG2P instance.
    ///
    /// - Parameter dictionaryPath: Optional explicit filesystem path to the `sea_g2p.bin` dictionary.
    ///   If not specified, SeaG2P checks Bundle resources, the `SEA_G2P_DICT_PATH` environment variable,
    ///   standard application directory locations, or falls back to an embedded empty dictionary.
    public init(dictionaryPath: String? = nil) {
        if let explicitPath = dictionaryPath {
            _ = SeaG2P.setDictionaryPath(explicitPath)
        } else {
            // Check application Bundle for bundled dictionary if available
            #if canImport(Foundation)
            if let bundlePath = Bundle.main.path(forResource: "sea_g2p", ofType: "bin") {
                _ = SeaG2P.setDictionaryPath(bundlePath)
            }
            #endif
        }
    }

    /// Sets the dictionary file path globally for the G2P engine.
    ///
    /// Note: Must be invoked before the first call to `phonemize(_:)` because the engine
    /// initialization is locked upon first access.
    ///
    /// - Parameter path: Absolute or relative filesystem path to `sea_g2p.bin`.
    /// - Returns: `true` if the path was accepted, `false` otherwise.
    @discardableResult
    public static func setDictionaryPath(_ path: String) -> Bool {
        return path.withCString { cPath in
            sea_g2p_set_dict_path(cPath) == 0
        }
    }

    /// Normalizes punctuation spacing in Vietnamese text for optimal TTS synthesis.
    ///
    /// Inserts spaces before terminal or separating punctuation marks (`!`, `?`, `,`, `.`, `;`, `:`).
    ///
    /// - Parameter text: Input text.
    /// - Returns: Normalized string with balanced punctuation spacing.
    /// - Throws: `SeaG2PError` if C-bridge processing fails.
    public func normalizePunctuation(_ text: String) throws -> String {
        if text.isEmpty {
            return ""
        }

        return try text.withCString { cInput in
            guard let ptr = sea_g2p_normalize_punc(cInput) else {
                throw SeaG2PError.processingFailed("sea_g2p_normalize_punc returned NULL pointer")
            }
            defer { sea_g2p_free_string(ptr) }
            return String(cString: ptr)
        }
    }

    /// Converts Vietnamese/bilingual text into acoustic phoneme representations.
    ///
    /// - Parameter text: Input text (preferably punctuation-normalized).
    /// - Returns: Space-separated phonetic string.
    /// - Throws: `SeaG2PError` if C-bridge processing fails.
    public func phonemize(_ text: String) throws -> String {
        if text.isEmpty {
            return ""
        }

        return try text.withCString { cInput in
            guard let ptr = sea_g2p_phonemize(cInput) else {
                throw SeaG2PError.processingFailed("sea_g2p_phonemize returned NULL pointer")
            }
            defer { sea_g2p_free_string(ptr) }
            return String(cString: ptr)
        }
    }
}
