import Foundation

/// Errors thrown by VieNeuTTS SDK operations.
public enum VieNeuError: Error, LocalizedError, Equatable, Sendable {
    /// A required model file was not found in the bundle or cache directory.
    case modelNotFound(String)

    /// A model file failed integrity or checksum validation.
    case modelCorrupted(String)

    /// A model asset could not be downloaded from the remote repository.
    case downloadFailed(String)

    /// The input text is empty or contains only whitespace.
    case emptyInput

    /// The reference audio is shorter than 1.0 second or is completely silent.
    case referenceAudioTooShortOrSilent

    /// The speech synthesis or neural inference pipeline encountered an error.
    case synthesisFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model file not found: \(name)"
        case .modelCorrupted(let reason):
            return "Model file is corrupted: \(reason)"
        case .downloadFailed(let reason):
            return "Model download failed: \(reason)"
        case .emptyInput:
            return "Input text is empty or contains only whitespace."
        case .referenceAudioTooShortOrSilent:
            return "Reference audio must be at least 1.0 second long and not silent."
        case .synthesisFailed(let reason):
            return "Speech synthesis failed: \(reason)"
        }
    }
}
