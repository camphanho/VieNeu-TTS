import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Actor responsible for downloading VieNeuTTS model assets from Hugging Face or a custom repository.
public actor ModelDownloader {
    /// Default Hugging Face repository containing v3 Nano ONNX models and configurations.
    public static let defaultRepo = "pnnbao-ump/VieNeu-TTS-v3-Nano"

    /// List of model filenames required for the v3 Nano neural pipeline.
    public static let requiredFiles: [String] = [
        "text_encoder.onnx",
        "duration_predictor.onnx",
        "vector_estimator.onnx",
        "codec_decoder.onnx",
        "config.json",
        "constants.npz"
    ]

    /// Repository identifier (e.g. `pnnbao-ump/VieNeu-TTS-v3-Nano`).
    public let repo: String

    /// Base download URL for the assets.
    public let baseURL: URL

    /// URLSession instance used for network transfers.
    public let session: URLSession

    /// Optional SHA-256 checksums to validate against after download.
    public let expectedChecksums: [String: String]?

    /// Initializes a ModelDownloader instance.
    ///
    /// - Parameters:
    ///   - repo: Hugging Face repo name. Defaults to `defaultRepo`.
    ///   - baseURL: Custom base URL override (optional).
    ///   - session: URLSession instance. Defaults to `.shared`.
    ///   - expectedChecksums: Optional dictionary of expected SHA-256 checksums per file.
    public init(
        repo: String = defaultRepo,
        baseURL: URL? = nil,
        session: URLSession = .shared,
        expectedChecksums: [String: String]? = nil
    ) {
        self.repo = repo
        self.baseURL = baseURL ?? URL(string: "https://huggingface.co/\(repo)/resolve/main")!
        self.session = session
        self.expectedChecksums = expectedChecksums
    }

    /// Constructs the full download URL for an asset filename.
    public func downloadURL(for filename: String) -> URL {
        baseURL.appendingPathComponent(filename)
    }

    /// Downloads all required model assets to the target cache directory.
    ///
    /// - Parameters:
    ///   - directory: Destination directory. If `nil`, defaults to `Caches/VieNeuTTS/models/<repo>`.
    ///   - progress: Optional callback receiving progress fraction from 0.0 to 1.0.
    /// - Returns: A `CachedModelProvider` pointing to the directory.
    /// - Throws: `VieNeuError.downloadFailed` or `VieNeuError.modelCorrupted`.
    public func download(
        to directory: URL? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> CachedModelProvider {
        let targetDir: URL
        if let dir = directory {
            targetDir = dir
        } else {
            let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            targetDir = base.appendingPathComponent("VieNeuTTS/models/\(repo)")
        }

        do {
            try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        } catch {
            throw VieNeuError.downloadFailed("Failed to create destination directory: \(error.localizedDescription)")
        }

        let totalFiles = Self.requiredFiles.count
        var completedFiles = 0

        for filename in Self.requiredFiles {
            let destinationURL = targetDir.appendingPathComponent(filename)

            var needDownload = true
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                if let expectedHash = expectedChecksums?[filename],
                   let existingData = try? Data(contentsOf: destinationURL) {
                    let actualHash = SHA256Helper.hexString(for: existingData)
                    if actualHash.lowercased() == expectedHash.lowercased() {
                        needDownload = false
                    }
                }
            }

            if needDownload {
                let remoteURL = downloadURL(for: filename)
                do {
                    let (tempURL, response) = try await session.download(from: remoteURL)
                    if let httpResponse = response as? HTTPURLResponse,
                       !(200...299).contains(httpResponse.statusCode) {
                        throw VieNeuError.downloadFailed(
                            "HTTP status \(httpResponse.statusCode) downloading \(filename) from \(remoteURL)"
                        )
                    }

                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try? FileManager.default.removeItem(at: destinationURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destinationURL)
                } catch let error as VieNeuError {
                    throw error
                } catch {
                    throw VieNeuError.downloadFailed(
                        "Failed to download \(filename): \(error.localizedDescription)"
                    )
                }
            }

            completedFiles += 1
            progress?(Double(completedFiles) / Double(totalFiles))
        }

        let provider = CachedModelProvider(directory: targetDir, expectedChecksums: expectedChecksums)
        try provider.validateChecksums()
        return provider
    }

    /// Convenience download overload taking only a progress callback.
    public func download(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> CachedModelProvider {
        try await download(to: nil, progress: progress)
    }
}
