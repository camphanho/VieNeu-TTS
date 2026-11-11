import XCTest
@testable import VieNeuTTS

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

private final class ProgressTracker: @unchecked Sendable {
    private var recorded: [Double] = []
    private let lock = NSLock()

    func append(_ val: Double) {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(val)
    }

    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}

final class ModelProviderTests: XCTestCase {
    func testBundleModelProviderThrowsOnMissingModel() {
        let provider = BundleModelProvider(bundle: .main)
        XCTAssertThrowsError(try provider.modelURL(for: "non_existent.onnx")) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
        }
    }

    func testBundleModelProviderConstantsAndConfigThrowWhenMissing() {
        let provider = BundleModelProvider(bundle: .main)
        XCTAssertThrowsError(try provider.constantsURL()) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
        }
        XCTAssertThrowsError(try provider.configURL()) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
        }
    }

    func testCorruptedModelDetectionThrows() throws {
        // Review Focus #3: Corrupted model file triggers .modelCorrupted error
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let corruptedFile = tempDir.appendingPathComponent("text_encoder.onnx")
        try "corrupted data".write(to: corruptedFile, atomically: true, encoding: .utf8)

        let provider = CachedModelProvider(directory: tempDir, expectedChecksums: ["text_encoder.onnx": "valid_sha256_hash"])
        XCTAssertThrowsError(try provider.validateChecksums()) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelCorrupted = vieNeuError else {
                XCTFail("Expected .modelCorrupted error, got \(error)")
                return
            }
        }
    }

    func testValidChecksumPasses() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let testFile = tempDir.appendingPathComponent("constants.npz")
        try "abc".write(to: testFile, atomically: true, encoding: .utf8)

        // Known SHA256 of "abc"
        let expectedHash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        let provider = CachedModelProvider(directory: tempDir, expectedChecksums: ["constants.npz": expectedHash])
        XCTAssertNoThrow(try provider.validateChecksums())

        let constantsURL = try provider.constantsURL()
        XCTAssertEqual(constantsURL.lastPathComponent, "constants.npz")
    }

    func testCachedModelProviderThrowsOnMissingFile() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let provider = CachedModelProvider(directory: tempDir)
        XCTAssertThrowsError(try provider.modelURL(for: "missing_file.onnx")) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound(let name) = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
            XCTAssertEqual(name, "missing_file.onnx")
        }
    }

    func testCachedModelProviderResolvesExistingFiles() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let configFile = tempDir.appendingPathComponent("config.json")
        try "{}".write(to: configFile, atomically: true, encoding: .utf8)

        let provider = CachedModelProvider(directory: tempDir)
        let resolvedConfig = try provider.configURL()
        XCTAssertEqual(resolvedConfig.lastPathComponent, "config.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: resolvedConfig.path))
    }

    func testCachedModelProviderThrowsWhenChecksumFileIsMissing() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let provider = CachedModelProvider(
            directory: tempDir,
            expectedChecksums: ["non_existent.onnx": "some_hash"]
        )
        XCTAssertThrowsError(try provider.validateChecksums()) { error in
            guard let vieNeuError = error as? VieNeuError, case .modelNotFound(let name) = vieNeuError else {
                XCTFail("Expected .modelNotFound error, got \(error)")
                return
            }
            XCTAssertEqual(name, "non_existent.onnx")
        }
    }

    func testModelDownloaderURLConstruction() async {
        let downloader = ModelDownloader(repo: "pnnbao-ump/VieNeu-TTS-v3-Nano")
        let url = await downloader.downloadURL(for: "text_encoder.onnx")
        XCTAssertEqual(url.absoluteString, "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/text_encoder.onnx")

        let required = ModelDownloader.requiredFiles
        XCTAssertTrue(required.contains("text_encoder.onnx"))
        XCTAssertTrue(required.contains("duration_predictor.onnx"))
        XCTAssertTrue(required.contains("vector_estimator.onnx"))
        XCTAssertTrue(required.contains("codec_decoder.onnx"))
        XCTAssertTrue(required.contains("config.json"))
        XCTAssertTrue(required.contains("constants.npz"))
    }

    func testModelDownloaderSkipsCachedVerifiedFiles() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        var checksums: [String: String] = [:]
        // Pre-create all required files
        for filename in ModelDownloader.requiredFiles {
            let fileURL = tempDir.appendingPathComponent(filename)
            let dummyData = "content of \(filename)".data(using: .utf8)!
            try dummyData.write(to: fileURL)
            let hash = SHA256Helper.hexString(for: dummyData)
            checksums[filename] = hash
        }

        // Downloader with localhost dummy URL that would fail if network was hit
        let dummyBaseURL = URL(string: "http://127.0.0.1:59999")!
        let downloader = ModelDownloader(
            baseURL: dummyBaseURL,
            expectedChecksums: checksums
        )

        let tracker = ProgressTracker()
        let provider = try await downloader.download(to: tempDir) { p in
            tracker.append(p)
        }

        let recorded = tracker.values
        XCTAssertFalse(recorded.isEmpty)
        XCTAssertEqual(recorded.last, 1.0)
        XCTAssertEqual(try provider.configURL().lastPathComponent, "config.json")
    }

    func testModelDownloaderThrowsOnNetworkFailure() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let badBaseURL = URL(string: "http://127.0.0.1:59999/does-not-exist")!
        let downloader = ModelDownloader(
            baseURL: badBaseURL
        )

        do {
            _ = try await downloader.download(to: tempDir)
            XCTFail("Expected downloadFailed error on unreachable network endpoint")
        } catch let error as VieNeuError {
            guard case .downloadFailed = error else {
                XCTFail("Expected .downloadFailed error, got \(error)")
                return
            }
        }
    }

    func testModelDownloaderSkipsExistingFilesWithoutChecksums() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Pre-create all required files with non-zero size
        for filename in ModelDownloader.requiredFiles {
            let fileURL = tempDir.appendingPathComponent(filename)
            try "non-empty-model-content".write(to: fileURL, atomically: true, encoding: .utf8)
        }

        // Downloader with dummy URL and NO checksums
        let dummyBaseURL = URL(string: "http://127.0.0.1:59999")!
        let downloader = ModelDownloader(baseURL: dummyBaseURL, expectedChecksums: nil)

        let provider = try await downloader.download(to: tempDir)
        XCTAssertEqual(try provider.constantsURL().lastPathComponent, "constants.npz")
    }

    func testModelDownloaderPreservesCancellation() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let badBaseURL = URL(string: "http://127.0.0.1:59999/does-not-exist")!
        let downloader = ModelDownloader(baseURL: badBaseURL)

        let downloadTask = Task {
            try await downloader.download(to: tempDir)
        }
        downloadTask.cancel()

        do {
            _ = try await downloadTask.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // Expected
        } catch {
            XCTFail("Expected CancellationError, but got \(error)")
        }
    }

    func testVieNeuErrorDescriptions() {
        let notFound = VieNeuError.modelNotFound("model.onnx")
        XCTAssertNotNil(notFound.errorDescription)
        XCTAssertTrue(notFound.errorDescription?.contains("model.onnx") == true)

        let corrupted = VieNeuError.modelCorrupted("bad checksum")
        XCTAssertNotNil(corrupted.errorDescription)
        XCTAssertTrue(corrupted.errorDescription?.contains("bad checksum") == true)

        let dlFailed = VieNeuError.downloadFailed("timeout")
        XCTAssertNotNil(dlFailed.errorDescription)
        XCTAssertTrue(dlFailed.errorDescription?.contains("timeout") == true)

        let empty = VieNeuError.emptyInput
        XCTAssertNotNil(empty.errorDescription)

        let tooShort = VieNeuError.referenceAudioTooShortOrSilent
        XCTAssertNotNil(tooShort.errorDescription)

        let synthFailed = VieNeuError.synthesisFailed("engine error")
        XCTAssertNotNil(synthFailed.errorDescription)
    }

    func testTurboRequiredFilesConfiguration() {
        XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("vieneu_prefill.onnx"))
        XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("vieneu_decode_step.onnx"))
        XCTAssertTrue(ModelDownloader.turboRequiredFiles.contains("moss_audio_tokenizer_decode_full.onnx"))
        XCTAssertEqual(VieNeuConfig.mock(mode: .v3turbo).mode, .v3turbo)
    }

    func testParseNPZFromRealConstantsFile() throws {
        let npzPath = "/home/camph/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Nano/snapshots/aba295eb96a6fa6003ebe417cc1f2802a7adc1dc/constants.npz"
        guard FileManager.default.fileExists(atPath: npzPath) else { return }
        let data = try Data(contentsOf: URL(fileURLWithPath: npzPath))
        let constants = ModelConstants.parseNPZ(data: data)
        XCTAssertNotNil(constants)
        XCTAssertEqual(constants?.null_spk.count, 192)
        XCTAssertEqual(constants?.null_style.count, 50)
        XCTAssertEqual(constants?.null_style[0].count, 256)
        XCTAssertEqual(constants?.latent_mean.count, 24)
        XCTAssertEqual(constants?.latent_std.count, 24)
        XCTAssertEqual(constants?.latent_scale, 0.25)
    }
}
