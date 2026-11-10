import Foundation
import VieNeuTTS

#if canImport(Combine)
import Combine
#else
public protocol ObservableObject: AnyObject {}
@propertyWrapper
public struct Published<Value> {
    public var wrappedValue: Value
    public init(wrappedValue: Value) {
        self.wrappedValue = wrappedValue
    }
}
#endif

#if canImport(AVFoundation)
import AVFoundation
#endif

/// Main view model driving the VieNeu-TTS SwiftUI demo application.
///
/// Orchestrates model downloading, speech synthesis (full and sentence-by-sentence streaming),
/// audio recording via `AVAudioRecorder`, instant voice cloning, and audio playback.
@MainActor
public final class TTSViewModel: ObservableObject {
    // MARK: - Published State Properties

    /// Indicates whether the neural TTS engine and voices are fully loaded and ready for synthesis.
    @Published public var isReady: Bool = false

    /// Indicates whether neural model files are currently downloading.
    @Published public var isDownloading: Bool = false

    /// Fractional download progress from 0.0 to 1.0.
    @Published public var downloadProgress: Double = 0.0

    /// Indicates whether a synthesis or voice cloning operation is currently executing.
    @Published public var isSynthesizing: Bool = false

    /// Indicates whether audio is currently being recorded from the microphone.
    @Published public var isRecording: Bool = false

    /// Indicates whether audio playback is currently active.
    @Published public var isPlaying: Bool = false

    /// The currently selected voice profile for speech synthesis.
    @Published public var selectedVoice: VoiceProfile

    /// All available preset and user-cloned voice profiles.
    @Published public var availableVoices: [VoiceProfile] = []

    /// Real-Time Factor (RTF = generation time / audio duration).
    /// Values < 1.0 indicate faster-than-realtime synthesis.
    @Published public var rtf: Double = 0.0

    /// Duration of the most recently synthesized audio in seconds.
    @Published public var audioDuration: TimeInterval = 0.0

    /// Wall-clock time in seconds taken to synthesize the audio.
    @Published public var generationTime: TimeInterval = 0.0

    /// Current duration of the active or completed microphone recording in seconds.
    @Published public var recordingDuration: TimeInterval = 0.0

    /// Playback progress fraction from 0.0 to 1.0.
    @Published public var playbackProgress: Double = 0.0

    /// Current operational status message for user feedback.
    @Published public var statusMessage: String = "Chưa khởi tạo"

    /// Error message to display in UI alerts or banners, if any.
    @Published public var errorMessage: String? = nil

    /// Euler ODE integration steps (e.g. 8 for fast preview, 16 for high fidelity).
    @Published public var eulerSteps: Int = 16

    /// Speech synthesis speed multiplier (0.8x to 1.5x).
    @Published public var speed: Float = 1.0

    /// File URL of the most recently synthesized audio WAV file.
    @Published public var synthesizedAudioURL: URL? = nil

    /// File URL of the most recently recorded microphone voice sample.
    @Published public var recordedAudioURL: URL? = nil

    /// Chunks accumulated during active streaming synthesis.
    @Published public var activeStreamChunks: [AudioChunk] = []

    /// Indicates whether streaming synthesis is actively receiving chunks.
    @Published public var isStreamingActive: Bool = false

    // MARK: - Private Dependencies

    /// Primary VieNeuTTS facade instance.
    public private(set) var tts: VieNeuTTS?

    /// Active model provider resolving ONNX weights and configurations.
    public private(set) var modelProvider: (any ModelProvider)?

    #if canImport(AVFoundation)
    private var audioRecorder: AVAudioRecorder?
    private var audioPlayer: AVAudioPlayer?
    private var recordingTimer: Timer?
    private var playbackTimer: Timer?
    private let audioDelegate = AudioDelegateHelper()
    #endif

    // MARK: - Initialization

    /// Initializes a new view model with default mock voice selections.
    public init() {
        let defaultVoice = VoiceProfile.mock(name: "Minh Quân")
        self.selectedVoice = defaultVoice
        self.availableVoices = [
            defaultVoice,
            VoiceProfile.mock(name: "Mai Anh")
        ]

        #if canImport(AVFoundation)
        setupAudioDelegates()
        #endif
    }

    // MARK: - Setup & Engine Initialization

    /// Initializes the VieNeuTTS neural synthesis engine.
    ///
    /// - Parameter fromBundle: If `true`, attempts to load model files from the main app bundle.
    ///   Otherwise, downloads required models from Hugging Face via `ModelDownloader`.
    public func initialize(fromBundle: Bool = false) async {
        guard !isReady && !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0.0
        statusMessage = "Đang kiểm tra và chuẩn bị mô hình..."
        errorMessage = nil

        do {
            let provider: any ModelProvider
            if fromBundle && Bundle.main.url(forResource: "config", withExtension: "json") != nil {
                provider = BundleModelProvider(bundle: .main)
                downloadProgress = 1.0
            } else {
                let downloader = ModelDownloader()
                provider = try await downloader.download { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.downloadProgress = progress
                        self.statusMessage = "Đang tải mô hình: \(Int(progress * 100))%"
                    }
                }
            }

            self.modelProvider = provider
            let config = VieNeuConfig(
                modelProvider: provider,
                eulerSteps: eulerSteps,
                speed: speed
            )

            let ttsInstance = try await VieNeuTTS(config: config)
            self.tts = ttsInstance

            // Load preset voices
            var voices: [VoiceProfile] = []
            let minhQuan = (try? ttsInstance.loadPresetVoice(name: "Minh Quân")) ?? VoiceProfile.mock(name: "Minh Quân")
            voices.append(minhQuan)

            if let maiAnh = try? ttsInstance.loadPresetVoice(name: "Mai Anh") {
                voices.append(maiAnh)
            } else {
                voices.append(VoiceProfile.mock(name: "Mai Anh"))
            }

            // Load enrolled custom voices from disk
            let customVoices = loadCustomVoicesFromDisk()
            voices.append(contentsOf: customVoices)

            self.availableVoices = voices
            self.selectedVoice = minhQuan
            self.isReady = true
            self.isDownloading = false
            self.statusMessage = "Mô hình v3 Nano đã sẵn sàng!"
        } catch {
            // Fallback to mock configuration in testing or preview environments
            if tts == nil {
                let mockConfig = VieNeuConfig.mock(eulerSteps: eulerSteps, speed: speed)
                self.tts = try? await VieNeuTTS(config: mockConfig)
                self.isReady = (self.tts != nil)
            }
            self.isDownloading = false
            self.errorMessage = "Khởi tạo mô hình: \(error.localizedDescription)"
            self.statusMessage = "Chế độ thử nghiệm sẵn sàng."
        }
    }

    // MARK: - Speech Synthesis

    /// Synthesizes complete input text into an audio waveform.
    ///
    /// - Parameter text: Vietnamese text to synthesize.
    public func synthesizeText(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            errorMessage = "Vui lòng nhập văn bản cần đọc."
            return
        }

        guard isReady, let currentTTS = tts else {
            errorMessage = "Hệ thống đang tải mô hình, vui lòng đợi trong giây lát."
            return
        }

        stopPlayback()
        isSynthesizing = true
        statusMessage = "Đang tổng hợp giọng nói..."
        errorMessage = nil

        let startTime = Date()

        do {
            let activeTTS = try await prepareTTSInstance(reusing: currentTTS)
            let audioBuffer = try await activeTTS.synthesize(text: trimmed, voice: selectedVoice)

            let elapsed = Date().timeIntervalSince(startTime)
            self.generationTime = elapsed
            self.audioDuration = audioBuffer.duration
            self.rtf = audioBuffer.duration > 0 ? (elapsed / audioBuffer.duration) : 0.0

            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("tts_\(UUID().uuidString).wav")
            try audioBuffer.writeWAV(to: tempURL)
            self.synthesizedAudioURL = tempURL
            self.isSynthesizing = false
            self.statusMessage = String(format: "Hoàn tất! Thời lượng: %.2fs | RTF: %.2f", audioBuffer.duration, rtf)

            // Auto-play synthesized result
            playAudio(from: tempURL)
        } catch {
            self.isSynthesizing = false
            self.errorMessage = "Lỗi tổng hợp giọng nói: \(error.localizedDescription)"
            self.statusMessage = "Tổng hợp thất bại."
        }
    }

    /// Synthesizes input text sentence-by-sentence in a streaming manner.
    ///
    /// - Parameter text: Vietnamese text to stream.
    public func synthesizeStreamText(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            errorMessage = "Vui lòng nhập văn bản cần đọc."
            return
        }

        guard isReady, let currentTTS = tts else {
            errorMessage = "Hệ thống chưa sẵn sàng."
            return
        }

        stopPlayback()
        isSynthesizing = true
        isStreamingActive = true
        activeStreamChunks = []
        statusMessage = "Đang bắt đầu phát trực tiếp..."
        errorMessage = nil

        let startTime = Date()
        var allSamples: [Float] = []

        do {
            let activeTTS = try await prepareTTSInstance(reusing: currentTTS)
            let stream = activeTTS.synthesizeStream(text: trimmed, voice: selectedVoice)

            for try await chunk in stream {
                self.activeStreamChunks.append(chunk)
                allSamples.append(contentsOf: chunk.audio.samples)
                if chunk.pauseAfter > 0 {
                    let silenceSamples = Int(chunk.pauseAfter * 24000.0)
                    allSamples.append(contentsOf: [Float](repeating: 0.0, count: silenceSamples))
                }

                self.statusMessage = "Đã nhận câu \(chunk.index + 1): \"\(chunk.text.prefix(24))...\""

                // Auto-play first chunk to minimize initial latency
                if chunk.index == 0 {
                    let firstURL = FileManager.default.temporaryDirectory.appendingPathComponent("chunk_0_\(UUID().uuidString).wav")
                    try? chunk.audio.writeWAV(to: firstURL)
                    playAudio(from: firstURL)
                }
            }

            let elapsed = Date().timeIntervalSince(startTime)
            let combinedBuffer = AudioBuffer(samples: allSamples, sampleRate: 24000)
            self.generationTime = elapsed
            self.audioDuration = combinedBuffer.duration
            self.rtf = combinedBuffer.duration > 0 ? (elapsed / combinedBuffer.duration) : 0.0

            let combinedURL = FileManager.default.temporaryDirectory.appendingPathComponent("stream_full_\(UUID().uuidString).wav")
            try combinedBuffer.writeWAV(to: combinedURL)
            self.synthesizedAudioURL = combinedURL

            self.isSynthesizing = false
            self.isStreamingActive = false
            self.statusMessage = String(format: "Hoàn tất phát luồng (%d câu)! RTF: %.2f", activeStreamChunks.count, rtf)
        } catch {
            self.isSynthesizing = false
            self.isStreamingActive = false
            self.errorMessage = "Lỗi phát trực tiếp: \(error.localizedDescription)"
            self.statusMessage = "Phát luồng bị gián đoạn."
        }
    }

    // MARK: - Voice Cloning

    /// Enrolls a new custom voice profile from the active microphone recording.
    ///
    /// - Parameter name: Name identifier for the new voice profile.
    public func cloneVoiceFromRecording(name: String) async {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedName.count >= 2 else {
            errorMessage = "Vui lòng nhập tên giọng nói (tối thiểu 2 ký tự)."
            return
        }

        guard let recordedURL = recordedAudioURL else {
            errorMessage = "Chưa có bản ghi âm mẫu. Vui lòng ghi âm trước khi nhân bản."
            return
        }

        guard recordingDuration >= 1.0 else {
            errorMessage = "Thời lượng ghi âm quá ngắn (cần từ 3 đến 8 giây)."
            return
        }

        guard isReady, let currentTTS = tts else {
            errorMessage = "Hệ thống chưa sẵn sàng."
            return
        }

        isSynthesizing = true
        statusMessage = "Đang phân tích đặc trưng giọng nói..."
        errorMessage = nil

        do {
            let profile = try await currentTTS.cloneVoice(from: recordedURL, name: trimmedName)
            saveCustomVoiceToDisk(profile)

            self.availableVoices.append(profile)
            self.selectedVoice = profile
            self.isSynthesizing = false
            self.statusMessage = "Nhân bản giọng '\(trimmedName)' thành công!"
        } catch {
            self.isSynthesizing = false
            self.errorMessage = "Nhân bản giọng thất bại: \(error.localizedDescription)"
            self.statusMessage = "Không thể nhân bản giọng nói."
        }
    }

    // MARK: - Audio Recording

    /// Starts recording a voice sample from the device microphone (target: 3 to 8 seconds).
    public func startRecording() {
        #if canImport(AVFoundation)
        stopPlayback()
        errorMessage = nil

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
            try audioSession.setActive(true)
        } catch {
            errorMessage = "Không thể khởi động bộ ghi âm: \(error.localizedDescription)"
            return
        }

        let permission = audioSession.recordPermission
        switch permission {
        case .undetermined:
            audioSession.requestRecordPermission { [weak self] granted in
                Task { @MainActor [weak self] in
                    if granted {
                        self?.beginRecordingSession()
                    } else {
                        self?.errorMessage = "Vui lòng cấp quyền Microphone trong Cài đặt để ghi âm mẫu giọng."
                    }
                }
            }
        case .denied:
            errorMessage = "Quyền truy cập Microphone bị từ chối. Vui lòng bật trong Cài đặt."
        case .granted:
            beginRecordingSession()
        @unknown default:
            beginRecordingSession()
        }
        #else
        isRecording = true
        recordingDuration = 3.0
        statusMessage = "Đang ghi âm (Mô phỏng)..."
        #endif
    }

    /// Stops the active microphone recording session.
    public func stopRecording() {
        #if canImport(AVFoundation)
        recordingTimer?.invalidate()
        recordingTimer = nil

        guard isRecording else { return }
        audioRecorder?.stop()
        isRecording = false
        statusMessage = String(format: "Đã ghi âm xong (%.1fs)", recordingDuration)
        #else
        isRecording = false
        statusMessage = "Đã hoàn thành ghi âm mô phỏng."
        #endif
    }

    #if canImport(AVFoundation)
    private func beginRecordingSession() {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("voice_recording_\(UUID().uuidString).wav")
        self.recordedAudioURL = tempURL

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 24000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        do {
            let recorder = try AVAudioRecorder(url: tempURL, settings: settings)
            recorder.delegate = audioDelegate
            recorder.prepareToRecord()
            recorder.record()

            self.audioRecorder = recorder
            self.isRecording = true
            self.recordingDuration = 0.0
            self.statusMessage = "Đang ghi âm mẫu giọng (3-8 giây)..."

            recordingTimer?.invalidate()
            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
                Task { @MainActor [weak self] in
                    guard let self = self, self.isRecording else {
                        timer.invalidate()
                        return
                    }
                    self.recordingDuration += 0.1
                    if self.recordingDuration >= 8.0 {
                        self.stopRecording()
                    }
                }
            }
        } catch {
            self.errorMessage = "Không thể ghi âm: \(error.localizedDescription)"
            self.isRecording = false
        }
    }
    #endif

    // MARK: - Audio Playback

    /// Plays synthesized audio from the beginning.
    public func playSynthesizedAudio() {
        guard let url = synthesizedAudioURL else { return }
        playAudio(from: url)
    }

    /// Plays the recorded microphone sample.
    public func playRecordedAudio() {
        guard let url = recordedAudioURL else { return }
        playAudio(from: url)
    }

    /// Plays an audio file from the specified URL.
    public func playAudio(from url: URL) {
        #if canImport(AVFoundation)
        stopPlayback()

        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)

            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = audioDelegate
            player.prepareToPlay()
            player.play()

            self.audioPlayer = player
            self.isPlaying = true
            self.playbackProgress = 0.0

            playbackTimer?.invalidate()
            playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] timer in
                Task { @MainActor [weak self] in
                    guard let self = self, let p = self.audioPlayer, self.isPlaying else {
                        timer.invalidate()
                        return
                    }
                    if p.duration > 0 {
                        self.playbackProgress = min(1.0, p.currentTime / p.duration)
                    }
                }
            }
        } catch {
            errorMessage = "Lỗi phát âm thanh: \(error.localizedDescription)"
            isPlaying = false
        }
        #else
        isPlaying = true
        statusMessage = "Đang phát âm thanh (Mô phỏng)..."
        #endif
    }

    /// Pauses ongoing playback.
    public func pausePlayback() {
        #if canImport(AVFoundation)
        audioPlayer?.pause()
        isPlaying = false
        #else
        isPlaying = false
        #endif
    }

    /// Stops playback and resets progress.
    public func stopPlayback() {
        #if canImport(AVFoundation)
        playbackTimer?.invalidate()
        playbackTimer = nil
        audioPlayer?.stop()
        audioPlayer = nil
        isPlaying = false
        playbackProgress = 0.0
        #else
        isPlaying = false
        playbackProgress = 0.0
        #endif
    }

    // MARK: - Voice Profile Export & Import

    /// Exports a voice profile to a shareable JSON file.
    ///
    /// - Parameter profile: Voice profile to export.
    /// - Returns: File URL to the exported JSON file.
    public func exportVoiceProfile(_ profile: VoiceProfile) -> URL? {
        let safeName = profile.name.replacingOccurrences(of: " ", with: "_")
        let exportURL = FileManager.default.temporaryDirectory.appendingPathComponent("voice_\(safeName).json")
        do {
            try profile.export(to: exportURL)
            return exportURL
        } catch {
            errorMessage = "Xuất giọng nói thất bại: \(error.localizedDescription)"
            return nil
        }
    }

    /// Imports a voice profile from a JSON file URL.
    ///
    /// - Parameter url: Source JSON file URL.
    public func importVoiceProfile(from url: URL) {
        do {
            let profile = try VoiceProfile.load(from: url)
            saveCustomVoiceToDisk(profile)
            if !availableVoices.contains(where: { $0.name == profile.name }) {
                availableVoices.append(profile)
            }
            selectedVoice = profile
            statusMessage = "Đã nhập giọng '\(profile.name)' thành công!"
        } catch {
            errorMessage = "Nhập giọng nói thất bại: \(error.localizedDescription)"
        }
    }

    // MARK: - Private Helpers

    /// Reuses existing ONNX sessions while updating synthesis parameters like eulerSteps and speed.
    private func prepareTTSInstance(reusing existing: VieNeuTTS) async throws -> VieNeuTTS {
        if existing.config.eulerSteps == eulerSteps && abs(existing.config.speed - speed) < 1e-4 {
            return existing
        }

        let engine = await existing.engine
        let newConfig = VieNeuConfig(
            modelProvider: existing.config.modelProvider,
            eulerSteps: eulerSteps,
            speed: speed,
            onnxEngineOverride: engine
        )
        let updated = try await VieNeuTTS(config: newConfig)
        self.tts = updated
        return updated
    }

    private func customVoicesDirectory() -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("CustomVoices")
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    private func saveCustomVoiceToDisk(_ profile: VoiceProfile) {
        let dir = customVoicesDirectory()
        let safeName = profile.name.replacingOccurrences(of: " ", with: "_")
        let fileURL = dir.appendingPathComponent("\(safeName).json")
        try? profile.export(to: fileURL)
    }

    private func loadCustomVoicesFromDisk() -> [VoiceProfile] {
        let dir = customVoicesDirectory()
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            return []
        }

        var loaded: [VoiceProfile] = []
        for file in files where file.pathExtension.lowercased() == "json" {
            if let profile = try? VoiceProfile.load(from: file) {
                loaded.append(profile)
            }
        }
        return loaded
    }

    #if canImport(AVFoundation)
    private func setupAudioDelegates() {
        audioDelegate.onPlayerDidFinish = { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.isPlaying = false
                self.playbackProgress = 0.0
                self.playbackTimer?.invalidate()
                self.playbackTimer = nil
            }
        }

        audioDelegate.onRecorderDidFinish = { [weak self] success in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.isRecording = false
                self.recordingTimer?.invalidate()
                self.recordingTimer = nil
                if !success {
                    self.errorMessage = "Quá trình ghi âm gặp sự cố."
                }
            }
        }
    }
    #endif
}

#if canImport(AVFoundation)
/// Helper object managing AVFoundation delegate callbacks across thread boundaries.
private final class AudioDelegateHelper: NSObject, AVAudioPlayerDelegate, AVAudioRecorderDelegate, @unchecked Sendable {
    var onPlayerDidFinish: (@Sendable (Bool) -> Void)?
    var onRecorderDidFinish: (@Sendable (Bool) -> Void)?

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onPlayerDidFinish?(flag)
    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        onRecorderDidFinish?(flag)
    }
}
#endif
