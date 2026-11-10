#if canImport(SwiftUI)
import SwiftUI
import VieNeuTTS

/// View providing text-to-speech synthesis controls, voice selection, tuning knobs, and metrics.
public struct SynthesisView: View {
    @ObservedObject public var viewModel: TTSViewModel

    @State private var inputText: String = "Xin chào! Tôi là mô hình VieNeu-TTS v3 Nano chạy trực tiếp trên thiết bị Apple."
    @FocusState private var isTextEditorFocused: Bool

    private let sampleSentences = [
        "Xin chào! Tôi là mô hình VieNeu-TTS v3 Nano chạy trực tiếp trên thiết bị Apple.",
        "Hôm nay thời tiết thế nào? Bạn có kế hoạch gì chưa?",
        "Hệ thống trí tuệ nhân tạo tạo giọng nói tiếng Việt tự nhiên và truyền cảm.",
        "Công nghệ Flow-Matching giúp tạo âm thanh chất lượng cao với tốc độ siêu nhanh."
    ]

    public init(viewModel: TTSViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header Banner
                headerSection

                // Text Input & Quick Samples
                textInputSection

                // Voice Picker & Synthesis Configuration
                configurationSection

                // Action Buttons (Full Synthesize & Streaming)
                actionButtonsSection

                // Performance Metrics Bento Grid
                if viewModel.generationTime > 0 || viewModel.audioDuration > 0 {
                    metricsSection
                }

                // Audio Playback Controls
                if viewModel.synthesizedAudioURL != nil {
                    playbackSection
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .navigationTitle("Tổng Hợp Giọng Nói")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Xong") {
                    isTextEditorFocused = false
                }
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "waveform.circle.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 36, height: 36)
                    .foregroundColor(.accentColor)

                VStack(alignment: .leading, spacing: 4) {
                    Text(viewModel.selectedMode == .v3turbo ? "VieNeu-TTS v3 Turbo (48 kHz)" : "VieNeu-TTS v3 Nano (24 kHz)")
                        .font(.headline)
                        .fontWeight(.bold)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()

                // Ready Badge
                HStack(spacing: 4) {
                    Circle()
                        .fill(viewModel.isReady ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    Text(viewModel.isReady ? "Sẵn sàng" : "Đang tải")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(viewModel.isReady ? .green : .orange)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(viewModel.isReady ? Color.green.opacity(0.12) : Color.orange.opacity(0.12))
                )
            }

            // Architecture Mode Switcher
            Picker("Chế độ", selection: Binding(
                get: { viewModel.selectedMode },
                set: { newMode in
                    Task {
                        await viewModel.setMode(newMode)
                    }
                }
            )) {
                Text("v3 Nano (24 kHz)").tag(VieNeuModelMode.v3nano)
                Text("v3 Turbo (48 kHz)").tag(VieNeuModelMode.v3turbo)
            }
            .pickerStyle(.segmented)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var textInputSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Văn bản cần đọc")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
                Spacer()
                if !inputText.isEmpty {
                    Button(action: { inputText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                            .imageScale(.small)
                    }
                }
            }

            ZStack(alignment: .topLeading) {
                if inputText.isEmpty {
                    Text("Nhập văn bản tiếng Việt tại đây...")
                        .font(.body)
                        .foregroundColor(Color(uiColor: .placeholderText))
                        .padding(.top, 8)
                        .padding(.leading, 5)
                }

                TextEditor(text: $inputText)
                    .focused($isTextEditorFocused)
                    .frame(minHeight: 90, maxHeight: 140)
                    .padding(4)
                    .scrollContentBackground(.hidden)
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(uiColor: .tertiarySystemGroupedBackground))
            )

            // Quick Samples Chips
            VStack(alignment: .leading, spacing: 6) {
                Text("Mẫu gợi ý:")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(sampleSentences, id: \.self) { sample in
                            Button(action: {
                                inputText = sample
                            }) {
                                Text(sample)
                                    .font(.caption)
                                    .lineLimit(1)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(
                                        Capsule()
                                            .fill(Color(uiColor: .secondarySystemGroupedBackground))
                                    )
                                    .overlay(
                                        Capsule()
                                            .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                                    )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Voice Picker
            VStack(alignment: .leading, spacing: 8) {
                Text("Giọng đọc")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)

                Menu {
                    ForEach(viewModel.availableVoices, id: \.name) { voice in
                        Button(action: {
                            viewModel.selectedVoice = voice
                        }) {
                            HStack {
                                Text(voice.name)
                                if voice.name == viewModel.selectedVoice.name {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack {
                        Image(systemName: "person.wave.2.fill")
                            .foregroundColor(.accentColor)
                        Text(viewModel.selectedVoice.name)
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                        Spacer()
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(uiColor: .tertiarySystemGroupedBackground))
                    )
                }
            }

            Divider()

            // Euler Steps Control
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Số bước Euler ODE")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(viewModel.eulerSteps == 8 ? "8 bước (Nhanh)" : "16 bước (Chuẩn)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.accentColor)
                }

                Picker("Bước", selection: $viewModel.eulerSteps) {
                    Text("8 bước").tag(8)
                    Text("16 bước").tag(16)
                }
                .pickerStyle(.segmented)
            }

            // Speed Control
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Tốc độ đọc")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.1fx", viewModel.speed))
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.accentColor)
                }

                Slider(value: $viewModel.speed, in: 0.8...1.5, step: 0.1)
                    .tint(.accentColor)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var actionButtonsSection: some View {
        HStack(spacing: 12) {
            // Full Synthesis Button
            Button(action: {
                isTextEditorFocused = false
                Task {
                    await viewModel.synthesizeText(inputText)
                }
            }) {
                HStack {
                    if viewModel.isSynthesizing && !viewModel.isStreamingActive {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.9)
                    } else {
                        Image(systemName: "play.circle.fill")
                    }
                    Text(viewModel.isSynthesizing && !viewModel.isStreamingActive ? "Đang đọc..." : "Đọc văn bản")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(viewModel.isReady && !viewModel.isSynthesizing ? Color.accentColor : Color.gray.opacity(0.4))
                )
                .foregroundColor(.white)
            }
            .disabled(!viewModel.isReady || viewModel.isSynthesizing || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            // Streaming Synthesis Button
            Button(action: {
                isTextEditorFocused = false
                Task {
                    await viewModel.synthesizeStreamText(inputText)
                }
            }) {
                HStack {
                    if viewModel.isStreamingActive {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .accentColor))
                            .scaleEffect(0.9)
                    } else {
                        Image(systemName: "waveform.badge.magnifyingglass")
                    }
                    Text(viewModel.isStreamingActive ? "Đang luồng..." : "Phát luồng")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.accentColor.opacity(0.12))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.accentColor, lineWidth: 1)
                )
                .foregroundColor(.accentColor)
            }
            .disabled(!viewModel.isReady || viewModel.isSynthesizing || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Hiệu Suất Suy Luận (On-Device)")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            HStack(spacing: 10) {
                // Audio Duration
                metricCard(
                    title: "Thời Lượng",
                    value: String(format: "%.2fs", viewModel.audioDuration),
                    icon: "clock.fill",
                    color: .blue
                )

                // Generation Time
                metricCard(
                    title: "Thời Gian Sinh",
                    value: String(format: "%.2fs", viewModel.generationTime),
                    icon: "bolt.fill",
                    color: .purple
                )

                // Real-Time Factor (RTF)
                metricCard(
                    title: "Tỷ Lệ RTF",
                    value: String(format: "%.2fx", viewModel.rtf),
                    icon: "speedometer",
                    color: viewModel.rtf < 0.3 ? .green : (viewModel.rtf < 1.0 ? .orange : .red)
                )
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private func metricCard(title: String, value: String, icon: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundColor(color)
                Spacer()
            }
            Text(value)
                .font(.system(.title3, design: .rounded))
                .fontWeight(.bold)
                .foregroundColor(color)
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(uiColor: .tertiarySystemGroupedBackground))
        )
    }

    private var playbackSection: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Phát Lại Âm Thanh")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%.2fs", viewModel.audioDuration))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Scrubber / Progress Bar
            ProgressView(value: viewModel.playbackProgress, total: 1.0)
                .tint(.accentColor)

            HStack(spacing: 20) {
                Spacer()

                // Play / Pause Button
                Button(action: {
                    if viewModel.isPlaying {
                        viewModel.pausePlayback()
                    } else {
                        viewModel.playSynthesizedAudio()
                    }
                }) {
                    Image(systemName: viewModel.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .resizable()
                        .frame(width: 44, height: 44)
                        .foregroundColor(.accentColor)
                }

                // Stop Button
                Button(action: {
                    viewModel.stopPlayback()
                }) {
                    Image(systemName: "stop.circle.fill")
                        .resizable()
                        .frame(width: 36, height: 36)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }
}
#else
public struct SynthesisView {
    public init(viewModel: TTSViewModel) {}
}
#endif
