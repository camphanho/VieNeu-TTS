#if canImport(SwiftUI)
import SwiftUI
import VieNeuTTS

#if canImport(UIKit)
import UIKit
#endif

/// View providing microphone recording, voice enrollment, and voice profile sharing.
public struct VoiceCloningView: View {
    @ObservedObject public var viewModel: TTSViewModel

    @State private var voiceName: String = ""
    @State private var exportedShareURL: URL?
    @State private var isShowingShareSheet: Bool = false
    @FocusState private var isNameFieldFocused: Bool

    public init(viewModel: TTSViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Informational Banner
                instructionBanner

                // Audio Recording Station
                recordingStationSection

                // Recorded Sample Preview & Voice Enrollment
                if viewModel.recordedAudioURL != nil && !viewModel.isRecording {
                    enrollmentSection
                }

                // Voice Library & Export
                voiceLibrarySection
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .navigationTitle("Nhân Bản Giọng Nói")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $isShowingShareSheet) {
            if let shareURL = exportedShareURL {
                ActivityView(activityItems: [shareURL])
            }
        }
    }

    // MARK: - Subviews

    private var instructionBanner: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "sparkles")
                .font(.title3)
                .foregroundColor(.accentColor)

            VStack(alignment: .leading, spacing: 4) {
                Text("Nhân bản giọng tức thì (Instant Voice Cloning)")
                    .font(.subheadline)
                    .fontWeight(.bold)
                Text("Thu âm từ 3 đến 8 giây giọng nói mẫu bằng micro. Mô hình sẽ phân tích âm sắc x-vector và phong cách phát âm trên thiết bị để tạo giọng mới.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var recordingStationSection: some View {
        VStack(spacing: 16) {
            Text("Thu Âm Mẫu Giọng")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Timer & Countdown Display
            VStack(spacing: 6) {
                Text(String(format: "%.1f s", viewModel.recordingDuration))
                    .font(.system(size: 38, weight: .bold, design: .rounded))
                    .foregroundColor(timerColor)

                Text(recordingGuidanceText)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
            }

            // Progress Bar (0 to 8s)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color(uiColor: .tertiarySystemGroupedBackground))
                        .frame(height: 10)

                    RoundedRectangle(cornerRadius: 6)
                        .fill(timerColor)
                        .frame(width: geo.size.width * CGFloat(min(1.0, viewModel.recordingDuration / 8.0)), height: 10)

                    // 3-second minimum marker
                    Rectangle()
                        .fill(Color.primary.opacity(0.3))
                        .frame(width: 2, height: 14)
                        .offset(x: geo.size.width * (3.0 / 8.0) - 1, y: -2)
                }
            }
            .frame(height: 14)
            .padding(.horizontal, 20)

            // Record / Stop Circular Button
            Button(action: {
                if viewModel.isRecording {
                    viewModel.stopRecording()
                } else {
                    viewModel.startRecording()
                }
            }) {
                ZStack {
                    Circle()
                        .fill(viewModel.isRecording ? Color.red.opacity(0.2) : Color.accentColor.opacity(0.15))
                        .frame(width: 86, height: 86)
                        .scaleEffect(viewModel.isRecording ? 1.15 : 1.0)
                        .animation(viewModel.isRecording ? Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true) : .default, value: viewModel.isRecording)

                    Circle()
                        .fill(viewModel.isRecording ? Color.red : Color.accentColor)
                        .frame(width: 68, height: 68)

                    Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                }
            }
            .padding(.vertical, 8)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var enrollmentSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Nghe Lại & Nhân Bản")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            // Live Playback Controls for Recorded Clip
            HStack(spacing: 12) {
                Button(action: {
                    if viewModel.isPlaying {
                        viewModel.pausePlayback()
                    } else {
                        viewModel.playRecordedAudio()
                    }
                }) {
                    HStack {
                        Image(systemName: viewModel.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        Text(viewModel.isPlaying ? "Tạm dừng" : "Nghe lại mẫu")
                    }
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        Capsule().fill(Color.accentColor.opacity(0.12))
                    )
                    .foregroundColor(.accentColor)
                }

                Spacer()

                Button(action: {
                    viewModel.startRecording()
                }) {
                    HStack {
                        Image(systemName: "arrow.counterclockwise")
                        Text("Thu lại")
                    }
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                }
            }

            Divider()

            // Voice Name Input
            VStack(alignment: .leading, spacing: 6) {
                Text("Tên giọng mới")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)

                TextField("Ví dụ: Giọng Nam Hà Nội, Giọng Minh Tú...", text: $voiceName)
                    .focused($isNameFieldFocused)
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(uiColor: .tertiarySystemGroupedBackground))
                    )
            }

            // Clone Button
            Button(action: {
                isNameFieldFocused = false
                Task {
                    await viewModel.cloneVoiceFromRecording(name: voiceName)
                    if viewModel.errorMessage == nil {
                        voiceName = ""
                    }
                }
            }) {
                HStack {
                    if viewModel.isSynthesizing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.9)
                        Text("Đang nhân bản giọng...")
                            .fontWeight(.semibold)
                    } else {
                        Image(systemName: "person.crop.circle.badge.plus")
                        Text("Sao chép giọng")
                            .fontWeight(.semibold)
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(canClone ? Color.accentColor : Color.gray.opacity(0.4))
                )
                .foregroundColor(.white)
            }
            .disabled(!canClone || viewModel.isSynthesizing)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private var voiceLibrarySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Danh Sách Giọng Đọc (\(viewModel.availableVoices.count))")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                ForEach(viewModel.availableVoices, id: \.name) { voice in
                    voiceRow(voice)
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(uiColor: .secondarySystemGroupedBackground))
        )
    }

    private func voiceRow(_ voice: VoiceProfile) -> some View {
        let isPreset = (voice.name == "Minh Quân" || voice.name == "Mai Anh")
        let isSelected = (voice.name == viewModel.selectedVoice.name)

        return HStack(spacing: 12) {
            Image(systemName: isPreset ? "waveform" : "person.fill.checkmark")
                .foregroundColor(isSelected ? .accentColor : .secondary)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(voice.name)
                        .font(.body)
                        .fontWeight(isSelected ? .bold : .regular)

                    if isSelected {
                        Text("Đang chọn")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.accentColor.opacity(0.15)))
                            .foregroundColor(.accentColor)
                    }
                }

                Text(isPreset ? "Giọng chuẩn hệ thống" : "Giọng đã nhân bản (Custom)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Export / Share Button
            Button(action: {
                if let url = viewModel.exportVoiceProfile(voice) {
                    self.exportedShareURL = url
                    self.isShowingShareSheet = true
                }
            }) {
                Image(systemName: "square.and.arrow.up")
                    .foregroundColor(.secondary)
                    .imageScale(.medium)
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(uiColor: .tertiarySystemGroupedBackground))
        )
        .onTapGesture {
            viewModel.selectedVoice = voice
        }
    }

    // MARK: - Helpers

    private var canClone: Bool {
        voiceName.trimmingCharacters(in: .whitespacesAndNewlines).count >= 2
            && viewModel.recordingDuration >= 1.0
            && viewModel.recordedAudioURL != nil
            && viewModel.isReady
    }

    private var timerColor: Color {
        if viewModel.recordingDuration < 3.0 {
            return .orange
        } else if viewModel.recordingDuration <= 8.0 {
            return .green
        } else {
            return .red
        }
    }

    private var recordingGuidanceText: String {
        if viewModel.isRecording {
            if viewModel.recordingDuration < 3.0 {
                return "Đang ghi âm... nói ít nhất 3 giây"
            } else {
                return "Đã đạt chuẩn! Có thể nhấn dừng bất cứ lúc nào."
            }
        } else if viewModel.recordingDuration > 0 {
            return "Đã ghi âm xong. Nhập tên và nhấn Sao chép giọng."
        } else {
            return "Nhấn nút micro để bắt đầu ghi âm (3-8 giây)."
        }
    }
}

#if canImport(UIKit)
/// Helper UIKit wrapper presenting the standard system share sheet.
private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
#else
public struct VoiceCloningView {
    public init(viewModel: TTSViewModel) {}
}
#endif
