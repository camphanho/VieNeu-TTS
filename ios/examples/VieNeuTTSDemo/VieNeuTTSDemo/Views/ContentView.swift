#if canImport(SwiftUI)
import SwiftUI

/// Main container view providing modern tabbed navigation across speech synthesis and voice cloning.
public struct ContentView: View {
    @ObservedObject public var viewModel: TTSViewModel

    public init(viewModel: TTSViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ZStack(alignment: .top) {
            TabView {
                NavigationStack {
                    SynthesisView(viewModel: viewModel)
                }
                .tabItem {
                    Label("Đọc Văn Bản", systemImage: "waveform.and.mic")
                }

                NavigationStack {
                    VoiceCloningView(viewModel: viewModel)
                }
                .tabItem {
                    Label("Nhân Bản Giọng", systemImage: "person.wave.2.fill")
                }
            }

            // Model Download Overlay Banner
            if viewModel.isDownloading {
                downloadBanner
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .zIndex(100)
            }
        }
        .animation(.easeInOut, value: viewModel.isDownloading)
        .alert("Thông Báo", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.errorMessage = nil } }
        )) {
            Button("Đóng", role: .cancel) {
                viewModel.errorMessage = nil
            }
        } message: {
            if let err = viewModel.errorMessage {
                Text(err)
            }
        }
    }

    private var downloadBanner: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                ProgressView()
                    .scaleEffect(0.9)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Đang tải mô hình VieNeu-TTS v3 Nano...")
                        .font(.caption)
                        .fontWeight(.semibold)
                    Text("\(Int(viewModel.downloadProgress * 100))%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }

            ProgressView(value: viewModel.downloadProgress, total: 1.0)
                .tint(.accentColor)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(uiColor: .systemBackground))
                .shadow(color: Color.black.opacity(0.12), radius: 8, x: 0, y: 4)
        )
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }
}
#else
public struct ContentView {
    public init(viewModel: TTSViewModel) {}
}
#endif
