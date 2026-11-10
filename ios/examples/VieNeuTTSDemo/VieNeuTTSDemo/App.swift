#if canImport(SwiftUI)
import SwiftUI

/// Main application entrypoint for the VieNeuTTS iOS demo app.
@main
public struct VieNeuTTSDemoApp: App {
    @StateObject private var viewModel = TTSViewModel()

    public init() {}

    public var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .task {
                    await viewModel.initialize()
                }
        }
    }
}
#else
public struct VieNeuTTSDemoApp {
    public init() {}
}
#endif
