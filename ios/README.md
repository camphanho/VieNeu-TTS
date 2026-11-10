# VieNeuTTS for Apple Platforms

`VieNeuTTS` is a Swift Package for the v3 Nano on-device Vietnamese text-to-speech pipeline. It builds on Linux for unit testing and on Apple platforms for ONNX Runtime inference and SeaG2P phonemization.

## Requirements

- Swift 5.9 or newer
- macOS with Xcode and Rust only when building the SeaG2P binary framework
- Downloaded v3 Nano ONNX model assets at application runtime

## Build SeaG2P

Apple builds use `Frameworks/SeaG2P.xcframework`. Generate it from the repository root on macOS:

```bash
bash ios/scripts/build_seag2p.sh
```

The helper builds `ios/SeaG2PBridge` for iOS devices and both simulator architectures, combines simulator libraries, and writes:

```text
ios/VieNeuTTS/Frameworks/SeaG2P.xcframework
```

It requires Xcode command-line tools (`lipo`, `xcodebuild`) and Rust targets supplied through `rustup`. Commit or otherwise distribute the generated XCFramework with the Apple app that consumes this package; do not run the helper on Linux.

## Swift Package builds

```bash
swift test --package-path ios/VieNeuTTS
```

`Package.swift` conditionally adds ONNX Runtime and the binary SeaG2P target only on Apple platforms. Linux test runners instead compile the C SeaG2P shim, keeping core unit tests hermetic without an Apple XCFramework or ONNX Runtime.
