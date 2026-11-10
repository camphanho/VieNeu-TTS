// swift-tools-version: 5.9
import PackageDescription
import Foundation

let packageDirectory = Context.packageDirectory
let xcframeworkPath = "\(packageDirectory)/Frameworks/SeaG2P.xcframework"

#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
let hasXCFramework = FileManager.default.fileExists(atPath: xcframeworkPath)
#else
let hasXCFramework = false
#endif

var packageDependencies: [Package.Dependency] = []
var vieNeuDependencies: [Target.Dependency] = []
var targets: [Target] = []

#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
packageDependencies.append(
    .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0")
)
vieNeuDependencies.append(
    .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
)
#endif

if hasXCFramework {
    targets.append(
        .binaryTarget(
            name: "SeaG2P",
            path: "Frameworks/SeaG2P.xcframework"
        )
    )
    vieNeuDependencies.append("SeaG2P")
} else {
    let bridgeDir = "\(packageDirectory)/../SeaG2PBridge"
    let staticLibRelease = "\(bridgeDir)/target/release/libsea_g2p_bridge.a"
    let staticLibDebug = "\(bridgeDir)/target/debug/libsea_g2p_bridge.a"
    let staticLib = FileManager.default.fileExists(atPath: staticLibRelease) ? staticLibRelease : staticLibDebug

    var linkerFlags: [String] = []
    if FileManager.default.fileExists(atPath: staticLib) {
        linkerFlags.append(staticLib)
    } else {
        linkerFlags.append(contentsOf: ["-L\(bridgeDir)/target/debug", "-lsea_g2p_bridge"])
    }
    #if os(Linux)
    linkerFlags.append(contentsOf: ["-lpthread", "-ldl", "-lm"])
    #endif

    targets.append(
        .target(
            name: "CSeaG2P",
            path: "Sources/CSeaG2P",
            publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags(linkerFlags)
            ]
        )
    )
    vieNeuDependencies.append("CSeaG2P")
}

targets.append(
    .target(
        name: "VieNeuTTS",
        dependencies: vieNeuDependencies,
        path: "Sources/VieNeuTTS"
    )
)

targets.append(
    .testTarget(
        name: "VieNeuTTSTests",
        dependencies: ["VieNeuTTS"],
        path: "Tests/VieNeuTTSTests"
    )
)

let package = Package(
    name: "VieNeuTTS",
    platforms: [
        .iOS(.v16),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "VieNeuTTS",
            targets: ["VieNeuTTS"]
        )
    ],
    dependencies: packageDependencies,
    targets: targets
)
