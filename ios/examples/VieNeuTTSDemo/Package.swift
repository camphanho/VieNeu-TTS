// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VieNeuTTSDemo",
    platforms: [
        .iOS(.v16),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "VieNeuTTSDemo",
            targets: ["VieNeuTTSDemo"]
        )
    ],
    dependencies: [
        .package(path: "../../VieNeuTTS")
    ],
    targets: [
        .target(
            name: "VieNeuTTSDemo",
            dependencies: [
                .product(name: "VieNeuTTS", package: "VieNeuTTS")
            ],
            path: "VieNeuTTSDemo",
            exclude: ["Info.plist"]
        )
    ]
)
