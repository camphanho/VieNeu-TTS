#!/usr/bin/env bash
set -euo pipefail

# ios/scripts/build_seag2p.sh
#
# Compiles SeaG2PBridge Rust crate for iOS targets and packages them
# into SeaG2P.xcframework for VieNeuTTS Swift Package.
#
# Targets:
#   - aarch64-apple-ios       (Physical iOS device - arm64)
#   - aarch64-apple-ios-sim   (iOS Simulator on Apple Silicon - arm64)
#   - x86_64-apple-ios / sim  (iOS Simulator on Intel - x86_64)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BRIDGE_DIR="${REPO_ROOT}/ios/SeaG2PBridge"
OUTPUT_DIR="${REPO_ROOT}/ios/VieNeuTTS/Frameworks"
XCFRAMEWORK_PATH="${OUTPUT_DIR}/SeaG2P.xcframework"
TARGET_DIR="${BRIDGE_DIR}/target"

echo "=== Building SeaG2P Bridge XCFramework ==="
echo "Repo root:       ${REPO_ROOT}"
echo "Bridge crate:    ${BRIDGE_DIR}"
echo "Target output:   ${XCFRAMEWORK_PATH}"

# Check for cargo
if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo is required but not installed." >&2
    exit 1
fi

# Detect x86_64 simulator target name supported by rustc
X86_SIM_TARGET="x86_64-apple-ios"
if rustc --print target-list | grep -q "^x86_64-apple-ios-sim$"; then
    X86_SIM_TARGET="x86_64-apple-ios-sim"
fi

TARGETS=(
    "aarch64-apple-ios"
    "aarch64-apple-ios-sim"
    "${X86_SIM_TARGET}"
)

# Add targets via rustup if available
if command -v rustup >/dev/null 2>&1; then
    echo "--> Ensuring rustup targets are installed..."
    for target in "${TARGETS[@]}"; do
        rustup target add "${target}" 2>/dev/null || true
    done
fi

echo "--> Compiling Rust C-ABI bridge for targets..."
for target in "${TARGETS[@]}"; do
    echo "  Compiling for ${target}..."
    cargo build --manifest-path "${BRIDGE_DIR}/Cargo.toml" \
        --target "${target}" \
        --release
done

# Check if lipo and xcodebuild exist (requires macOS with Xcode)
if ! command -v lipo >/dev/null 2>&1; then
    echo "Warning: lipo not found. Building XCFramework requires macOS."
    echo "Compiled static libraries are located in ${TARGET_DIR}."
    exit 0
fi

if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "Warning: xcodebuild not found. Packaging XCFramework requires macOS with Xcode."
    echo "Compiled static libraries are located in ${TARGET_DIR}."
    exit 0
fi

SIM_DIR="${TARGET_DIR}/universal-sim"
mkdir -p "${SIM_DIR}"

echo "--> Creating universal simulator static library with lipo..."
lipo -create \
    "${TARGET_DIR}/aarch64-apple-ios-sim/release/libsea_g2p_bridge.a" \
    "${TARGET_DIR}/${X86_SIM_TARGET}/release/libsea_g2p_bridge.a" \
    -output "${SIM_DIR}/libsea_g2p_bridge.a"

echo "--> Packaging SeaG2P.xcframework via xcodebuild..."
mkdir -p "${OUTPUT_DIR}"
rm -rf "${XCFRAMEWORK_PATH}"

xcodebuild -create-xcframework \
    -library "${TARGET_DIR}/aarch64-apple-ios/release/libsea_g2p_bridge.a" \
    -headers "${BRIDGE_DIR}/include" \
    -library "${SIM_DIR}/libsea_g2p_bridge.a" \
    -headers "${BRIDGE_DIR}/include" \
    -output "${XCFRAMEWORK_PATH}"

echo "=== Successfully built SeaG2P.xcframework at ${XCFRAMEWORK_PATH} ==="
