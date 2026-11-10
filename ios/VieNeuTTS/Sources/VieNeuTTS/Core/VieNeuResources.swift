import Foundation

/// Universal resource locator for VieNeu-TTS bundled assets (dictionaries, voices, constants).
public enum VieNeuResources: Sendable {
    /// Locates a bundled asset by full filename (e.g. `"sea_g2p.bin"`, `"voices_v3_nano.json"`).
    public static func url(for filename: String) -> URL? {
        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension

        // 1. Direct check in Bundle.main
        if let u = Bundle.main.url(forResource: name, withExtension: ext) { return u }
        if let u = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Resources") { return u }

        // 2. Check all loaded bundles (covers SPM resource bundles)
        for b in Bundle.allBundles {
            if let u = b.url(forResource: name, withExtension: ext) { return u }
            if let u = b.url(forResource: name, withExtension: ext, subdirectory: "Resources") { return u }
        }

        // 3. Direct inspection in main bundle container subdirectories and test paths
        let mainDir = Bundle.main.bundleURL
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidatePaths = [
            mainDir.appendingPathComponent("Resources/\(filename)"),
            mainDir.appendingPathComponent(filename),
            mainDir.appendingPathComponent("VieNeuTTSDemo_VieNeuTTSDemo.bundle/Resources/\(filename)"),
            mainDir.appendingPathComponent("VieNeuTTSDemo_VieNeuTTSDemo.bundle/\(filename)"),
            cwd.appendingPathComponent("src/vieneu/assets/\(filename)"),
            cwd.appendingPathComponent("assets/\(filename)"),
            cwd.appendingPathComponent("../../src/vieneu/assets/\(filename)"),
            cwd.appendingPathComponent("../../../src/vieneu/assets/\(filename)"),
            cwd.appendingPathComponent(".venv/lib/python3.12/site-packages/sea_g2p/\(filename)"),
            cwd.appendingPathComponent("../../.venv/lib/python3.12/site-packages/sea_g2p/\(filename)"),
            cwd.appendingPathComponent("../../../.venv/lib/python3.12/site-packages/sea_g2p/\(filename)")
        ]
        for path in candidatePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                return path
            }
        }

        return nil
    }
}
