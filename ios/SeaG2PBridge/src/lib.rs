//! C-ABI bridge for sea-g2p on Apple platforms (iOS / macOS).
//!
//! Exposes punctuation normalization and grapheme-to-phoneme conversion
//! through a minimal C-ABI surface for embedding in Swift and C/C++ hosts.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::OnceLock;

use sea_g2p::g2p::G2PEngine;
use sea_g2p::lang::vi::Normalizer;

static NORMALIZER: OnceLock<Normalizer> = OnceLock::new();
static ENGINE: OnceLock<Option<G2PEngine>> = OnceLock::new();
static CUSTOM_DICT_PATH: OnceLock<String> = OnceLock::new();

/// Returns the singleton Normalizer for Vietnamese text.
fn get_normalizer() -> &'static Normalizer {
    NORMALIZER.get_or_init(|| Normalizer::new("vi", None))
}

/// Sets an explicit path for the sea_g2p.bin dictionary.
pub fn set_dict_path(path: &str) {
    let _ = CUSTOM_DICT_PATH.set(path.to_string());
}

/// Searches candidate filesystem locations for sea_g2p.bin.
fn find_dict_path() -> Option<PathBuf> {
    // 1. Explicitly configured custom path
    if let Some(custom) = CUSTOM_DICT_PATH.get() {
        let p = PathBuf::from(custom);
        if p.exists() {
            return Some(p);
        }
    }

    // 2. Environment variable
    if let Ok(env_path) = std::env::var("SEA_G2P_DICT_PATH") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }

    // 3. Search common locations relative to current working directory and repo root
    let candidates = [
        "sea_g2p.bin",
        "Resources/sea_g2p.bin",
        "Frameworks/sea_g2p.bin",
        "VieNeuTTSDemo_VieNeuTTSDemo.bundle/Resources/sea_g2p.bin",
        "VieNeuTTSDemo_VieNeuTTSDemo.bundle/sea_g2p.bin",
        ".venv/lib/python3.12/site-packages/sea_g2p/sea_g2p.bin",
        ".venv/lib/python3.11/site-packages/sea_g2p/sea_g2p.bin",
        ".venv/lib/python3.10/site-packages/sea_g2p/sea_g2p.bin",
        "../../.venv/lib/python3.12/site-packages/sea_g2p/sea_g2p.bin",
        "../../.venv/lib/python3.11/site-packages/sea_g2p/sea_g2p.bin",
        "../../.venv/lib/python3.10/site-packages/sea_g2p/sea_g2p.bin",
        "../../../.venv/lib/python3.12/site-packages/sea_g2p/sea_g2p.bin",
    ];

    if let Ok(cwd) = std::env::current_dir() {
        for rel in &candidates {
            let candidate = cwd.join(rel);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            for rel in &candidates {
                let candidate = exe_dir.join(rel);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Creates a minimal valid SEAP v1 empty binary dictionary in the temp directory.
fn create_fallback_dict() -> std::io::Result<PathBuf> {
    let tmp_path = std::env::temp_dir().join("sea_g2p_minimal_fallback.bin");
    if !tmp_path.exists() {
        let mut file = File::create(&tmp_path)?;
        // SEAP v1 header:
        // magic "SEAP", version 1, counts=0, offsets=32
        let mut header = Vec::with_capacity(32);
        header.extend_from_slice(b"SEAP");
        header.extend_from_slice(&1u32.to_le_bytes()); // version 1
        header.extend_from_slice(&0u32.to_le_bytes()); // string_count
        header.extend_from_slice(&0u32.to_le_bytes()); // merged_count
        header.extend_from_slice(&0u32.to_le_bytes()); // common_count
        header.extend_from_slice(&32u32.to_le_bytes()); // string_offsets_pos
        header.extend_from_slice(&32u32.to_le_bytes()); // merged_pos
        header.extend_from_slice(&32u32.to_le_bytes()); // common_pos
        file.write_all(&header)?;
        file.flush()?;
    }
    Ok(tmp_path)
}

/// Returns the singleton G2PEngine.
fn get_engine() -> Option<&'static G2PEngine> {
    ENGINE
        .get_or_init(|| {
            let path = find_dict_path().or_else(|| create_fallback_dict().ok());
            if let Some(p) = path {
                if let Some(path_str) = p.to_str() {
                    return G2PEngine::new(path_str).ok();
                }
            }
            None
        })
        .as_ref()
}

/// Normalizes punctuation spacing for Vietnamese TTS.
/// Ensures punctuation marks (! ? , . ; :) have preceding spaces
/// while keeping words and existing spacing well-formed.
pub fn normalize_punctuation(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(input.len() + 16);
    let mut chars = input.chars().peekable();
    let mut prev_char: Option<char> = None;

    while let Some(c) = chars.next() {
        if matches!(c, '!' | '?' | ',' | '.' | ';' | ':') {
            if let Some(prev) = prev_char {
                if !prev.is_whitespace() {
                    result.push(' ');
                }
            }
            result.push(c);
            if let Some(&next) = chars.peek() {
                if !next.is_whitespace() && !matches!(next, '!' | '?' | ',' | '.' | ';' | ':') {
                    result.push(' ');
                }
            }
        } else {
            result.push(c);
        }
        prev_char = Some(c);
    }
    result
}

/// C-ABI: Normalizes punctuation of the input string.
/// Returns an owned NUL-terminated C string, or NULL on error.
/// Caller must free the returned string via `sea_g2p_free_string`.
#[no_mangle]
pub extern "C" fn sea_g2p_normalize_punc(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    if c_str.is_empty() {
        return match CString::new("") {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        };
    }

    let normalized = normalize_punctuation(c_str);
    match CString::new(normalized) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// C-ABI: Phonemizes Vietnamese text into phonetic representation.
/// Returns an owned NUL-terminated C string, or NULL on error.
/// Caller must free the returned string via `sea_g2p_free_string`.
#[no_mangle]
pub extern "C" fn sea_g2p_phonemize(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    if c_str.is_empty() {
        return match CString::new("") {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        };
    }

    let normalizer = get_normalizer();
    let normalized = normalizer.normalize(c_str, false);
    let target = if normalized.is_empty() {
        c_str
    } else {
        &normalized
    };

    let phonemes = if let Some(engine) = get_engine() {
        engine.phonemize(target)
    } else {
        target.to_string()
    };

    match CString::new(phonemes) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// C-ABI: Frees a string allocated and returned by sea_g2p functions.
/// Passing NULL is a safe no-op.
#[no_mangle]
pub extern "C" fn sea_g2p_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

/// C-ABI: Sets an explicit dictionary file path.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn sea_g2p_set_dict_path(path: *const c_char) -> i32 {
    if path.is_null() {
        return -1;
    }
    if let Ok(path_str) = unsafe { CStr::from_ptr(path) }.to_str() {
        set_dict_path(path_str);
        0
    } else {
        -1
    }
}
