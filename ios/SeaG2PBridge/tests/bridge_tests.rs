// ios/SeaG2PBridge/tests/bridge_tests.rs
use sea_g2p_bridge::*;
use std::ffi::{CStr, CString};

#[test]
fn test_c_abi_normalization_and_phonemization() {
    let input = CString::new("Xin chào Việt Nam!").unwrap();
    let norm_ptr = sea_g2p_normalize_punc(input.as_ptr());
    assert!(!norm_ptr.is_null());
    let norm_str = unsafe { CStr::from_ptr(norm_ptr).to_str().unwrap() };
    assert!(norm_str.contains("Xin chào Việt Nam !"));
    sea_g2p_free_string(norm_ptr);

    let phone_ptr = sea_g2p_phonemize(input.as_ptr());
    assert!(!phone_ptr.is_null());
    let phone_str = unsafe { CStr::from_ptr(phone_ptr).to_str().unwrap() };
    assert!(!phone_str.is_empty());
    sea_g2p_free_string(phone_ptr);
}

#[test]
fn test_c_abi_empty_string() {
    let input = CString::new("").unwrap();
    let norm_ptr = sea_g2p_normalize_punc(input.as_ptr());
    assert!(!norm_ptr.is_null());
    let norm_str = unsafe { CStr::from_ptr(norm_ptr).to_str().unwrap() };
    assert_eq!(norm_str, "");
    sea_g2p_free_string(norm_ptr);

    let phone_ptr = sea_g2p_phonemize(input.as_ptr());
    assert!(!phone_ptr.is_null());
    let phone_str = unsafe { CStr::from_ptr(phone_ptr).to_str().unwrap() };
    assert_eq!(phone_str, "");
    sea_g2p_free_string(phone_ptr);
}

#[test]
fn test_c_abi_null_handling() {
    let norm_ptr = sea_g2p_normalize_punc(std::ptr::null());
    assert!(norm_ptr.is_null());

    let phone_ptr = sea_g2p_phonemize(std::ptr::null());
    assert!(phone_ptr.is_null());

    // Should not crash on null free
    sea_g2p_free_string(std::ptr::null_mut());
}

#[test]
fn test_c_abi_punctuation_spacing() {
    let input = CString::new("Hà Nội, thủ đô của Việt Nam.").unwrap();
    let norm_ptr = sea_g2p_normalize_punc(input.as_ptr());
    assert!(!norm_ptr.is_null());
    let norm_str = unsafe { CStr::from_ptr(norm_ptr).to_str().unwrap() };
    assert!(norm_str.contains(","));
    assert!(norm_str.contains("."));
    sea_g2p_free_string(norm_ptr);
}
