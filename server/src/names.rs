//! Names from the page, made safe to use in a path. Nothing from a request may leave its folder.

/// A key for a saved record or a cached patch: letters, digits, `_` and `-` only.
pub fn is_key(key: &str) -> bool {
    !key.is_empty() && key.len() <= 128 && key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// A file's name as the user knows it, safe on every system: no folders, no characters a system
/// refuses, not one of Windows's reserved names. None when nothing is left of it.
pub fn file_name(name: &str) -> Option<String> {
    // (the last part only, whichever separator it has)
    let base = name.rsplit(['/', '\\']).next().unwrap_or("");
    let clean: String = base.chars().map(|c| if c.is_control() || "<>:\"|?*".contains(c) { '_' } else { c }).take(200).collect();
    let clean = clean.trim().trim_end_matches('.').trim_start_matches('.').to_string();
    if clean.is_empty() {
        return None;
    }
    let stem = clean.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) && stem.as_bytes()[3].is_ascii_digit());
    Some(if reserved { format!("_{clean}") } else { clean })
}

/// `name`, or `name (2).ext` and so on: the first that `taken` says is free.
pub fn unique(name: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(name) {
        return name.to_string();
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) if i > 0 => name.split_at(i),
        _ => (name, ""),
    };
    (2u32..).map(|n| format!("{stem} ({n}){ext}")).find(|n| !taken(n)).expect("a free name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_cant_leave_their_folder() {
        for bad in ["", "..", "../x", "a/b", "a\\b", "a:b", "a.b", "%2e%2e"] {
            assert!(!is_key(bad), "{bad}");
        }
        assert!(is_key("c1"));
        assert!(is_key("f00d-BEEF_1"));
        assert!(!is_key(&"a".repeat(129)));
    }

    #[test]
    fn file_names_lose_their_folders() {
        assert_eq!(file_name("../../etc/passwd").as_deref(), Some("passwd"));
        assert_eq!(file_name("..\\..\\boot.ini").as_deref(), Some("boot.ini"));
        assert_eq!(file_name("C:\\Windows\\x.exe").as_deref(), Some("x.exe"));
        assert_eq!(file_name("/abs/x.png").as_deref(), Some("x.png"));
        assert_eq!(file_name(".."), None);
        assert_eq!(file_name("a/"), None);
        assert_eq!(file_name("   "), None);
    }

    #[test]
    fn file_names_work_on_every_system() {
        assert_eq!(file_name("a:b?.png").as_deref(), Some("a_b_.png"));
        assert_eq!(file_name("clip.").as_deref(), Some("clip"));
        assert_eq!(file_name(".hidden").as_deref(), Some("hidden"));
        assert_eq!(file_name("CON").as_deref(), Some("_CON"));
        assert_eq!(file_name("nul.mp4").as_deref(), Some("_nul.mp4"));
        assert_eq!(file_name("com1.txt").as_deref(), Some("_com1.txt"));
        assert_eq!(file_name("console.txt").as_deref(), Some("console.txt"));
        assert_eq!(file_name("日本語 clip.mp4").as_deref(), Some("日本語 clip.mp4"));
    }

    #[test]
    fn unique_names_count_up() {
        assert_eq!(unique("a.mp4", |_| false), "a.mp4");
        assert_eq!(unique("a.mp4", |n| n == "a.mp4"), "a (2).mp4");
        assert_eq!(unique("a.mp4", |n| n == "a.mp4" || n == "a (2).mp4"), "a (3).mp4");
        assert_eq!(unique("noext", |n| n == "noext"), "noext (2)");
    }
}
