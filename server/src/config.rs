//! The settings file, next to the executable. Written with the defaults the first time.

use crate::names::file_name;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub const FILE_NAME: &str = "config.toml";

// Below the ephemeral ranges the systems hand out (Windows from 49152, Linux from 32768), and clear
// of the ports dev servers and other local apps usually take.
const DEFAULT_PORT: u16 = 24817;

const DEFAULT_FILE: &str = include_str!("../../packaging/config.toml");

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub port: u16,
    pub open_browser: bool,
    pub models: PathBuf,
    pub outputs: PathBuf,
    pub work: PathBuf,
    pub cache: PathBuf,
    pub cache_gb: u64,
    model_urls: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self { port: DEFAULT_PORT, open_browser: true, models: "models".into(), outputs: "outputs".into(), work: "data/work".into(), cache: "data/cache".into(), cache_gb: 4, model_urls: BTreeMap::new() }
    }
}

impl Config {
    /// Reads `home`'s settings file, writing the defaults there if it has none.
    pub fn load(home: &Path) -> Result<Self, String> {
        let path = home.join(FILE_NAME);
        let mut config: Config = match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).map_err(|e| format!("{} has a mistake in it:\n{e}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::fs::write(&path, DEFAULT_FILE).map_err(|e| format!("couldn't write {}: {e}", path.display()))?;
                toml::from_str(DEFAULT_FILE).map_err(|e| format!("the default settings have a mistake in them: {e}"))?
            }
            Err(e) => return Err(format!("couldn't read {}: {e}", path.display())),
        };
        if config.port == 0 {
            return Err(format!("{}: the port can't be 0", path.display()));
        }
        // (components: "data/work" with this system's separators, as it's shown to the user)
        let here = |p: &Path| home.join(p).components().collect::<PathBuf>();
        config.models = here(&config.models);
        config.outputs = here(&config.outputs);
        config.work = here(&config.work);
        config.cache = here(&config.cache);
        Ok(config)
    }
}

/// The model files that have an address: file name -> address. Only model files, by a plain name,
/// from the web.
fn parse_urls(text: &str) -> Result<BTreeMap<String, String>, toml::de::Error> {
    let mut urls = toml::from_str::<Config>(text)?.model_urls;
    urls.retain(|name, url| {
        file_name(name).as_deref() == Some(name) && name.ends_with(".safetensors") && (url.starts_with("https://") || url.starts_with("http://"))
    });
    Ok(urls)
}

/// `home`'s model addresses as the file says now: they can change while the app runs.
pub fn model_urls(home: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = home.join(FILE_NAME);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("couldn't read {}: {e}", path.display()))?;
    parse_urls(&text).map_err(|e| format!("{} has a mistake in it: {}", path.display(), e.message()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_written_says_the_same_as_the_defaults() {
        let written: Config = toml::from_str(DEFAULT_FILE).unwrap();
        let default = Config::default();
        assert_eq!(written.port, default.port);
        assert_eq!(written.open_browser, default.open_browser);
        assert_eq!(written.models, default.models);
        assert_eq!(written.outputs, default.outputs);
        assert_eq!(written.work, default.work);
        assert_eq!(written.cache, default.cache);
        assert_eq!(written.cache_gb, default.cache_gb);
    }

    #[test]
    fn a_misspelled_setting_is_an_error() {
        assert!(toml::from_str::<Config>("prot = 1").is_err());
    }

    #[test]
    fn the_file_written_names_every_model_in_the_manifest() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!("../manifest.json")).unwrap();
        let names = toml::from_str::<Config>(DEFAULT_FILE).unwrap().model_urls;
        let paths: Vec<_> = manifest["components"].as_object().unwrap().values().flat_map(|c| c["builds"].as_object().unwrap().values().map(|b| b["path"].as_str().unwrap())).collect();
        assert_eq!(names.len(), paths.len());
        for path in paths {
            assert!(names.contains_key(path), "{path}");
        }
    }

    #[test]
    fn only_model_files_from_the_web_are_taken() {
        let urls = parse_urls(
            r#"[model_urls]
            "a.safetensors" = "https://example.com/a"
            "b.safetensors" = "javascript:alert(1)"
            "../c.safetensors" = "https://example.com/c"
            "d.exe" = "https://example.com/d"
            "#,
        )
        .unwrap();
        assert_eq!(urls.keys().collect::<Vec<_>>(), ["a.safetensors"]);
    }
}
