//! Getting model files into the models folder, from the addresses in the settings file. The page
//! only names a file: which address it comes from is that file's to say.

use crate::{config, workspace::PART, App};
use axum::{
    body::{Body, Bytes},
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::{convert::Infallible, path::Path as FilePath, sync::Arc, time::Duration};
use tokio::{fs, io::AsyncWriteExt, sync::mpsc};

/// Removes what downloads stopped by closing the app left in the models folder.
pub fn clear_parts(models: &FilePath) -> Result<(), String> {
    for entry in std::fs::read_dir(models).map_err(|e| format!("couldn't read {}: {e}", models.display()))?.flatten() {
        if entry.file_name().to_string_lossy().ends_with(PART) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    Ok(())
}

/// -> { urls: { file name: address }, problem: what's wrong with the file, if anything }
pub async fn model_urls(State(app): State<Arc<App>>) -> Json<Value> {
    Json(match config::model_urls(&app.home) {
        Ok(urls) => json!({ "urls": urls }),
        Err(problem) => json!({ "urls": {}, "problem": problem }),
    })
}

/// The size the manifest gives a file, if it knows it.
fn size_in_manifest(manifest: &str, name: &str) -> Option<u64> {
    let manifest: Value = serde_json::from_str(manifest).ok()?;
    manifest["components"].as_object()?.values().find_map(|c| {
        c["builds"].as_object()?.values().find(|b| b["path"] == name)?["size"].as_u64()
    })
}

/// Fetches `url` into the models folder as `name`. It's a part file until it's whole, so a file by
/// that name is always a complete one. `tx` hears how far it's got, and closes if the page has gone.
async fn fetch(app: &App, url: &str, name: &str, size: Option<u64>, tx: &mpsc::Sender<Bytes>) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("FaceJutsu/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let res = client.get(url).send().await.map_err(|e| format!("couldn't reach the address ({e})"))?;
    if !res.status().is_success() {
        return Err(format!("the address answered {}", res.status()));
    }
    let total = size.or(res.content_length());
    let part = app.models.join(format!("{}{PART}", app.workspace.next_id()));
    let mut file = fs::File::create(&part).await.map_err(|e| e.to_string())?;
    let written = async {
        let mut stream = res.bytes_stream();
        let mut done = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("the download broke off ({e})"))?;
            file.write_all(&chunk).await.map_err(|e| e.to_string())?;
            done += chunk.len() as u64;
            if tx.is_closed() {
                return Err("stopped".to_string());
            }
            // (a page that's behind misses some: it only shows the latest)
            let _ = tx.try_send(line(json!({ "done": done, "total": total })));
        }
        file.flush().await.map_err(|e| e.to_string())?;
        match size {
            Some(want) if want != done => Err(format!("got {done} bytes, not the {want} expected: the address may hold another build of it")),
            _ => Ok(()),
        }
    }
    .await;
    drop(file);
    let finished = match written {
        Ok(()) => fs::rename(&part, app.models.join(name)).await.map_err(|e| e.to_string()),
        Err(e) => Err(e),
    };
    if finished.is_err() {
        let _ = fs::remove_file(&part).await;
    }
    finished
}

fn line(value: Value) -> Bytes {
    Bytes::from(format!("{value}\n"))
}

/// Downloads a model file. The answer is lines of JSON as it goes: { done, total } (total may be
/// null), and last { ok: true } or { error }. Leaving stops it.
pub async fn download_model(State(app): State<Arc<App>>, Path(name): Path<String>) -> Result<Response, StatusCode> {
    let urls = config::model_urls(&app.home).map_err(|e| {
        eprintln!("{e}");
        StatusCode::CONFLICT
    })?;
    let url = urls.get(&name).ok_or(StatusCode::NOT_FOUND)?.clone();
    let size = size_in_manifest(&crate::api::manifest_text(&app).await, &name);
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(async move {
        let last = match fetch(&app, &url, &name, size, &tx).await {
            Ok(()) => json!({ "ok": true }),
            Err(error) => {
                eprintln!("{name}: {error}");
                json!({ "error": error })
            }
        };
        let _ = tx.send(line(last)).await;
    });
    let body = futures_util::stream::unfold(rx, |mut rx| async { rx.recv().await.map(|b| (Ok::<_, Infallible>(b), rx)) });
    Ok(([(header::CONTENT_TYPE, "application/x-ndjson"), (header::CACHE_CONTROL, "no-store")], Body::from_stream(body)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_come_from_the_manifest() {
        let manifest = include_str!("../manifest.json");
        assert_eq!(size_in_manifest(manifest, "gpen-bfr-512.int8.safetensors"), Some(81716100));
        assert_eq!(size_in_manifest(manifest, "other.safetensors"), None);
    }
}
