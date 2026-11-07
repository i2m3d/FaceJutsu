//! What the page asks of the server. web/app/workspace.js is the other side of this.

use crate::{
    downloads,
    names::{file_name, is_key, unique},
    workspace::{FACES, PART},
    App,
};
use axum::{
    body::{Body, Bytes},
    extract::{ws::WebSocketUpgrade, DefaultBodyLimit, Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{io, path::PathBuf, sync::Arc};
use tokio::{
    fs,
    io::{AsyncSeekExt, AsyncWriteExt},
};

type Shared = State<Arc<App>>;
type Answer = Result<Response, StatusCode>;

/// What the models folder should hold, when it has no manifest of its own.
const MANIFEST: &str = include_str!("../manifest.json");

pub fn routes() -> Router<Arc<App>> {
    Router::new()
        .route("/status", get(status))
        .route("/alive", get(alive))
        .route("/models", get(models))
        .route("/models/urls", get(downloads::model_urls))
        .route("/models/{name}/download", post(downloads::download_model))
        .route("/open/{what}", post(open_folder))
        .route("/work", get(work_summary).delete(work_clear))
        .route("/work/{store}", get(work_all))
        .route("/work/{store}/{key}", get(work_get).put(work_put).delete(work_remove))
        .route("/work/faces/{key}", get(faces_get).put(faces_put))
        .route("/cache", get(cache_usage).delete(cache_clear))
        .route("/cache/{target}", get(cache_list).post(cache_plan))
        .route("/cache/{target}/{key}", get(cache_get).put(cache_put))
        .route("/outputs", post(output_start))
        .route("/outputs/{id}", put(output_write))
        .route("/outputs/{id}/close", post(output_close))
        .route("/outputs/{id}/abort", delete(output_abort))
        // (photos and videos are as large as they are; this is the user's own machine)
        .layer(DefaultBodyLimit::disable())
}

fn failed(e: io::Error) -> StatusCode {
    eprintln!("{e}");
    match e.kind() {
        io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

// ------------------------------------------------------------------ the app

async fn status(State(app): Shared) -> Json<Value> {
    Json(json!({
        "app": "facejutsu",
        "version": env!("CARGO_PKG_VERSION"),
        "build": env!("FACEJUTSU_BUILD"),
        "models": app.models,
        "outputs": app.workspace.outputs,
        "work": app.workspace.work,
        "cache": app.workspace.cache,
    }))
}

/// A socket the page keeps open and nothing is said on: it closes the moment this server stops.
/// (A socket, as a long request would hold one of the few connections a browser allows per host.)
async fn alive(State(app): Shared, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|mut socket| async move {
        app.page_connected();
        while let Some(Ok(_)) = socket.recv().await {}
        app.page_gone();
    })
}

pub async fn manifest_text(app: &App) -> String {
    fs::read_to_string(app.models.join("manifest.json")).await.unwrap_or_else(|_| MANIFEST.to_string())
}

pub async fn manifest(State(app): Shared) -> Response {
    ([(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-cache")], manifest_text(&app).await).into_response()
}

/// The model files there are.
async fn models(State(app): Shared) -> Result<Json<Value>, StatusCode> {
    let mut entries = fs::read_dir(&app.models).await.map_err(failed)?;
    let mut files = vec![];
    while let Some(e) = entries.next_entry().await.map_err(failed)? {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".safetensors") {
            let size = e.metadata().await.map_err(failed)?.len();
            files.push(json!({ "name": name, "size": size }));
        }
    }
    Ok(Json(json!(files)))
}

/// Shows a folder in the system's file manager. Only these: the page doesn't get to name a path.
async fn open_folder(State(app): Shared, Path(what): Path<String>) -> Answer {
    let dir = match what.as_str() {
        "models" => &app.models,
        "outputs" => &app.workspace.outputs,
        _ => return Err(StatusCode::NOT_FOUND),
    };
    open::that_detached(dir).map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ------------------------------------------------------------------ saved work

fn work_dir(app: &App, store: &str) -> Result<PathBuf, StatusCode> {
    match store {
        "characters" | "targets" => Ok(app.workspace.work.join(store)),
        _ => Err(StatusCode::NOT_FOUND),
    }
}

fn work_file(app: &App, store: &str, key: &str) -> Result<PathBuf, StatusCode> {
    if !is_key(key) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(work_dir(app, store)?.join(format!("{key}.json")))
}

fn faces_file(app: &App, key: &str) -> Result<PathBuf, StatusCode> {
    if !is_key(key) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(app.workspace.faces.join(format!("{key}{FACES}")))
}

/// The files of `dir` that end in `ending`.
async fn files_ending(dir: &std::path::Path, ending: &str) -> io::Result<Vec<PathBuf>> {
    let mut out = vec![];
    let mut entries = fs::read_dir(dir).await?;
    while let Some(e) = entries.next_entry().await? {
        if e.file_name().to_string_lossy().ends_with(ending) {
            out.push(e.path());
        }
    }
    Ok(out)
}

/// The records' file names in `dir`.
async fn records(dir: &std::path::Path) -> io::Result<Vec<PathBuf>> {
    files_ending(dir, ".json").await
}

const JSON: [(header::HeaderName, &str); 2] = [(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-store")];

async fn work_summary(State(app): Shared) -> Result<Json<Value>, StatusCode> {
    let count = |store: &'static str| {
        let app = app.clone();
        async move { Ok::<usize, io::Error>(records(&work_dir(&app, store).map_err(|_| io::ErrorKind::NotFound)?).await?.len()) }
    };
    Ok(Json(json!({
        "characters": count("characters").await.map_err(failed)?,
        "targets": count("targets").await.map_err(failed)?,
    })))
}

async fn work_clear(State(app): Shared) -> Answer {
    for store in ["characters", "targets"] {
        let dir = work_dir(&app, store)?;
        for path in records(&dir).await.map_err(failed)? {
            fs::remove_file(path).await.map_err(failed)?;
        }
    }
    for path in files_ending(&app.workspace.faces, FACES).await.map_err(failed)? {
        fs::remove_file(path).await.map_err(failed)?;
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Every record of a store, as a JSON array.
async fn work_all(State(app): Shared, Path(store): Path<String>) -> Answer {
    let dir = work_dir(&app, &store)?;
    let mut body = String::from("[");
    for (i, path) in records(&dir).await.map_err(failed)?.into_iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&fs::read_to_string(path).await.map_err(failed)?);
    }
    body.push(']');
    Ok((JSON, body).into_response())
}

async fn work_get(State(app): Shared, Path((store, key)): Path<(String, String)>) -> Answer {
    // (no such record is an ordinary answer: most things opened are new. A 404 would be logged as
    // an error in the browser's console each time, so these say 204.)
    match fs::read(work_file(&app, &store, &key)?).await {
        Ok(body) => Ok((JSON, body).into_response()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(StatusCode::NO_CONTENT.into_response()),
        Err(e) => Err(failed(e)),
    }
}

async fn work_put(State(app): Shared, Path((store, key)): Path<(String, String)>, body: Bytes) -> Answer {
    serde_json::from_slice::<serde::de::IgnoredAny>(&body).map_err(|_| StatusCode::BAD_REQUEST)?;
    app.workspace.write_atomic(&work_file(&app, &store, &key)?, &body).await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn work_remove(State(app): Shared, Path((store, key)): Path<(String, String)>) -> Answer {
    match fs::remove_file(work_file(&app, &store, &key)?).await {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(failed(e)),
        _ => Ok(StatusCode::NO_CONTENT.into_response()),
    }
}

/// The faces found in a photo or video: bytes the page makes and reads (saved.js).
async fn faces_get(State(app): Shared, Path(key): Path<String>) -> Answer {
    match fs::read(faces_file(&app, &key)?).await {
        Ok(data) => Ok(([(header::CONTENT_TYPE, "application/octet-stream"), (header::CACHE_CONTROL, "no-store")], data).into_response()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(StatusCode::NO_CONTENT.into_response()),
        Err(e) => Err(failed(e)),
    }
}

async fn faces_put(State(app): Shared, Path(key): Path<String>, body: Bytes) -> Answer {
    app.workspace.write_atomic(&faces_file(&app, &key)?, &body).await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ------------------------------------------------------------------ swapped frames

async fn cache_usage(State(app): Shared) -> Json<Value> {
    use std::sync::atomic::Ordering::Relaxed;
    Json(json!({ "bytes": app.workspace.cache_bytes.load(Relaxed), "count": app.workspace.cache_count.load(Relaxed) }))
}

async fn cache_clear(State(app): Shared) -> Answer {
    app.workspace.cache_clear().await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// The patches kept of a photo or video: their keys, as a JSON array.
async fn cache_list(State(app): Shared, Path(target): Path<String>) -> Answer {
    if !is_key(&target) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let keys = app.workspace.cache_keys(&target).await.map_err(failed)?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(keys)).into_response())
}

/// What a video's frames are being swapped with now (see Workspace::cache_plan).
async fn cache_plan(State(app): Shared, Path(target): Path<String>, body: Bytes) -> Answer {
    if !is_key(&target) || !std::str::from_utf8(&body).is_ok_and(is_key) {
        return Err(StatusCode::BAD_REQUEST);
    }
    app.workspace.cache_plan(&target, &body).await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

fn patch_file(app: &App, target: &str, key: &str) -> Result<PathBuf, StatusCode> {
    if !is_key(target) || !is_key(key) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(app.workspace.cache.join(target).join(key))
}

async fn cache_get(State(app): Shared, Path((target, key)): Path<(String, String)>) -> Answer {
    match fs::read(patch_file(&app, &target, &key)?).await {
        Ok(data) => Ok(([(header::CONTENT_TYPE, "application/octet-stream"), (header::CACHE_CONTROL, "no-store")], data).into_response()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(StatusCode::NO_CONTENT.into_response()),
        Err(e) => Err(failed(e)),
    }
}

async fn cache_put(State(app): Shared, Path((target, key)): Path<(String, String)>, body: Bytes) -> Answer {
    patch_file(&app, &target, &key)?;
    app.workspace.cache_put(&target, &key, &body).await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ------------------------------------------------------------------ files

/// Writes a request's body to `file` as it comes.
async fn stream_to(file: &mut fs::File, body: Body) -> io::Result<()> {
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        file.write_all(&chunk.map_err(io::Error::other)?).await?;
    }
    file.flush().await
}

#[derive(Deserialize)]
struct NewOutput {
    name: String,
}

/// An output being written is only its part file, so one goes on across a restart of this server.
fn output_part(app: &App, id: &str) -> Result<PathBuf, StatusCode> {
    let part = app.workspace.outputs.join(format!("{id}{PART}"));
    if !is_key(id) || !part.is_file() {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(part)
}

/// Starts an output. It's a part file until it's closed, so a failed or stopped export leaves nothing.
async fn output_start(State(app): Shared, Json(q): Json<NewOutput>) -> Answer {
    file_name(&q.name).ok_or(StatusCode::BAD_REQUEST)?;
    let id = app.workspace.next_id();
    fs::File::create(app.workspace.outputs.join(format!("{id}{PART}"))).await.map_err(failed)?;
    Ok(Json(json!({ "id": id })).into_response())
}

#[derive(Deserialize)]
struct At {
    #[serde(default)]
    at: u64,
}

/// Writes at a position: a video's file is written out of order (its header is filled in last).
async fn output_write(State(app): Shared, Path(id): Path<String>, Query(q): Query<At>, body: Body) -> Answer {
    let part = output_part(&app, &id)?;
    let mut file = fs::OpenOptions::new().write(true).open(&part).await.map_err(failed)?;
    file.seek(io::SeekFrom::Start(q.at)).await.map_err(failed)?;
    stream_to(&mut file, body).await.map_err(failed)?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn output_close(State(app): Shared, Path(id): Path<String>, Json(q): Json<NewOutput>) -> Answer {
    let part = output_part(&app, &id)?;
    let name = file_name(&q.name).ok_or(StatusCode::BAD_REQUEST)?;
    let dir = &app.workspace.outputs;
    let free = unique(&name, |n| dir.join(n).exists());
    let path = dir.join(&free);
    fs::rename(&part, &path).await.map_err(failed)?;
    Ok(Json(json!({ "name": free, "path": path })).into_response())
}

async fn output_abort(State(app): Shared, Path(id): Path<String>) -> Answer {
    let part = output_part(&app, &id)?;
    let _ = fs::remove_file(part).await;
    Ok(StatusCode::NO_CONTENT.into_response())
}
