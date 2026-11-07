//! FaceJutsu's local server: the page (embedded in this executable), the models folder and the
//! workspace, on localhost only. All the work happens in the browser; this only keeps the files.

// (the executable is named FaceJutsu, and the crate after it)
#![allow(non_snake_case)]

mod api;
mod config;
mod downloads;
mod names;
mod workspace;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
use config::Config;
use rust_embed::RustEmbed;
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::net::TcpListener;
use tower_http::services::ServeDir;
use workspace::Workspace;

#[derive(RustEmbed)]
#[folder = "../web/"]
struct Page;

pub struct App {
    pub port: u16,
    pub home: PathBuf,
    pub models: PathBuf,
    pub workspace: Workspace,
    /// The pages connected now.
    pages: AtomicUsize,
    /// A file that's there while a page is connected. Left behind when this server stops first, so
    /// the next start knows a page may still be open.
    page_marker: PathBuf,
}

impl App {
    pub fn page_connected(&self) {
        if self.pages.fetch_add(1, Ordering::Relaxed) == 0 {
            let _ = std::fs::write(&self.page_marker, "");
        }
    }

    pub fn page_gone(&self) {
        if self.pages.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _ = std::fs::remove_file(&self.page_marker);
        }
    }
}

/// How long a start waits for a page open from before it to come back: it tries every second, and a
/// try at a stopped server can itself take a couple of seconds to fail.
const PAGE_RETURNS: Duration = Duration::from_secs(5);

/// Whether a page open from before this start (`was_open`) has come back, within `wait`.
async fn page_returned(pages: &AtomicUsize, was_open: bool, wait: Duration) -> bool {
    if !was_open {
        return false;
    }
    let start = Instant::now();
    while start.elapsed() < wait {
        if pages.load(Ordering::Relaxed) > 0 {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

struct Args {
    home: Option<PathBuf>,
    port: Option<u16>,
    dev: bool,
    no_browser: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args { home: None, port: None, dev: false, no_browser: false };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--home" => args.home = Some(it.next().ok_or("--home needs a folder")?.into()),
            "--port" => args.port = Some(it.next().and_then(|p| p.parse().ok()).ok_or("--port needs a number")?),
            "--dev" => args.dev = true,
            "--no-browser" => args.no_browser = true,
            "--version" => {
                println!("FaceJutsu {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                println!(
                    "FaceJutsu [--home <folder>] [--port <port>] [--no-browser] [--dev]\n\n\
                     --home        where {} is, and what its relative paths start from\n                  \
                     (default: this executable's folder; with --dev, the current one)\n\
                     --port        use this port instead of the settings file's\n\
                     --no-browser  don't open the browser\n\
                     --dev         serve web/ from disk, and tools/ and out/ beside it",
                    config::FILE_NAME
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other} (see --help)")),
        }
    }
    Ok(args)
}

fn home_folder(args: &Args) -> Result<PathBuf, String> {
    if let Some(h) = &args.home {
        return Ok(h.clone());
    }
    if args.dev {
        return std::env::current_dir().map_err(|e| e.to_string());
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    exe.parent().map(Path::to_path_buf).ok_or_else(|| "the executable has no folder".into())
}

/// Whether FaceJutsu is what's answering on `port`.
fn already_running(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_millis(1000)));
    let ask = format!("GET /api/status HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(ask.as_bytes()).is_err() {
        return false;
    }
    let mut reply = String::new();
    let _ = s.read_to_string(&mut reply);
    reply.contains("\"app\":\"facejutsu\"")
}

enum Bound {
    Listener(TcpListener, u16),
    Running(u16),
}

/// The port asked for, or the next free one. An earlier start of this app on one of them is used
/// instead of starting another.
async fn bind(port: u16) -> Result<Bound, String> {
    for p in port..=port.saturating_add(20) {
        match TcpListener::bind((Ipv4Addr::LOCALHOST, p)).await {
            Ok(l) => return Ok(Bound::Listener(l, p)),
            Err(_) if already_running(p) => return Ok(Bound::Running(p)),
            Err(_) => {}
        }
    }
    Err(format!("ports {port} to {} are all taken. Set another port in {}.", port.saturating_add(20), config::FILE_NAME))
}

/// Only this app's own page may talk to this server. Without this, any website open in the browser
/// could send requests to localhost (through its Origin) or read from it (with DNS rebinding,
/// through a Host of its own).
async fn only_this_app(State(app): State<Arc<App>>, req: Request, next: Next) -> Response {
    let ours = |host: &str| {
        let port = app.port;
        host == format!("localhost:{port}") || host == format!("127.0.0.1:{port}") || host == format!("[::1]:{port}")
    };
    let host = req.headers().get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    let origin_ok = match req.headers().get(header::ORIGIN) {
        None => true,
        Some(o) => o.to_str().ok().and_then(|o| o.strip_prefix("http://")).is_some_and(&ours),
    };
    if !ours(host) || !origin_ok {
        return StatusCode::FORBIDDEN.into_response();
    }
    next.run(req).await
}

async fn embedded(method: Method, uri: Uri) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let Some(file) = Page::get(path) else { return StatusCode::NOT_FOUND.into_response() };
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_str(mime.as_ref()).unwrap_or(HeaderValue::from_static("application/octet-stream")));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    (headers, Body::from(file.data.into_owned())).into_response()
}

fn router(app: Arc<App>, dev: bool) -> Router {
    let mut routes = Router::new()
        .nest("/api", api::routes())
        .route("/models/manifest.json", axum::routing::get(api::manifest))
        .nest_service("/models", ServeDir::new(&app.models));
    routes = if dev {
        routes
            .nest_service("/tools", ServeDir::new(app.home.join("tools")))
            .nest_service("/out", ServeDir::new(app.home.join("out")))
            .fallback_service(ServeDir::new(app.home.join("web")))
    } else {
        routes.fallback(embedded)
    };
    routes.layer(middleware::from_fn_with_state(app.clone(), only_this_app)).with_state(app)
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let home = home_folder(&args)?;
    std::fs::create_dir_all(&home).map_err(|e| format!("couldn't make {}: {e}", home.display()))?;
    let config = Config::load(&home)?;
    let open_browser = config.open_browser && !args.no_browser;
    let (listener, port) = match bind(args.port.unwrap_or(config.port)).await? {
        Bound::Listener(l, p) => (l, p),
        Bound::Running(p) => {
            let url = format!("http://localhost:{p}/");
            println!("FaceJutsu is already running at {url}");
            if open_browser {
                let _ = open::that(&url);
            }
            return Ok(());
        }
    };
    let workspace = Workspace::open(config.outputs.clone(), config.work.clone(), config.cache.clone(), config.cache_gb << 30)
        .await
        .map_err(|e| format!("couldn't use the folders {}, {} and {}: {e}", config.outputs.display(), config.work.display(), config.cache.display()))?;
    std::fs::create_dir_all(&config.models).map_err(|e| format!("couldn't make {}: {e}", config.models.display()))?;
    downloads::clear_parts(&config.models)?;
    let url = format!("http://localhost:{port}/");
    println!("FaceJutsu is running at {url}\n");
    println!("  Models:    {}", config.models.display());
    println!("  Outputs:   {}", config.outputs.display());
    println!("  Settings:  {}\n", home.join(config::FILE_NAME).display());
    println!("Close this window (or press Ctrl+C) to stop it.");
    let page_marker = config.work.join("page-open");
    let was_open = page_marker.exists();
    let app = Arc::new(App { port, home, models: config.models, workspace, pages: AtomicUsize::new(0), page_marker });
    if open_browser {
        let app = app.clone();
        tokio::spawn(async move {
            // a page still open from before this start comes back by itself, with the work in it:
            // no second page in front of that
            if page_returned(&app.pages, was_open, PAGE_RETURNS).await {
                println!("The page you had open has reconnected.");
                return;
            }
            if let Err(e) = open::that(&url) {
                println!("Couldn't open the browser ({e}): open {url} yourself.");
            }
        });
    }
    axum::serve(listener, router(app, args.dev)).await.map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("FaceJutsu couldn't start: {e}");
        // (started by a double-click, the window would close before this could be read)
        eprintln!("\nPress Enter to close.");
        let _ = std::io::stdin().read_line(&mut String::new());
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: Duration = Duration::from_secs(60);

    #[tokio::test]
    async fn with_no_page_open_before_a_start_doesnt_wait() {
        let start = Instant::now();
        assert!(!page_returned(&AtomicUsize::new(0), false, LONG).await);
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn a_page_that_comes_back_ends_the_wait() {
        let pages = Arc::new(AtomicUsize::new(0));
        let later = pages.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            later.store(1, Ordering::Relaxed);
        });
        let start = Instant::now();
        assert!(page_returned(&pages, true, LONG).await);
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn a_page_that_doesnt_come_back_is_given_up_on() {
        assert!(!page_returned(&AtomicUsize::new(0), true, Duration::from_millis(150)).await);
    }
}
