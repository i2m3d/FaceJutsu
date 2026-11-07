//! Gives the Windows executable its icon and the name the system shows for it.

use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=assets/FaceJutsu.ico");
    // tells this build from others, for a page left open across an update (api.rs status)
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../web");
    let build = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |t| t.as_millis());
    println!("cargo:rustc-env=FACEJUTSU_BUILD={build}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winresource::WindowsResource::new()
            .set_icon("assets/FaceJutsu.ico")
            .set("ProductName", "FaceJutsu")
            .set("FileDescription", "FaceJutsu")
            .compile()
            .expect("couldn't embed the icon");
    }
}
