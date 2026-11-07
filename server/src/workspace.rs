//! The user's files, each folder where the settings say:
//!
//!   outputs/      the finished photos and videos
//!   data/work/    the saved work: characters/, targets/ (the edits made) and faces/ (the faces found)
//!   data/cache/   the swapped frames kept so they aren't swapped again, within a budget: a folder per
//!                 photo or video, with a file per frame and its plan (see cache_plan)

use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::fs;

const OLD_CACHE: &str = "cache-old-";
const PLAN: &str = "plan";
pub const PART: &str = ".part";
pub const FACES: &str = ".faces";
/// How long an output left unfinished is kept across starts: the page writing it may still be open,
/// waiting to go on with it.
const OUTPUT_PART_KEPT: Duration = Duration::from_secs(24 * 60 * 60);

pub struct Workspace {
    pub outputs: PathBuf,
    pub work: PathBuf,
    pub faces: PathBuf,
    pub cache: PathBuf,
    cache_budget: u64,
    pub cache_bytes: AtomicU64,
    pub cache_count: AtomicU64,
    /// The video that filled the cache with nothing left to drop for it: its new frames aren't kept.
    cache_stopped: Mutex<Option<String>>,
    serial: AtomicU64,
}

struct Frame {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

/// The folders of the cache.
fn cache_folders(cache: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(cache) else { return vec![] };
    entries.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).collect()
}

/// The swapped frames in a folder of the cache.
fn frames_in(folder: &Path) -> Vec<Frame> {
    let Ok(entries) = std::fs::read_dir(folder) else { return vec![] };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name != PLAN && !name.ends_with(PART)
        })
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some(Frame { path: e.path(), size: m.len(), modified: m.modified().ok()? })
        })
        .collect()
}

/// Size and count of the swapped frames.
fn measure(cache: &Path) -> (u64, u64) {
    cache_folders(cache).iter().flat_map(|f| frames_in(f)).fold((0, 0), |(b, n), f| (b + f.size, n + 1))
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Removes `frames` in turn, until `bytes` is down to `goal`.
fn drop_frames(frames: Vec<Frame>, goal: u64, bytes: &mut u64, count: &mut u64) {
    for f in frames {
        if *bytes <= goal {
            break;
        }
        if std::fs::remove_file(&f.path).is_ok() {
            *bytes -= f.size;
            *count -= 1;
        }
    }
}

/// Makes room for the frames of `target`, down to `goal` bytes if it can: first the other videos',
/// the one last added to longest ago first, then those of `target` from before its plan last
/// changed. The rest of `target` stays. Those of `target` go in the order of their names, hashes, so
/// in no order of the video: going oldest first would drop the very frames a second export in order
/// comes to next, and it would find none of them. -> the (bytes, count) left
fn make_room(cache: &Path, target: &str, goal: u64) -> (u64, u64) {
    let (mut bytes, mut count) = measure(cache);
    let mine = cache.join(target);
    let mut others: Vec<PathBuf> = cache_folders(cache).into_iter().filter(|f| *f != mine).collect();
    // (by their frames' dates: dropping a frame changes the folder's own)
    others.sort_by_cached_key(|f| frames_in(f).iter().map(|x| x.modified).max());
    for folder in others {
        if bytes <= goal {
            break;
        }
        drop_frames(frames_in(&folder), goal, &mut bytes, &mut count);
        if frames_in(&folder).is_empty() {
            let _ = std::fs::remove_dir_all(&folder);
        }
    }
    if let Some(changed) = modified(&mine.join(PLAN)) {
        let mut stale: Vec<Frame> = frames_in(&mine).into_iter().filter(|f| f.modified < changed).collect();
        stale.sort_by(|a, b| a.path.cmp(&b.path));
        drop_frames(stale, goal, &mut bytes, &mut count);
    }
    (bytes, count)
}

impl Workspace {
    /// `cache_budget`: the bytes of swapped frames to keep at most.
    pub async fn open(outputs: PathBuf, work: PathBuf, cache: PathBuf, cache_budget: u64) -> io::Result<Self> {
        let w = Self {
            outputs,
            faces: work.join("faces"),
            work,
            cache,
            cache_budget,
            cache_bytes: AtomicU64::new(0),
            cache_count: AtomicU64::new(0),
            cache_stopped: Mutex::new(None),
            serial: AtomicU64::new(1),
        };
        for dir in [&w.outputs, &w.cache, &w.faces, &w.work.join("characters"), &w.work.join("targets")] {
            fs::create_dir_all(dir).await?;
        }
        // what a stop in the middle of something left behind
        let mut dirs = vec![w.outputs.clone(), w.faces.clone(), w.work.join("characters"), w.work.join("targets")];
        dirs.extend(cache_folders(&w.cache));
        for dir in dirs {
            let keep = if dir == w.outputs { OUTPUT_PART_KEPT } else { Duration::ZERO };
            let mut entries = fs::read_dir(dir).await?;
            while let Some(e) = entries.next_entry().await? {
                if e.file_name().to_string_lossy().ends_with(PART) {
                    let age = e.metadata().await.ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).unwrap_or_default();
                    if age >= keep {
                        let _ = fs::remove_file(e.path()).await;
                    }
                }
            }
        }
        if let Some(beside) = w.cache.parent() {
            let mut entries = fs::read_dir(beside).await?;
            while let Some(e) = entries.next_entry().await? {
                if e.file_name().to_string_lossy().starts_with(OLD_CACHE) {
                    tokio::spawn(async move { fs::remove_dir_all(e.path()).await });
                }
            }
        }
        let (bytes, count) = measure(&w.cache);
        w.cache_bytes.store(bytes, Ordering::Relaxed);
        w.cache_count.store(count, Ordering::Relaxed);
        Ok(w)
    }

    /// A name no other file being written has.
    pub fn next_id(&self) -> String {
        format!("{}-{}", now_ms(), self.serial.fetch_add(1, Ordering::Relaxed))
    }

    /// Writes `data` to `path` so that a stop midway leaves the old file (or none).
    pub async fn write_atomic(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let part = path.with_extension(format!("{}{PART}", self.next_id()));
        fs::write(&part, data).await?;
        if let Err(e) = fs::rename(&part, path).await {
            let _ = fs::remove_file(&part).await;
            return Err(e);
        }
        Ok(())
    }

    /// Says what a video's frames are being swapped with now: `plan` names it. Its file's date is
    /// when it last changed, so frames older than that are the first of the video to go.
    pub async fn cache_plan(&self, target: &str, plan: &[u8]) -> io::Result<()> {
        let dir = self.cache.join(target);
        let path = dir.join(PLAN);
        if fs::read(&path).await.is_ok_and(|old| old == plan) {
            return Ok(());
        }
        fs::create_dir_all(&dir).await?;
        self.write_atomic(&path, plan).await?;
        *self.cache_stopped.lock().unwrap() = None;
        Ok(())
    }

    /// The keys of the frames kept of a video.
    pub async fn cache_keys(&self, target: &str) -> io::Result<Vec<String>> {
        let folder = self.cache.join(target);
        let frames = tokio::task::spawn_blocking(move || frames_in(&folder)).await.map_err(io::Error::other)?;
        Ok(frames.iter().filter_map(|f| Some(f.path.file_name()?.to_str()?.to_string())).collect())
    }

    /// Keeps a swapped frame, if there's room for it or it can be made (make_room). A patch's key
    /// names everything that went into it, so one already kept is the same.
    pub async fn cache_put(&self, target: &str, key: &str, data: &[u8]) -> io::Result<()> {
        let dir = self.cache.join(target);
        let path = dir.join(key);
        if fs::try_exists(&path).await? {
            return Ok(());
        }
        let size = data.len() as u64;
        let fits = || self.cache_bytes.load(Ordering::Relaxed) + size <= self.cache_budget;
        if !fits() {
            if self.cache_stopped.lock().unwrap().as_deref() == Some(target) {
                return Ok(());
            }
            self.cache_trim(target).await;
            let mut stopped = self.cache_stopped.lock().unwrap();
            *stopped = (!fits()).then(|| target.to_string());
            if stopped.is_some() {
                return Ok(());
            }
        }
        if let Err(e) = self.write_atomic(&path, data).await {
            if e.kind() != io::ErrorKind::NotFound {
                return Err(e);
            }
            fs::create_dir_all(&dir).await?;
            self.write_atomic(&path, data).await?;
        }
        self.cache_count.fetch_add(1, Ordering::Relaxed);
        self.cache_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(())
    }

    async fn cache_trim(&self, target: &str) {
        let cache = self.cache.clone();
        let target = target.to_string();
        let goal = self.cache_budget / 10 * 9;
        let (bytes, count) = tokio::task::spawn_blocking(move || make_room(&cache, &target, goal)).await.unwrap_or((0, 0));
        self.cache_bytes.store(bytes, Ordering::Relaxed);
        self.cache_count.store(count, Ordering::Relaxed);
    }

    /// Empties the cache at once: the files go in the background. (A rename takes no time, however
    /// many there are; what's left of them goes on the next start, if it stops first.)
    pub async fn cache_clear(&self) -> io::Result<()> {
        let old = self.cache.with_file_name(format!("{OLD_CACHE}{}", self.next_id()));
        fs::rename(&self.cache, &old).await?;
        fs::create_dir_all(&self.cache).await?;
        self.cache_bytes.store(0, Ordering::Relaxed);
        self.cache_count.store(0, Ordering::Relaxed);
        *self.cache_stopped.lock().unwrap() = None;
        tokio::spawn(async move { fs::remove_dir_all(old).await });
        Ok(())
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: &[u8] = &[0; 100];

    /// A workspace of its own, gone at the end of the test.
    struct Temp(PathBuf);

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    async fn workspace(budget: u64) -> (Workspace, Temp) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("facejutsu-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        (Workspace::open(dir.join("outputs"), dir.join("data/work"), dir.join("data/cache"), budget).await.unwrap(), Temp(dir))
    }

    /// Long enough that what comes after has a later date, on any file system.
    async fn later() {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    async fn put(w: &Workspace, target: &str, frames: std::ops::Range<u32>) {
        for i in frames {
            w.cache_put(target, &format!("{i:04}"), FRAME).await.unwrap();
        }
    }

    fn kept(w: &Workspace, target: &str) -> Vec<u32> {
        let mut out: Vec<u32> = frames_in(&w.cache.join(target)).iter().map(|f| f.path.file_name().unwrap().to_str().unwrap().parse().unwrap()).collect();
        out.sort();
        out
    }

    fn usage(w: &Workspace) -> (u64, u64) {
        (w.cache_bytes.load(Ordering::Relaxed), w.cache_count.load(Ordering::Relaxed))
    }

    #[tokio::test]
    async fn a_video_too_long_for_the_budget_keeps_what_it_has() {
        let (w, _t) = workspace(1000).await;
        put(&w, "a", 0..15).await;
        assert_eq!(kept(&w, "a"), (0..10).collect::<Vec<_>>());
        assert_eq!(usage(&w), (1000, 10));
        // the second time through finds them, and they're all still there
        put(&w, "a", 0..15).await;
        assert_eq!(kept(&w, "a"), (0..10).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn other_videos_go_first_the_one_used_longest_ago_first() {
        let (w, _t) = workspace(1000).await;
        put(&w, "old", 0..4).await;
        later().await;
        put(&w, "recent", 0..6).await;
        later().await;
        put(&w, "now", 0..2).await;
        assert_eq!(kept(&w, "old").len(), 2);
        assert_eq!(kept(&w, "recent"), (0..6).collect::<Vec<_>>());
        put(&w, "now", 2..4).await;
        assert!(!w.cache.join("old").exists());
        assert_eq!(kept(&w, "recent"), (0..6).collect::<Vec<_>>());
        assert_eq!(kept(&w, "now"), vec![0, 1, 2, 3]);
        assert_eq!(usage(&w), (1000, 10));
    }

    #[tokio::test]
    async fn another_video_can_take_the_room_of_one_that_stopped() {
        let (w, _t) = workspace(1000).await;
        put(&w, "a", 0..12).await;
        assert_eq!(kept(&w, "a").len(), 10);
        put(&w, "b", 0..10).await;
        assert_eq!(kept(&w, "b").len(), 10);
        assert!(kept(&w, "a").is_empty());
    }

    #[tokio::test]
    async fn frames_from_before_a_plan_changed_make_room_for_new_ones() {
        let (w, _t) = workspace(1000).await;
        w.cache_plan("a", b"first").await.unwrap();
        put(&w, "a", 0..10).await;
        later().await;
        w.cache_plan("a", b"second").await.unwrap();
        later().await;
        put(&w, "a", 100..110).await;
        assert_eq!(kept(&w, "a"), (100..110).collect::<Vec<_>>());
        assert_eq!(usage(&w), (1000, 10));
    }

    #[tokio::test]
    async fn the_same_plan_again_is_no_change() {
        let (w, _t) = workspace(1000).await;
        w.cache_plan("a", b"first").await.unwrap();
        put(&w, "a", 0..10).await;
        later().await;
        w.cache_plan("a", b"first").await.unwrap();
        later().await;
        put(&w, "a", 100..110).await;
        assert_eq!(kept(&w, "a"), (0..10).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_new_plan_lets_a_stopped_video_go_on() {
        let (w, _t) = workspace(1000).await;
        w.cache_plan("a", b"first").await.unwrap();
        put(&w, "a", 0..12).await;
        assert_eq!(kept(&w, "a").len(), 10);
        later().await;
        w.cache_plan("a", b"second").await.unwrap();
        later().await;
        put(&w, "a", 100..102).await;
        assert!(kept(&w, "a").contains(&100) && kept(&w, "a").contains(&101));
        assert!(usage(&w).0 <= 1000);
    }

    #[tokio::test]
    async fn only_frames_count_and_they_are_measured_on_opening() {
        let (w, _t) = workspace(1000).await;
        w.cache_plan("a", b"first").await.unwrap();
        put(&w, "a", 0..3).await;
        std::fs::write(w.cache.join("a").join("left-behind.part"), b"x").unwrap();
        let again = Workspace::open(w.outputs.clone(), w.work.clone(), w.cache.clone(), 1000).await.unwrap();
        assert_eq!(usage(&again), (300, 3));
        assert!(!w.cache.join("a").join("left-behind.part").exists());
        assert!(w.cache.join("a").join(PLAN).exists());
    }

    #[tokio::test]
    async fn an_unfinished_output_stays_for_a_day() {
        let (w, _t) = workspace(1000).await;
        let recent = w.outputs.join("1-1.part");
        let old = w.outputs.join("1-2.part");
        std::fs::write(&recent, b"x").unwrap();
        std::fs::write(&old, b"x").unwrap();
        let long_ago = SystemTime::now() - OUTPUT_PART_KEPT - Duration::from_secs(60);
        std::fs::File::options().write(true).open(&old).unwrap().set_modified(long_ago).unwrap();
        Workspace::open(w.outputs.clone(), w.work.clone(), w.cache.clone(), 1000).await.unwrap();
        assert!(recent.exists());
        assert!(!old.exists());
    }

    #[tokio::test]
    async fn the_keys_are_the_frames_kept() {
        let (w, _t) = workspace(1000).await;
        assert!(w.cache_keys("a").await.unwrap().is_empty());
        w.cache_plan("a", b"first").await.unwrap();
        put(&w, "a", 0..2).await;
        let mut keys = w.cache_keys("a").await.unwrap();
        keys.sort();
        assert_eq!(keys, ["0000", "0001"]);
    }

    #[tokio::test]
    async fn clearing_empties_it_and_it_goes_on() {
        let (w, _t) = workspace(1000).await;
        put(&w, "a", 0..12).await;
        w.cache_clear().await.unwrap();
        assert_eq!(usage(&w), (0, 0));
        put(&w, "a", 0..3).await;
        assert_eq!(kept(&w, "a"), vec![0, 1, 2]);
        assert_eq!(usage(&w), (300, 3));
    }
}
