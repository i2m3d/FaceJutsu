//! A video's faces for the apps: `VideoScan` keeps the analysis of `track` and does its GPU work
//! (detecting, embedding) through a `FaceSwapper`. The app decodes the frames and decides which to
//! look at when: the skim's keyframes, and passes over stretches in order around the playhead.

use std::sync::{Arc, Mutex};

use crate::photo::Face;
use crate::session::{EngineError, FaceSwapper, FoundFace};
use crate::track::{self, normalize, Scan, Thumb, LOOKED, SKIMMED};

/// A face of a frame, and who it is.
#[derive(uniffi::Record, Debug, Clone)]
pub struct ScanFace {
    pub face: FoundFace,
    pub person: u32,
    /// the face's own detail around this frame, for a Subtle enhance (none: not measured)
    pub detail: Option<f64>,
}

/// A person of the analysis and their time on screen so far: `runs` are frame ranges [first,
/// last + 1), flattened.
#[derive(uniffi::Record, Debug, Clone)]
pub struct ScanPerson {
    pub person: u32,
    /// who they are: the person they were found to be one with, or themselves
    pub root: u32,
    pub frames: u32,
    /// some of their frames are only skimmed keyframes
    pub loose: bool,
    pub runs: Vec<u32>,
}

/// A clearly better view of a person than any before, in the frame just looked at: where to cut
/// their thumbnail from (upright pixels).
#[derive(uniffi::Record, Debug, Clone)]
pub struct ThumbWanted {
    pub person: u32,
    pub bbox: Vec<f64>,
}

fn thumbs(t: Vec<Thumb>) -> Vec<ThumbWanted> {
    t.into_iter().map(|t| ThumbWanted { person: t.person as u32, bbox: t.bbox.to_vec() }).collect()
}

#[derive(uniffi::Object)]
pub struct VideoScan {
    swapper: Arc<FaceSwapper>,
    inner: Mutex<Scan>,
}

impl VideoScan {
    fn scan(&self) -> Result<std::sync::MutexGuard<'_, Scan>, EngineError> {
        self.inner.lock().map_err(|_| EngineError::Failed("the scan failed before".into()))
    }

    fn check(&self, i: u32) -> Result<usize, EngineError> {
        let n = self.scan()?.a.times.len();
        if (i as usize) < n { Ok(i as usize) } else { Err(EngineError::Failed(format!("frame {i} of {n}"))) }
    }

    /// The faces in a frame (GPU).
    fn detect(&self, input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<Vec<Face>, EngineError> {
        Ok(self.swapper.detect(input, width, height, stride, rotation)?.iter().map(|f| f.face()).collect::<Result<_, _>>()?)
    }
}

/// Embeds faces of a frame through the swapper, normalized.
#[allow(unused_variables)]
fn embedder(swapper: &FaceSwapper, input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> impl FnMut(&[Face]) -> Result<Vec<Vec<f32>>, String> + '_ {
    move |faces: &[Face]| {
        #[cfg(target_vendor = "apple")]
        {
            let f = crate::session::frame(input, width, height, stride, rotation)?;
            let es = swapper.with(|s| unsafe { s.embed_faces(&f, faces) }).map_err(|e| e.to_string())?;
            Ok(es.iter().map(|e| normalize(e)).collect())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (swapper, faces, normalize as fn(&[f32]) -> Vec<f32>);
            Err("no GPU backend on this platform yet".to_string())
        }
    }
}

/// Each face's own detail in a frame, through the swapper.
#[allow(unused_variables)]
fn own_details(swapper: &FaceSwapper, faces: &[Face], input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<Vec<f64>, String> {
    #[cfg(target_vendor = "apple")]
    {
        let f = crate::session::frame(input, width, height, stride, rotation)?;
        swapper.with(|s| unsafe { s.face_details(&f, faces) }).map_err(|e| e.to_string())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        Err("no GPU backend on this platform yet".to_string())
    }
}

#[uniffi::export]
impl VideoScan {
    /// A new analysis of a video whose frames are at `times` (seconds from the first, ascending),
    /// with keyframes `keys` (frame indices).
    #[uniffi::constructor]
    pub fn new(swapper: Arc<FaceSwapper>, times: Vec<f64>, keys: Vec<u32>) -> Arc<VideoScan> {
        let keys = keys.into_iter().map(|k| k as usize).filter(|&k| k < times.len()).collect();
        Arc::new(VideoScan { swapper, inner: Mutex::new(Scan::new(times, keys)) })
    }

    /// An analysis saved with `save`.
    #[uniffi::constructor]
    pub fn restore(swapper: Arc<FaceSwapper>, saved: Vec<u8>) -> Result<Arc<VideoScan>, EngineError> {
        let a: track::Analysis = serde_json::from_slice(&saved).map_err(|e| EngineError::Failed(format!("the saved faces: {e}")))?;
        if a.state.len() != a.times.len() || a.faces.len() != a.times.len() {
            return Err(EngineError::Failed("the saved faces don't match their frames".into()));
        }
        Ok(Arc::new(VideoScan { swapper, inner: Mutex::new(Scan::from(a)) }))
    }

    pub fn save(&self) -> Result<Vec<u8>, EngineError> {
        serde_json::to_vec(&self.scan()?.a).map_err(|e| EngineError::Failed(e.to_string()))
    }

    pub fn times(&self) -> Vec<f64> {
        self.scan().map(|s| s.a.times.clone()).unwrap_or_default()
    }

    pub fn keys(&self) -> Vec<u32> {
        self.scan().map(|s| s.a.keys.iter().map(|&k| k as u32).collect()).unwrap_or_default()
    }

    /// Per frame: 0 not looked at, 1 its keyframe skimmed, 2 looked at by a pass.
    pub fn states(&self) -> Vec<u8> {
        self.scan().map(|s| s.a.state.clone()).unwrap_or_default()
    }

    pub fn state(&self, frame: u32) -> u8 {
        self.scan().ok().and_then(|s| s.a.state.get(frame as usize).copied()).unwrap_or(0)
    }

    /// The keyframes of frames [from, to) still to skim, in the turn the skim takes them: coarse
    /// to fine, `step` seconds apart at the finest, the ones near frame `at` first.
    pub fn skim_order(&self, at: u32, from: u32, to: u32, step: f64) -> Vec<u32> {
        let Ok(s) = self.scan() else { return Vec::new() };
        let to = (to as usize).min(s.a.times.len());
        s.skim_order(at as usize, from as usize, to, step).into_iter().filter(|&k| s.a.state[k] == 0).map(|k| k as u32).collect()
    }

    /// The skim's look at keyframe `frame` (a BGRA frame as `FaceSwapper::detect` takes it): its
    /// faces and who they are.
    pub fn glance(&self, frame: u32, input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<Vec<ThumbWanted>, EngineError> {
        let k = self.check(frame)?;
        if self.scan()?.a.state[k] != 0 {
            return Ok(Vec::new());
        }
        let found = self.detect(input, width, height, stride, rotation)?;
        let mut embed = embedder(&self.swapper, input, width, height, stride, rotation);
        Ok(thumbs(self.scan()?.glance(k, found, &mut embed)?))
    }

    /// Starts a pass: a walk over frames in order from frame `from`.
    pub fn begin_pass(&self, from: u32) {
        if let Ok(mut s) = self.scan() {
            s.begin_pass(from as usize);
        }
    }

    /// The pass's look at frame `frame`, the next in order: its faces linked to the frames
    /// before, and who they are. A frame looked at before only goes on into the pass (its pixels
    /// aren't used, and `input` may be 0).
    pub fn look(&self, frame: u32, input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<Vec<ThumbWanted>, EngineError> {
        let i = self.check(frame)?;
        let state = self.scan()?.a.state[i];
        let found = if state == LOOKED || state == SKIMMED { None } else { Some(self.detect(input, width, height, stride, rotation)?) };
        let measure = found.clone();
        let mut embed = embedder(&self.swapper, input, width, height, stride, rotation);
        let wanted = thumbs(self.scan()?.look(i, found, &mut embed)?);
        // (a face without it goes by what's measured around it, or at the swap)
        if let Some(Ok(details)) = measure.filter(|f| !f.is_empty()).map(|f| own_details(&self.swapper, &f, input, width, height, stride, rotation)) {
            self.scan()?.set_details(i, &details);
        }
        Ok(wanted)
    }

    /// Ends the pass before frame `to`.
    pub fn end_pass(&self, to: u32) {
        if let Ok(mut s) = self.scan() {
            s.end_pass(to as usize);
        }
    }

    /// Frame `frame`'s faces (upright pixels) and who they are.
    pub fn faces(&self, frame: u32) -> Vec<ScanFace> {
        let Ok(s) = self.scan() else { return Vec::new() };
        s.faces(frame as usize).iter().map(|r| ScanFace { face: FoundFace { bbox: r.bbox.to_vec(), score: r.score, kps: r.kps.iter().flatten().copied().collect() }, person: r.person as u32, detail: r.detail }).collect()
    }

    /// Per frame, the analysis person of each face (who they were first taken for, not the person
    /// they were later found to be one with).
    pub fn frame_people(&self) -> Vec<Vec<u32>> {
        let Ok(s) = self.scan() else { return Vec::new() };
        s.a.faces.iter().map(|f| f.iter().flatten().map(|r| r.person as u32).collect()).collect()
    }

    /// Everyone found so far and their time on screen.
    pub fn people(&self) -> Vec<ScanPerson> {
        let Ok(mut s) = self.scan() else { return Vec::new() };
        s.regroup();
        s.stretches()
            .into_iter()
            .map(|p| ScanPerson { person: p.person as u32, root: p.root as u32, frames: p.frames as u32, loose: p.loose, runs: p.runs.iter().flat_map(|&(a, b)| [a as u32, b as u32]).collect() })
            .collect()
    }

    /// How much of frames [from, to) a pass has looked at, 0 to 1.
    pub fn coverage(&self, from: u32, to: u32) -> f64 {
        self.scan().map(|s| s.coverage(from as usize, to as usize)).unwrap_or(0.0)
    }
}
