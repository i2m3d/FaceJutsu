//! What the apps swap faces with: a `FaceSwapper` loads the models once, makes identities from the
//! faces in photos, finds the faces in frames and swaps them, each with its own identity (video
//! frames, or a photo as one frame).

use std::sync::{Arc, Mutex};

use crate::photo::Face;

#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum EngineError {
    Failed(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Failed(why) => f.write_str(why),
        }
    }
}

impl From<String> for EngineError {
    fn from(why: String) -> EngineError {
        EngineError::Failed(why)
    }
}

/// A face as found: its box [x1, y1, x2, y2], score, and five keypoints [x, y] flattened (eyes,
/// nose, mouth corners), in the upright image's pixels.
#[derive(uniffi::Record, Debug, Clone)]
pub struct FoundFace {
    pub bbox: Vec<f64>,
    pub score: f64,
    pub kps: Vec<f64>,
}

impl From<&Face> for FoundFace {
    fn from(f: &Face) -> FoundFace {
        FoundFace { bbox: f.bbox.to_vec(), score: f.score, kps: f.kps.iter().flatten().copied().collect() }
    }
}

impl FoundFace {
    pub(crate) fn face(&self) -> Result<Face, String> {
        if self.bbox.len() != 4 || self.kps.len() != 10 {
            return Err(format!("a face with {} box values and {} keypoint values", self.bbox.len(), self.kps.len()));
        }
        let k = &self.kps;
        Ok(Face {
            bbox: [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]],
            score: self.score,
            kps: [[k[0], k[1]], [k[2], k[3]], [k[4], k[5]], [k[6], k[7]], [k[8], k[9]]],
        })
    }
}

/// The face of a photo that becomes a character, and its embedding (for `set_identity`).
#[derive(uniffi::Record, Debug, Clone)]
pub struct SourceFace {
    pub face: FoundFace,
    pub embedding: Vec<f32>,
}

/// A face to swap in a frame, the identity it becomes, and how (its stretch's Swap, Enhance and
/// Mask settings).
#[derive(uniffi::Record, Debug, Clone)]
pub struct PlannedFace {
    pub face: FoundFace,
    pub identity: u32,
    /// Detail: 1 (Fast), 2 (Balanced) or 3 (Max), the swap run on a crop this many times its size
    #[uniffi(default = 1)]
    pub detail: u32,
    /// Enhance's Quality: 0 (Off), 256 (Fast) or 512 (Best), the enhancer's size
    #[uniffi(default = 0)]
    pub enhance: u32,
    /// Style: Subtle (only detail finer than the swap's is added) or Strong (the face redrawn)
    #[uniffi(default = true)]
    pub subtle: bool,
    #[uniffi(default = 0.8)]
    pub strength: f32,
    /// Edges, 0..1
    #[uniffi(default = 0.3)]
    pub edges: f32,
    /// Objects: what covers the face stays in front
    #[uniffi(default = false)]
    pub objects: bool,
    /// Coverage, -1..1
    #[uniffi(default = 0.0)]
    pub coverage: f32,
    /// The face's own detail as the scan measured it, for a Subtle enhance (none: measured here)
    #[uniffi(default = None)]
    pub own_detail: Option<f64>,
}

impl PlannedFace {
    #[cfg(target_vendor = "apple")]
    fn look(&self) -> Result<crate::gpu::Look, String> {
        let enhance = match self.enhance {
            0 => None,
            256 | 512 => Some(crate::gpu::Enhance { size: self.enhance as usize, subtle: self.subtle, strength: self.strength.clamp(0.0, 1.0) }),
            n => return Err(format!("no {n} px enhancer")),
        };
        Ok(crate::gpu::Look {
            detail: self.detail.clamp(1, 4) as usize,
            enhance,
            edges: self.edges.clamp(0.0, 1.0) as f64,
            objects: self.objects,
            coverage: self.coverage.clamp(-1.0, 1.0) as f64,
            own_detail: self.own_detail,
        })
    }
}

#[derive(uniffi::Object)]
pub struct FaceSwapper {
    #[cfg(target_vendor = "apple")]
    pub(crate) inner: Mutex<crate::gpu::Session>,
    #[cfg(not(target_vendor = "apple"))]
    pub(crate) inner: Mutex<()>,
}

#[cfg(not(target_vendor = "apple"))]
const NO_GPU: &str = "no GPU backend on this platform yet";

/// A BGRA frame of the caller's: `height` rows of `stride` bytes at `input`, shown turned
/// `rotation` degrees clockwise.
#[cfg(target_vendor = "apple")]
pub(crate) fn frame(input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<crate::gpu::Frame, String> {
    if input == 0 {
        return Err("no pixels".into());
    }
    Ok(crate::gpu::Frame { ptr: input as usize as *const u8, w: width as usize, h: height as usize, stride: stride as usize, rotation })
}

impl FaceSwapper {
    /// Runs `f` on the GPU session.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn with<T>(&self, f: impl FnOnce(&mut crate::gpu::Session) -> Result<T, String>) -> Result<T, EngineError> {
        let mut s = self.inner.lock().map_err(|_| EngineError::Failed("the swapper failed before".into()))?;
        Ok(f(&mut s)?)
    }

    #[cfg(not(target_vendor = "apple"))]
    pub(crate) fn with<T>(&self, _f: impl FnOnce(&mut ()) -> Result<T, String>) -> Result<T, EngineError> {
        let _ = &self.inner;
        Err(EngineError::Failed(NO_GPU.into()))
    }
}

#[uniffi::export]
impl FaceSwapper {
    /// Sets up the GPU and loads the models in `models_dir`.
    #[uniffi::constructor]
    pub fn new(models_dir: String) -> Result<Arc<FaceSwapper>, EngineError> {
        #[cfg(target_vendor = "apple")]
        return Ok(Arc::new(FaceSwapper { inner: Mutex::new(crate::gpu::Session::new(&models_dir)?) }));
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (models_dir, Mutex::new(()));
            Err(EngineError::Failed(NO_GPU.into()))
        }
    }

    /// The largest face in an upright photo (`height` rows of `stride` bytes, BGRA or else RGB) and
    /// its embedding, to make a character of.
    pub fn source_face(&self, pixels: Vec<u8>, width: u32, height: u32, stride: u32, bgra: bool) -> Result<SourceFace, EngineError> {
        #[cfg(target_vendor = "apple")]
        return self.with(|s| {
            let (face, embedding) = s.source_face(&pixels, width as usize, height as usize, stride as usize, bgra)?;
            Ok(SourceFace { face: (&face).into(), embedding })
        });
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (pixels, width, height, stride, bgra);
            self.with(|_| Err(String::new()))
        }
    }

    /// Faces planned with identity `id` become the face `embedding` is of (from `source_face`).
    pub fn set_identity(&self, id: u32, embedding: Vec<f32>) -> Result<(), EngineError> {
        #[cfg(target_vendor = "apple")]
        return self.with(|s| s.set_identity(id, embedding));
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (id, embedding);
            self.with(|_| Ok(()))
        }
    }

    /// The faces in a BGRA frame (`height` rows of `stride` bytes at `input`, shown turned
    /// `rotation` degrees clockwise), in upright coordinates, best first. The memory must stay
    /// valid until this returns.
    pub fn detect(&self, input: u64, width: u32, height: u32, stride: u32, rotation: u32) -> Result<Vec<FoundFace>, EngineError> {
        #[cfg(target_vendor = "apple")]
        return self.with(|s| {
            let f = frame(input, width, height, stride, rotation)?;
            // SAFETY: the caller promises the frame for the call, as documented above.
            Ok(unsafe { s.detect_frame(&f)? }.iter().map(FoundFace::from).collect())
        });
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (input, width, height, stride, rotation);
            self.with(|_| Ok(Vec::new()))
        }
    }

    /// Swaps the planned faces of a BGRA frame (upright coordinates, as `detect` gives them), each
    /// with its identity, into `output` (`height` rows of `output_stride` bytes); the rest of the
    /// frame is copied as it is. Both must stay valid and not overlap until this returns. Returns
    /// the GPU time in ms.
    pub fn swap_planned(&self, input: u64, output: u64, width: u32, height: u32, input_stride: u32, output_stride: u32, rotation: u32, plan: Vec<PlannedFace>) -> Result<f64, EngineError> {
        if output == 0 {
            return Err(EngineError::Failed("no pixels".into()));
        }
        #[cfg(target_vendor = "apple")]
        return self.with(|s| {
            let f = frame(input, width, height, input_stride, rotation)?;
            let plan = plan.iter().map(|p| Ok((p.face.face()?, p.identity, p.look()?))).collect::<Result<Vec<_>, String>>()?;
            // SAFETY: the caller promises both buffers for the call, as documented above.
            unsafe { s.swap_plan(&f, output as usize as *mut u8, output_stride as usize, &plan) }
        });
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (input, width, height, input_stride, output_stride, rotation, plan);
            self.with(|_| Ok(0.0))
        }
    }

}

