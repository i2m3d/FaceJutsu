//! A swapping session for the apps: the GPU and the models set up once, the identities faces are
//! swapped to, and frames looked at and swapped one after another (a photo is a video of one
//! frame).

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use super::graph::{Arena, Img8, Ops};
use super::pipeline::{detect, embed, face_details, identity, identity_of, largest, swap_planned, Identity, Look, Models};
use super::{conv, Device, Gpu};
use crate::photo::{Face, Orientation};

/// The memory page size on Apple's arm64 devices: what memory the GPU uses in place must be
/// aligned to and a whole number of.
const PAGE: usize = 16384;

pub struct Session {
    gpu: Gpu,
    models: Models,
    /// the identities faces are swapped to, by the app's id
    identities: HashMap<u32, Identity>,
    detect_arena: Arena,
    swap_arena: Arena,
}

// SAFETY: Metal devices, queues, buffers and pipeline states may be used from any thread; command
// buffers and encoders are made and finished within one call. The apps hold a session behind a
// mutex, so one thread uses it at a time.
unsafe impl Send for Session {}

/// A frame of the caller's: where it is and how it's laid out and shown.
#[derive(Clone, Copy)]
pub struct Frame {
    pub ptr: *const u8,
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub rotation: u32,
}

/// The caller's memory at `ptr`, `rows` rows of `stride` bytes, as a buffer the GPU uses in place,
/// when it's page-aligned (as pixel buffers are).
///
/// # Safety
/// The memory must be allocated in whole pages (as pixel buffers and large allocations are) and
/// stay valid while the buffer is in use.
unsafe fn wrap(device: &Device, ptr: *mut u8, w: usize, rows: usize, stride: usize) -> Option<Img8> {
    if ptr as usize % PAGE != 0 {
        return None;
    }
    let len = (rows * stride).next_multiple_of(PAGE);
    let ptr = NonNull::new(ptr as *mut c_void)?;
    // SAFETY: page-aligned memory of whole pages, as the caller promises.
    let buf = unsafe { device.newBufferWithBytesNoCopy_length_options_deallocator(ptr, len, MTLResourceOptions::StorageModeShared, None) }?;
    Some(Img8 { buf, h: rows, w, stride, bgra: true })
}

impl Session {
    pub fn new(models_dir: &str) -> Result<Session, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let ops = Ops::new(&device)?;
        let models = Models::load(&device, Path::new(models_dir))?;
        Ok(Session {
            gpu: Gpu { kernels: conv::Kernels::new(device, queue), ops },
            models,
            identities: HashMap::new(),
            detect_arena: Arena::default(),
            swap_arena: Arena::default(),
        })
    }

    fn upload_photo(&self, pixels: &[u8], w: usize, h: usize, stride: usize, bgra: bool) -> Result<Img8, String> {
        if pixels.len() < h * stride {
            return Err(format!("{} bytes for {h} rows of {stride}", pixels.len()));
        }
        let img = Img8::shared(&self.gpu.kernels.device.clone(), h, w, stride, bgra)?;
        // SAFETY: the buffer is h * stride bytes of shared memory the GPU isn't using.
        unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), img.bytes(), h * stride) };
        Ok(img)
    }

    /// The largest face in a photo (RGB or BGRA, rows `stride` bytes apart, upright) and its
    /// embedding, which `set_identity` makes an identity of.
    pub fn source_face(&mut self, pixels: &[u8], w: usize, h: usize, stride: usize, bgra: bool) -> Result<(Face, Vec<f32>), String> {
        objc2::rc::autoreleasepool(|_| {
            let img = self.upload_photo(pixels, w, h, stride, bgra)?;
            let o = Orientation::upright(w, h);
            let found = detect(&mut self.gpu, &self.models, &[(&img, &o)], None, false)?;
            let face = largest(&found.value[0].faces).ok_or("no face in the photo")?.clone();
            let who = identity(&mut self.gpu, &self.models, &img, &o, &face, false)?.value;
            Ok((face, who.embedding))
        })
    }

    /// Faces given identity `id` become the face `embedding` is of.
    pub fn set_identity(&mut self, id: u32, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != 512 {
            return Err(format!("an embedding of {} values", embedding.len()));
        }
        let who = identity_of(&self.gpu, &self.models, embedding)?;
        self.identities.insert(id, who);
        Ok(())
    }

    /// The caller's frame on the GPU: in place when it's page-aligned, else copied to staging.
    ///
    /// # Safety
    /// As `Frame` says: readable for the whole call.
    unsafe fn frame_in(&self, f: &Frame) -> Result<Img8, String> {
        if f.stride < f.w * 4 {
            return Err(format!("rows of {} bytes for a {} px wide BGRA frame", f.stride, f.w));
        }
        let device = self.gpu.kernels.device.clone();
        // SAFETY: as the caller promises; the GPU only reads it.
        if let Some(img) = unsafe { wrap(&device, f.ptr as *mut u8, f.w, f.h, f.stride) } {
            return Ok(img);
        }
        let img = Img8::shared(&device, f.h, f.w, f.stride, true)?;
        // SAFETY: the caller's frame is h * stride bytes; the new buffer is that size and unused.
        unsafe { std::ptr::copy_nonoverlapping(f.ptr, img.bytes(), f.h * f.stride) };
        Ok(img)
    }

    /// The faces in a BGRA frame, the right way up (upright coordinates), best first.
    ///
    /// # Safety
    /// The frame must be readable for the whole call.
    pub unsafe fn detect_frame(&mut self, f: &Frame) -> Result<Vec<Face>, String> {
        objc2::rc::autoreleasepool(|_| {
            let o = Orientation::new(f.rotation, f.w, f.h)?;
            // SAFETY: as the caller promises.
            let img = unsafe { self.frame_in(f)? };
            let found = detect(&mut self.gpu, &self.models, &[(&img, &o)], Some(&mut self.detect_arena), false)?;
            Ok(found.value.into_iter().next().map(|x| x.faces).unwrap_or_default())
        })
    }

    /// The embeddings of faces in a BGRA frame (upright coordinates), as they come.
    ///
    /// # Safety
    /// The frame must be readable for the whole call.
    pub unsafe fn embed_faces(&mut self, f: &Frame, faces: &[Face]) -> Result<Vec<Vec<f32>>, String> {
        objc2::rc::autoreleasepool(|_| {
            let o = Orientation::new(f.rotation, f.w, f.h)?;
            // SAFETY: as the caller promises.
            let img = unsafe { self.frame_in(f)? };
            embed(&mut self.gpu, &self.models, &img, &o, faces)
        })
    }

    /// Each face's own detail in a BGRA frame (upright coordinates).
    ///
    /// # Safety
    /// The frame must be readable for the whole call.
    pub unsafe fn face_details(&mut self, f: &Frame, faces: &[Face]) -> Result<Vec<f64>, String> {
        objc2::rc::autoreleasepool(|_| {
            let o = Orientation::new(f.rotation, f.w, f.h)?;
            // SAFETY: as the caller promises.
            let img = unsafe { self.frame_in(f)? };
            face_details(&mut self.gpu, &img, &o, &faces.iter().collect::<Vec<_>>())
        })
    }

    /// Swaps the planned faces of a BGRA frame (upright coordinates), each with the identity of its
    /// id, into `out_stride`-byte rows at `output`; the rest of the frame is copied as it is.
    /// Returns the GPU time in ms.
    ///
    /// # Safety
    /// The frame must be readable, and `output` writable for `h` rows of `out_stride` bytes, for the
    /// whole call; they mustn't overlap. Page-aligned memory must be allocated in whole pages.
    pub unsafe fn swap_plan(&mut self, f: &Frame, output: *mut u8, out_stride: usize, plan: &[(Face, u32, Look)]) -> Result<f64, String> {
        objc2::rc::autoreleasepool(|_| {
            let o = Orientation::new(f.rotation, f.w, f.h)?;
            if out_stride < f.w * 4 {
                return Err(format!("rows of {out_stride} bytes for a {} px wide BGRA frame", f.w));
            }
            let device = self.gpu.kernels.device.clone();
            self.models.prepare(&device, plan.iter().filter(|p| self.identities.contains_key(&p.1)).map(|p| &p.2))?;
            let mut faces = Vec::new();
            for (face, id, look) in plan {
                // (a character removed since is left as it is)
                if let Some(who) = self.identities.get(id) {
                    faces.push((face, who, look));
                }
            }
            if faces.is_empty() {
                for r in 0..f.h {
                    // SAFETY: rows of w * 4 bytes, inside both the caller's buffers.
                    unsafe { std::ptr::copy_nonoverlapping(f.ptr.add(r * f.stride), output.add(r * out_stride), f.w * 4) };
                }
                return Ok(0.0);
            }
            // SAFETY: as the caller promises.
            let frame = unsafe { self.frame_in(f)? };
            let out_gpu = unsafe { wrap(&device, output, f.w, f.h, out_stride) };
            let staged = match &out_gpu {
                Some(_) => None,
                None => Some(Img8::shared(&device, f.h, f.w, out_stride, true)?),
            };
            let out = out_gpu.as_ref().or(staged.as_ref()).unwrap();
            let plan: Vec<(&Face, &Identity, &Look)> = faces.iter().map(|(f, w, l)| (*f, *w, *l)).collect();
            let ms = swap_planned(&mut self.gpu, &self.models, &frame, &o, &plan, out, Some(&mut self.swap_arena), false)?.ms;
            if let Some(s) = &staged {
                // SAFETY: the staging buffer is h * out_stride bytes, like the caller's, and done.
                unsafe { std::ptr::copy_nonoverlapping(s.bytes(), output, f.h * out_stride) };
            }
            Ok(ms)
        })
    }
}
