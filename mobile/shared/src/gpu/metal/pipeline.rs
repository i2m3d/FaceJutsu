//! Swapping faces on the device, in steps a photo and every frame of a video share: detect the
//! faces, take the source identity from one of them, swap faces in a frame and paste them back.
//! Images are uploaded once, as 8 bits; the CPU only decodes detections, aligns (the maps, not the
//! pixels) and computes the latent, between GPU runs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::conv::{buffer_with, Buffer};
use super::graph::{self, host, Arena, Img8, Ran, Sampling};
use super::nets::{Detector, Recognizer};
use super::restore::{mask_taps, Enhancer, Occluder};
use super::swapper::{Swapper, SIZE};
use super::{Device, Gpu};
use crate::enhancer::{self, WIDE_512};
use crate::model::Model;
use crate::occluder;
use crate::photo::{self, compose, edge_mask_profile, invert, similarity, template, Face, Image, Orientation, SWAP_128};

/// The enhancers by crop size, and their files.
const ENHANCERS: [(usize, &str); 2] = [(256, "gpen-bfr-256.int8.safetensors"), (512, "gpen-bfr-512.int8.safetensors")];
const OCCLUDER: &str = "xseg.int8.safetensors";

/// How a face is swapped besides who it becomes: the Swap, Enhance and Mask panels' settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Detail: the swap runs on a crop this many times the swapper's size (1 to 4), in k x k
    /// passes
    pub detail: usize,
    pub enhance: Option<Enhance>,
    /// how softly the swap's edges blend in, 0..1 (and the occlusion mask's)
    pub edges: f64,
    /// Objects: what covers the face (hands, glasses) stays in front
    pub objects: bool,
    /// how much of the face around what covers it is swapped, -1..1
    pub coverage: f64,
    /// the face's own detail when it's known, for a Subtle enhance (none: measured on the frame)
    pub own_detail: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enhance {
    /// the enhancer: 256 (Fast) or 512 (Best)
    pub size: usize,
    /// Subtle: only detail finer than the swap's is added; else (Strong) the face is redrawn
    pub subtle: bool,
    /// Strength, 0..1
    pub strength: f32,
}

impl Default for Look {
    fn default() -> Look {
        Look { detail: 1, enhance: None, edges: 0.3, objects: false, coverage: 0.0, own_detail: None }
    }
}

impl Look {
    /// The framing the enhancer and the occlusion mask see: the enhancer's.
    fn template(&self) -> &'static [[f64; 2]; 5] {
        match self.enhance {
            Some(e) if e.size == 512 => &WIDE_512,
            _ => &SWAP_128,
        }
    }
}

pub(super) struct Models {
    pub detector: Detector,
    pub recognizer: Recognizer,
    pub swapper: Swapper,
    /// where the models are, for those loaded once a face needs them
    dir: PathBuf,
    enhancers: Vec<(usize, Enhancer)>,
    occluder: Option<Occluder>,
    /// edge mask profiles by (size, edges), and the occlusion mask's taps by edges
    profiles: HashMap<(usize, u64), Buffer>,
    taps: HashMap<u64, (Buffer, usize)>,
}

impl Models {
    pub fn load(device: &Device, dir: &Path) -> Result<Models, String> {
        let mut m = Models {
            detector: Detector::load(device, &Model::open(&dir.join("scrfd-10g.bf16.safetensors"))?)?,
            recognizer: Recognizer::load(device, &Model::open(&dir.join("arcface-r50.int8.safetensors"))?)?,
            swapper: Swapper::load(device, &Model::open(&dir.join("inswapper-128.int8.safetensors"))?)?,
            dir: dir.to_path_buf(),
            enhancers: Vec::new(),
            occluder: None,
            profiles: HashMap::new(),
            taps: HashMap::new(),
        };
        m.prepare(device, [&Look::default()])?;
        Ok(m)
    }

    /// Loads the enhancers and the occlusion mask these looks need, if they aren't yet, and makes
    /// their masks.
    pub fn prepare<'a>(&mut self, device: &Device, looks: impl IntoIterator<Item = &'a Look>) -> Result<(), String> {
        for look in looks {
            let k = look.detail.clamp(1, 4);
            self.profile(device, SIZE * k, look.edges)?;
            if let Some(e) = look.enhance {
                if !self.enhancers.iter().any(|(s, _)| *s == e.size) {
                    let file = ENHANCERS.iter().find(|(s, _)| *s == e.size).ok_or(format!("no {} px enhancer", e.size))?.1;
                    let net = Enhancer::load(device, &Model::open(&self.dir.join(file))?)?;
                    self.enhancers.push((e.size, net));
                }
                self.profile(device, e.size, look.edges)?;
            }
            if look.objects {
                if self.occluder.is_none() {
                    self.occluder = Some(Occluder::load(device, &Model::open(&self.dir.join(OCCLUDER))?)?);
                }
                if !self.taps.contains_key(&look.edges.to_bits()) {
                    self.taps.insert(look.edges.to_bits(), mask_taps(device, look.edges)?);
                }
            }
        }
        Ok(())
    }

    fn profile(&mut self, device: &Device, size: usize, edges: f64) -> Result<Buffer, String> {
        if let Some(b) = self.profiles.get(&(size, edges.to_bits())) {
            return Ok(b.clone());
        }
        let b = buffer_with(device, &edge_mask_profile(size, edges))?;
        self.profiles.insert((size, edges.to_bits()), b.clone());
        Ok(b)
    }

    fn enhancer(&self, size: usize) -> Result<&Enhancer, String> {
        self.enhancers.iter().find(|(s, _)| *s == size).map(|(_, e)| e).ok_or(format!("the {size} px enhancer isn't loaded"))
    }

    fn profile_of(&self, size: usize, edges: f64) -> Result<&Buffer, String> {
        self.profiles.get(&(size, edges.to_bits())).ok_or(format!("no {size} px edge mask"))
    }
}

/// The faces found in an image, in upright coordinates, best first, and the detector's head
/// outputs they were decoded from.
pub(super) struct Found {
    pub faces: Vec<Face>,
}

/// Detects the faces in each image, in one GPU run (its buffers from `arena` when given).
pub(super) fn detect(gpu: &mut Gpu, m: &Models, images: &[(&Img8, &Orientation)], arena: Option<&mut Arena>, profile: bool) -> Result<Ran<Vec<Found>>, String> {
    let ran = graph::run_in(&mut gpu.kernels, &gpu.ops, arena, profile, |g| images.iter().map(|(img, o)| m.detector.encode(g, img, o)).collect::<Result<Vec<_>, _>>())?;
    let found = ran
        .value
        .iter()
        .map(|(heads, s)| {
            let heads: [Vec<f32>; 3] = std::array::from_fn(|i| {
                let n = photo::DET_SIZE / (8 << i);
                host(&heads[i], n * n * 30)
            });
            Found { faces: photo::decode_detections([&heads[0], &heads[1], &heads[2]], *s) }
        })
        .collect();
    Ok(Ran { value: found, ms: ran.ms, steps: ran.steps })
}

/// The largest face.
pub(super) fn largest(faces: &[Face]) -> Option<&Face> {
    let area = |f: &&Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
    faces.iter().max_by(|a, b| area(a).total_cmp(&area(b)))
}

/// Who the swapped faces become: a face's embedding and the swapper's latent from it.
pub(super) struct Identity {
    pub embedding: Vec<f32>,
    /// the latent on the GPU
    buf: Buffer,
}

/// The identity of `face` in `img`.
pub(super) fn identity(gpu: &mut Gpu, m: &Models, img: &Img8, o: &Orientation, face: &Face, profile: bool) -> Result<Ran<Identity>, String> {
    let ran = graph::run(&mut gpu.kernels, &gpu.ops, profile, |g| m.recognizer.encode(g, img, o, face))?;
    let embedding = host(&ran.value, 512);
    let latent = photo::latent(&embedding, &m.swapper.emap);
    let buf = buffer_with(&gpu.kernels.device, &latent)?;
    Ok(Ran { value: Identity { embedding, buf }, ms: ran.ms, steps: ran.steps })
}

/// The identity a face's embedding (as `identity` gives it) makes, without looking at the face again.
pub(super) fn identity_of(gpu: &Gpu, m: &Models, embedding: Vec<f32>) -> Result<Identity, String> {
    let latent = photo::latent(&embedding, &m.swapper.emap);
    let buf = buffer_with(&gpu.kernels.device, &latent)?;
    Ok(Identity { embedding, buf })
}

/// The ArcFace embeddings of `faces` in `img`, as they come (not normalized), in one GPU run.
pub(super) fn embed(gpu: &mut Gpu, m: &Models, img: &Img8, o: &Orientation, faces: &[Face]) -> Result<Vec<Vec<f32>>, String> {
    if faces.is_empty() {
        return Ok(Vec::new());
    }
    let ran = graph::run(&mut gpu.kernels, &gpu.ops, false, |g| faces.iter().map(|f| m.recognizer.encode(g, img, o, f)).collect::<Result<Vec<_>, _>>())?;
    Ok(ran.value.iter().map(|b| host(b, 512)).collect())
}

/// Each face's own detail in an image (photo::face_detail), measured on its crop: the crops on
/// the GPU in one run, the spectra on the CPU.
pub(super) fn face_details(gpu: &mut Gpu, img: &Img8, o: &Orientation, faces: &[&Face]) -> Result<Vec<f64>, String> {
    if faces.is_empty() {
        return Ok(Vec::new());
    }
    let n = photo::DETAIL_CROP;
    let ran = graph::run(&mut gpu.kernels, &gpu.ops, false, |g| {
        faces
            .iter()
            .map(|f| {
                let map = similarity(&f.kps, &template(&SWAP_128, n as f64));
                let sampling = Sampling { map: compose(&o.to_stored, &invert(&map)), size: (n, n), valid: (n, n), mul: 1.0, add: 0.0 };
                let crop = g.resample(img, &sampling)?;
                g.to_host(&crop)
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    Ok(ran.value.iter().map(|b| photo::face_detail(&photo::face_spectrum(&Image { w: n, data: host(b, n * n * 3) }))).collect())
}

/// Swaps each face (upright coordinates) in `frame` with its own identity and look, and writes
/// the result into `out` (the frame's size), in one GPU run (after one that measures the faces'
/// own detail, when a Subtle enhance needs it): every face's swap, then the enhancer on each
/// that has it, then what covers each face with Objects put back, from the frame as it came.
/// Faces with the same identity share its styles. `prepare` the models for the looks first.
pub(super) fn swap_planned(gpu: &mut Gpu, m: &Models, frame: &Img8, o: &Orientation, plan: &[(&Face, &Identity, &Look)], out: &Img8, arena: Option<&mut Arena>, profile: bool) -> Result<Ran<()>, String> {
    if (out.h, out.w) != (frame.h, frame.w) {
        return Err(format!("a {}x{} frame into a {}x{} one", frame.w, frame.h, out.w, out.h));
    }
    let subtle: Vec<usize> = (0..plan.len()).filter(|i| plan[*i].2.enhance.is_some_and(|e| e.subtle) && plan[*i].2.own_detail.is_none()).collect();
    let measured = face_details(gpu, frame, o, &subtle.iter().map(|i| plan[*i].0).collect::<Vec<_>>())?;
    let mut details: Vec<Option<f64>> = plan.iter().map(|p| p.2.own_detail).collect();
    for (i, d) in subtle.iter().zip(measured) {
        details[*i] = Some(d);
    }
    let to_upright = invert(&o.to_stored);
    // upright frame -> a crop of `size` px in a template's framing, and how to sample it
    let crop = |face: &Face, tpl: &[[f64; 2]; 5], size: usize, sub: usize, mul: f32, add: f32| {
        let map = similarity(&face.kps, &template(tpl, size as f64));
        let s = Sampling { map: compose(&o.to_stored, &invert(&map)), size: (sub, sub), valid: (sub, sub), mul, add };
        (map, s)
    };
    graph::run_in(&mut gpu.kernels, &gpu.ops, arena, profile, |g| {
        // crops come from the frame as it came, faces are pasted into the copy
        g.copy(frame, out)?;
        let mut styles: Vec<(*const Identity, Buffer)> = Vec::new();
        for (face, who, look) in plan {
            let key = *who as *const Identity;
            let at = match styles.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    styles.push((key, m.swapper.styles(g, &who.buf)?));
                    styles.len() - 1
                }
            };
            let k = look.detail.clamp(1, 4);
            let size = SIZE * k;
            let (map, sampling) = crop(face, &SWAP_128, size, SIZE, 1.0 / 255.0, 0.0);
            // Detail: k x k swaps of the crop's sub-crops, side by side in one buffer
            let subs = if k > 1 { Some(g.f32_buffer(size * size * 3)?) } else { None };
            let mut swapped = None;
            for s in 0..k * k {
                let x = g.resample_sub(frame, &sampling, k, s % k, s / k)?;
                let dest = subs.as_ref().map(|b| (b, s * SIZE * SIZE * 3 * 4));
                swapped = Some(m.swapper.encode_to(g, &x, &styles[at].1, dest)?.output);
            }
            let swapped = swapped.unwrap();
            g.paste_with(out, &swapped, size, k, m.profile_of(size, look.edges)?, &compose(&map, &to_upright), 1.0, false)?;
        }
        for (i, (face, _, look)) in plan.iter().enumerate() {
            let Some(e) = look.enhance else { continue };
            let net = m.enhancer(e.size)?;
            let n = net.size;
            let tpl = look.template();
            // the swapped face, in [-1, 1]
            let (map, sampling) = crop(face, tpl, n, n, 2.0 / 255.0, -1.0);
            let x = g.resample(out, &sampling)?;
            let restored = net.encode(g, &x)?;
            let y = if e.subtle {
                let lo = enhancer::band(enhancer::swap_detail(look.detail, details[i]), tpl);
                let band = g.band_down(&x, &restored, lo)?;
                g.enhance_out(&restored, Some((&band, lo)), &x)?
            } else {
                g.enhance_out(&restored, None, &x)?
            };
            g.paste_with(out, &y, n, 1, m.profile_of(n, look.edges)?, &compose(&map, &to_upright), e.strength, e.subtle)?;
        }
        for (face, _, look) in plan {
            if !look.objects {
                continue;
            }
            let occ = m.occluder.as_ref().ok_or("the occlusion mask isn't loaded")?;
            let n = occluder::SIZE;
            let (map, sampling) = crop(face, look.template(), n, n, 1.0 / 255.0, 0.0);
            let x = g.resample(frame, &sampling)?;
            let y = occ.encode(g, &x)?;
            let (taps, r) = m.taps.get(&look.edges.to_bits()).ok_or("no occlusion mask blur")?;
            let mask = g.occlusion_mask(&y, n, occluder::grow(look.coverage, n), taps, *r)?;
            g.unocclude(out, frame, &mask, n, &compose(&map, &to_upright))?;
        }
        Ok(())
    })
}
