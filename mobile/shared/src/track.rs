//! The faces of a video, told apart, as the desktop does it (web/app/tracks.js, scanner.js). Two
//! looks at the frames share one analysis:
//!
//! - the skim looks at keyframes over the whole video (they decode on their own, so it's quick)
//!   and says who is in it, by the faces' ArcFace embeddings;
//! - a pass looks at every frame of a stretch, in order: a tracker links each frame's faces into
//!   tracklets (a face's box overlapping its box in a recent frame), and each tracklet becomes the
//!   person its embedding is most like (one found before, or a new one).
//!
//! Frames looked at stay looked at. People found as two who are one (a face that turned away gets
//! a new tracklet) are found to be one once their embeddings agree and they're never on screen
//! together. This is CPU work over the detector's output; the GPU work comes in through closures.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::photo::Face;

/// frames a face may go missing and stay on its track
pub const GAP: usize = 12;
/// overlap with a tracklet's last box that continues it
const LINK_IOU: f64 = 0.3;
/// cosine similarity of a face to a person's mean embedding that makes it them
pub const SAME_PERSON: f32 = 0.4;
/// embeddings per tracklet at most, one every EMBED_EVERY of its frames
const EMBEDS: usize = 6;
const EMBED_EVERY: usize = 12;
/// frames either side of a frame that a face's own detail is the median of
const DETAIL_SPAN: usize = 8;
/// a new thumbnail only for a view this much better
const BETTER_VIEW: f64 = 1.3;

/// What a frame is at: not looked at, its keyframe skimmed, looked at by a pass.
pub const NOT_LOOKED: u8 = 0;
pub const SKIMMED: u8 = 1;
pub const LOOKED: u8 = 2;

/// A face found in a frame, in upright pixels, and who it is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rec {
    pub bbox: [f64; 4],
    pub score: f64,
    pub kps: [[f64; 2]; 5],
    pub person: usize,
    /// the face's own detail, measured when a pass looks at the frame
    #[serde(default)]
    pub detail: Option<f64>,
}

impl Rec {
    fn face(&self) -> Face {
        Face { bbox: self.bbox, score: self.score, kps: self.kps }
    }
}

/// A person of the analysis: the sum of their faces' normalized embeddings, and the person they
/// were found to be, when the two are one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PersonSum {
    pub sum: Option<Vec<f32>>,
    pub n: i32,
    pub into: Option<usize>,
}

/// Everything found in a video, saved with the work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analysis {
    /// every frame's time, seconds from the first, ascending
    pub times: Vec<f64>,
    /// the keyframes' indices, ascending
    pub keys: Vec<usize>,
    /// per frame: NOT_LOOKED, SKIMMED or LOOKED
    pub state: Vec<u8>,
    /// per frame, the faces found (none: not looked at)
    pub faces: Vec<Option<Vec<Rec>>>,
    pub people: Vec<PersonSum>,
}

/// A person's time on screen so far: how many frames, whether some are only skimmed keyframes, and
/// the runs of frames [first, last + 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Stretches {
    pub person: usize,
    /// who they are: the person they were found to be one with, or themselves
    pub root: usize,
    pub frames: usize,
    pub loose: bool,
    pub runs: Vec<(usize, usize)>,
}

/// A clearly better view of a person than any before: the frame to cut their thumbnail from.
#[derive(Clone, Debug)]
pub struct Thumb {
    pub person: usize,
    pub bbox: [f64; 4],
}

pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n = if n > 0.0 { n } else { 1.0 };
    v.iter().map(|x| x / n).collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn area(b: &[f64; 4]) -> f64 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

pub fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let i = [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
    let inter = area(&i);
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// how good a view of the face is for a thumbnail: big and confidently detected
fn quality(f: &Face) -> f64 {
    area(&f.bbox) * f.score
}

/// the index of the value in ascending `times` nearest to t
pub fn nearest(times: &[f64], t: f64) -> usize {
    if times.is_empty() {
        return 0;
    }
    let (mut lo, mut hi) = (0, times.len() - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if times[mid] < t { lo = mid + 1 } else { hi = mid }
    }
    if lo > 0 && t - times[lo - 1] < times[lo] - t { lo - 1 } else { lo }
}

// ------------------------------------------------------------------ the tracker

struct Tracklet {
    frames: usize,
    bbox: [f64; 4],
    last: usize,
    embeds: Vec<Vec<f32>>,
    person: Option<usize>,
    /// its person was taken from frames looked at before this pass
    known: bool,
    /// its faces: (frame, index in that frame's faces)
    recs: Vec<(usize, usize)>,
}

/// Links each frame's faces to the tracklets of the frames before, one walk over frames in order.
struct Tracker {
    tracklets: Vec<Tracklet>,
    active: Vec<usize>,
}

impl Tracker {
    fn new() -> Tracker {
        Tracker { tracklets: Vec::new(), active: Vec::new() }
    }

    /// The tracklet each of frame i's faces goes on: the most overlapping first.
    fn push(&mut self, i: usize, boxes: &[[f64; 4]]) -> Vec<usize> {
        let tracklets = &self.tracklets;
        self.active.retain(|&t| i - tracklets[t].last.min(i) <= GAP);
        let mut pairs = Vec::new();
        for (fi, b) in boxes.iter().enumerate() {
            for &t in &self.active {
                let o = iou(b, &self.tracklets[t].bbox);
                if o >= LINK_IOU {
                    pairs.push((o, fi, t));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut of = vec![usize::MAX; boxes.len()];
        let mut taken = HashSet::new();
        for (_, fi, t) in pairs {
            if of[fi] != usize::MAX || taken.contains(&t) {
                continue;
            }
            of[fi] = t;
            taken.insert(t);
        }
        for (fi, b) in boxes.iter().enumerate() {
            if of[fi] == usize::MAX {
                self.tracklets.push(Tracklet { frames: 0, bbox: *b, last: i, embeds: Vec::new(), person: None, known: false, recs: Vec::new() });
                of[fi] = self.tracklets.len() - 1;
                self.active.push(of[fi]);
            }
            let t = &mut self.tracklets[of[fi]];
            t.frames += 1;
            t.bbox = *b;
            t.last = i;
        }
        of
    }

    /// whether the tracklet wants an embedding of its face in the frame just pushed
    fn wants_embedding(&self, t: usize) -> bool {
        let t = &self.tracklets[t];
        t.embeds.len() < EMBEDS && (t.frames - 1) % EMBED_EVERY == 0
    }
}

// ------------------------------------------------------------------ the scan

/// Embeds faces of the frame at hand: their normalized ArcFace embeddings, in order.
pub type Embed<'a> = dyn FnMut(&[Face]) -> Result<Vec<Vec<f32>>, String> + 'a;

pub struct Scan {
    pub a: Analysis,
    /// person -> their mean embedding (normalized), made when wanted
    centroids: Vec<Option<Vec<f32>>>,
    /// who a person is (root) -> the frames they're in
    seen: HashMap<usize, HashSet<usize>>,
    /// person -> how good the view their thumbnail is of
    best: HashMap<usize, f64>,
    pass: Tracker,
    regroup_due: bool,
}

impl Scan {
    /// A new analysis of a video whose frames are at `times`, with keyframes `keys`.
    pub fn new(times: Vec<f64>, keys: Vec<usize>) -> Scan {
        let n = times.len();
        Scan::from(Analysis { times, keys, state: vec![NOT_LOOKED; n], faces: vec![None; n], people: Vec::new() })
    }

    pub fn from(a: Analysis) -> Scan {
        let mut s = Scan { centroids: Vec::new(), seen: HashMap::new(), best: HashMap::new(), pass: Tracker::new(), regroup_due: false, a };
        for i in 0..s.a.faces.len() {
            let people: Vec<usize> = s.a.faces[i].iter().flatten().map(|r| r.person).collect();
            for p in people {
                s.seen_in(p).insert(i);
            }
        }
        s
    }

    fn new_person(&mut self) -> usize {
        self.a.people.push(PersonSum::default());
        self.a.people.len() - 1
    }

    /// who person p is: the person they were found to be one with, or themselves
    pub fn root(&self, mut p: usize) -> usize {
        while let Some(q) = self.a.people.get(p).and_then(|x| x.into) {
            p = q;
        }
        p
    }

    fn seen_in(&mut self, p: usize) -> &mut HashSet<usize> {
        let r = self.root(p);
        self.seen.entry(r).or_default()
    }

    fn centroid(&mut self, p: usize) -> Option<Vec<f32>> {
        if self.centroids.len() <= p {
            self.centroids.resize(p + 1, None);
        }
        if self.centroids[p].is_none() {
            self.centroids[p] = self.a.people[p].sum.as_ref().map(|s| normalize(s));
        }
        self.centroids[p].clone()
    }

    /// adds embedding e to who person p is (or takes it away, sign -1)
    fn add_embed(&mut self, p: usize, e: &[f32], sign: f32) {
        let p = self.root(p);
        let x = &mut self.a.people[p];
        let sum = x.sum.get_or_insert_with(|| vec![0.0; e.len()]);
        for (s, v) in sum.iter_mut().zip(e) {
            *s += sign * v;
        }
        x.n += sign as i32;
        if x.n <= 0 {
            x.sum = None;
            x.n = 0;
        }
        if let Some(c) = self.centroids.get_mut(p) {
            *c = None;
        }
        self.regroup_due = true;
    }

    /// the person embedding e is of: the most like it of those not in `here`, or a new one
    fn matching(&mut self, e: &[f32], here: &HashSet<usize>) -> usize {
        let (mut best, mut who) = (SAME_PERSON, None);
        for p in 0..self.a.people.len() {
            let x = &self.a.people[p];
            if x.sum.is_none() || x.into.is_some() || here.contains(&p) {
                continue;
            }
            let s = dot(&self.centroid(p).unwrap(), e);
            if s >= best {
                best = s;
                who = Some(p);
            }
        }
        who.unwrap_or_else(|| self.new_person())
    }

    /// Tracklet t is someone else: person p. Its faces so far go to them.
    fn relabel(&mut self, t: usize, p: usize) {
        let Some(was) = self.pass.tracklets[t].person else { return };
        if was == p {
            return;
        }
        for e in self.pass.tracklets[t].embeds.clone() {
            self.add_embed(was, &e, -1.0);
            self.add_embed(p, &e, 1.0);
        }
        for (i, j) in self.pass.tracklets[t].recs.clone() {
            let old = self.a.faces[i].as_ref().unwrap()[j].person;
            self.seen_in(old).remove(&i);
            self.seen_in(p).insert(i);
            self.a.faces[i].as_mut().unwrap()[j].person = p;
        }
        self.pass.tracklets[t].person = Some(p);
    }

    /// Finds the people found as two who are one: while the two whose mean embeddings are most
    /// alike reach SAME_PERSON and are never on screen together, the later is found to be the
    /// earlier.
    pub fn regroup(&mut self) {
        if !self.regroup_due {
            return;
        }
        self.regroup_due = false;
        loop {
            let live: Vec<usize> = (0..self.a.people.len()).filter(|&p| self.a.people[p].sum.is_some() && self.a.people[p].into.is_none()).collect();
            let c: Vec<Vec<f32>> = live.iter().map(|&p| self.centroid(p).unwrap()).collect();
            let mut pairs = Vec::new();
            for i in 0..live.len() {
                for j in i + 1..live.len() {
                    let s = dot(&c[i], &c[j]);
                    if s >= SAME_PERSON {
                        pairs.push((s, live[i], live[j]));
                    }
                }
            }
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            let empty = HashSet::new();
            let pair = pairs.into_iter().find(|&(_, p, q)| {
                let (a, b) = (self.seen.get(&p).unwrap_or(&empty), self.seen.get(&q).unwrap_or(&empty));
                let (small, large) = if a.len() < b.len() { (a, b) } else { (b, a) };
                !small.iter().any(|f| large.contains(f))
            });
            let Some((_, p, q)) = pair else { return };
            let y = self.a.people[q].clone();
            let x = &mut self.a.people[p];
            if let (Some(xs), Some(ys)) = (x.sum.as_mut(), y.sum.as_ref()) {
                for (a, b) in xs.iter_mut().zip(ys) {
                    *a += b;
                }
            }
            x.n += y.n;
            self.a.people[q].into = Some(p);
            let moved = self.seen.remove(&q).unwrap_or_default();
            self.seen.entry(p).or_default().extend(moved);
            for k in [p, q] {
                if let Some(c) = self.centroids.get_mut(k) {
                    *c = None;
                }
            }
        }
    }

    /// whether person p's face f is a clearly better view than any before
    fn better_view(&mut self, p: usize, f: &Face) -> Option<Thumb> {
        let q = quality(f);
        let b = self.best.get(&p).copied().unwrap_or(0.0);
        if b > 0.0 && q <= b * BETTER_VIEW {
            return None;
        }
        self.best.insert(p, q);
        Some(Thumb { person: p, bbox: f.bbox })
    }

    // ---------------------------------------------------------- the skim

    /// Keyframe k, not looked at yet: its faces (`found`) and who they are.
    pub fn glance(&mut self, k: usize, found: Vec<Face>, embed: &mut Embed) -> Result<Vec<Thumb>, String> {
        if self.a.state[k] != NOT_LOOKED {
            return Ok(Vec::new());
        }
        let embeds = if found.is_empty() { Vec::new() } else { embed(&found)? };
        let mut here = HashSet::new();
        let (mut recs, mut thumbs) = (Vec::new(), Vec::new());
        for (f, e) in found.iter().zip(&embeds) {
            let p = self.matching(e, &here);
            self.add_embed(p, e, 1.0);
            here.insert(p);
            thumbs.extend(self.better_view(p, f));
            recs.push(Rec { bbox: f.bbox, score: f.score, kps: f.kps, person: p, detail: None });
            self.seen_in(p).insert(k);
        }
        self.a.faces[k] = Some(recs);
        self.a.state[k] = SKIMMED;
        Ok(thumbs)
    }

    /// The keyframes of frames [from, to) the skim looks at, in turn: the keyframes nearest the points
    /// of grids `step * 2^m` seconds apart, coarse to fine, the finer ones near frame `at` first.
    pub fn skim_order(&self, at: usize, from: usize, to: usize, step: f64) -> Vec<usize> {
        const BIAS: f64 = 30.0;
        let t = &self.a.times;
        let keys: Vec<usize> = self.a.keys.iter().copied().filter(|&k| k >= from && k < to).collect();
        if keys.is_empty() || to <= from {
            return Vec::new();
        }
        let (t0, t1, ta) = (t[from], t[to - 1], t[at.min(t.len() - 1)]);
        let mut grids = Vec::new();
        let mut s = step;
        while s < (t1 - t0) * 2.0 {
            grids.push(s);
            s *= 2.0;
        }
        grids.push(s);
        // (each point's turn, and the time it's at)
        let mut turns = Vec::new();
        for &g in &grids {
            let mut x = t0;
            while x <= t1 + 1e-9 {
                turns.push((((x - ta).abs() + BIAS) / g, x));
                x += g;
            }
        }
        turns.sort_by(|a, b| a.0.total_cmp(&b.0));
        let kt: Vec<f64> = keys.iter().map(|&k| t[k]).collect();
        let mut taken = HashSet::new();
        let mut out = Vec::new();
        for (_, x) in turns {
            let k = keys[nearest(&kt, x)];
            if taken.insert(k) {
                out.push(k);
            }
        }
        out
    }

    // ---------------------------------------------------------- a pass

    /// Starts a walk over frames in order from frame `lo`: the faces of the frames looked at just
    /// before it go on into it.
    pub fn begin_pass(&mut self, lo: usize) {
        self.pass = Tracker::new();
        for j in lo.saturating_sub(GAP)..lo.min(self.a.state.len()) {
            if self.a.state[j] == LOOKED {
                self.known(j);
            }
        }
    }

    /// Ends the walk at frame `hi`: its faces go on into the frames looked at just after.
    pub fn end_pass(&mut self, hi: usize) {
        let n = self.a.state.len();
        let mut j = hi;
        while j < n.min(hi + GAP) && self.a.state[j] == LOOKED {
            self.known(j);
            j += 1;
        }
        self.regroup();
    }

    /// Frame i, looked at by a pass before: its faces go on into the tracklets, and a tracklet of
    /// this walk they continue is their person.
    pub fn known(&mut self, i: usize) {
        let recs = self.a.faces[i].clone().unwrap_or_default();
        let boxes: Vec<[f64; 4]> = recs.iter().map(|r| r.bbox).collect();
        let of = self.pass.push(i, &boxes);
        for (j, &t) in of.iter().enumerate() {
            let p = recs[j].person;
            self.pass.tracklets[t].recs.push((i, j));
            let tr = &self.pass.tracklets[t];
            match tr.person {
                None => {
                    let tr = &mut self.pass.tracklets[t];
                    tr.person = Some(p);
                    tr.known = true;
                }
                Some(q) => {
                    if self.root(q) == self.root(p) || tr.known {
                        continue;
                    }
                    // (unless that'd put them on screen twice)
                    let rp = self.root(p);
                    let twice = tr.recs.iter().any(|&(f, k)| {
                        self.a.faces[f].as_ref().is_some_and(|fs| fs.iter().enumerate().any(|(x, r)| x != k && self.root(r.person) == rp))
                    });
                    if !twice {
                        self.relabel(t, p);
                    }
                    self.pass.tracklets[t].known = true;
                }
            }
        }
    }

    /// Frame i, not looked at by a pass yet: its faces (`found`, unless the skim found them on
    /// this keyframe already) linked to the frames before, and who they are.
    pub fn look(&mut self, i: usize, found: Option<Vec<Face>>, embed: &mut Embed) -> Result<Vec<Thumb>, String> {
        if self.a.state[i] == LOOKED {
            self.known(i);
            return Ok(Vec::new());
        }
        let loose = if self.a.state[i] == SKIMMED { self.a.faces[i].clone() } else { None };
        let faces: Vec<Face> = match &loose {
            Some(recs) => recs.iter().map(Rec::face).collect(),
            None => found.ok_or("the frame's faces weren't found")?,
        };
        let boxes: Vec<[f64; 4]> = faces.iter().map(|f| f.bbox).collect();
        let of = self.pass.push(i, &boxes);
        for &t in &of {
            if let Some(p) = self.pass.tracklets[t].person {
                self.pass.tracklets[t].person = Some(self.root(p));
            }
        }
        // the embeddings the tracklets want, in one go
        let want: Vec<usize> = if loose.is_none() { (0..faces.len()).filter(|&j| self.pass.wants_embedding(of[j])).collect() } else { Vec::new() };
        if !want.is_empty() {
            let es = embed(&want.iter().map(|&j| faces[j].clone()).collect::<Vec<_>>())?;
            for (&j, e) in want.iter().zip(es) {
                let t = of[j];
                if let Some(p) = self.pass.tracklets[t].person {
                    self.add_embed(p, &e, 1.0);
                }
                self.pass.tracklets[t].embeds.push(e);
            }
        }
        // the tracklets going on first: their people are taken here
        let mut order: Vec<usize> = (0..faces.len()).collect();
        order.sort_by_key(|&j| self.pass.tracklets[of[j]].person.is_none());
        let mut here = HashSet::new();
        let mut person = vec![0; faces.len()];
        let mut thumbs = Vec::new();
        for j in order {
            let t = of[j];
            // a new tracklet is who the skim took it for, or who it looks like; one going on is
            // who it was (a tracklet that'd put them on screen twice is someone else)
            let cur = self.pass.tracklets[t].person;
            if cur.is_none() || here.contains(&cur.unwrap()) {
                let was = loose.as_ref().map(|r| self.root(r[j].person));
                let last = self.pass.tracklets[t].embeds.last().cloned();
                let p = match (was, last) {
                    (Some(w), _) if !here.contains(&w) => w,
                    (_, Some(e)) => self.matching(&e, &here),
                    _ => self.new_person(),
                };
                if cur.is_none() {
                    self.pass.tracklets[t].person = Some(p);
                    for e in self.pass.tracklets[t].embeds.clone() {
                        self.add_embed(p, &e, 1.0);
                    }
                } else {
                    self.relabel(t, p);
                }
            }
            let p = self.pass.tracklets[t].person.unwrap();
            here.insert(p);
            person[j] = p;
            thumbs.extend(self.better_view(p, &faces[j]));
        }
        let recs: Vec<Rec> = faces.iter().zip(&person).map(|(f, &p)| Rec { bbox: f.bbox, score: f.score, kps: f.kps, person: p, detail: None }).collect();
        for (j, r) in recs.iter().enumerate() {
            self.pass.tracklets[of[j]].recs.push((i, j));
            let p = r.person;
            self.seen_in(p).insert(i);
        }
        self.a.faces[i] = Some(recs);
        self.a.state[i] = LOOKED;
        Ok(thumbs)
    }

    // ---------------------------------------------------------- what's been found

    /// How far apart the skim's keyframes are, for bridging a person's frames between them: one
    /// and a half the median keyframe spacing, 2 to 15 s.
    pub fn loose_span(&self) -> f64 {
        let t = &self.a.times;
        let mut d: Vec<f64> = self.a.keys.windows(2).map(|w| t[w[1]] - t[w[0]]).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        (1.5 * d.get(d.len() / 2).copied().unwrap_or(1.0)).clamp(2.0, 15.0)
    }

    /// Each person's stretches on screen, from the frames looked at so far. A run bridges up to GAP
    /// frames looked at without the face, and frames not looked at for up to `loose_span` seconds.
    pub fn stretches(&self) -> Vec<Stretches> {
        struct Acc {
            frames: usize,
            loose: bool,
            runs: Vec<(usize, usize)>,
            last: usize,
            k: isize,
        }
        let span = self.loose_span();
        let mut out: Vec<(usize, Acc)> = Vec::new();
        let mut index: HashMap<usize, usize> = HashMap::new();
        let mut k: isize = 0; // frames looked at before frame i
        for i in 0..self.a.faces.len() {
            let Some(fs) = &self.a.faces[i] else { continue };
            for f in fs {
                let at = *index.entry(f.person).or_insert_with(|| {
                    out.push((f.person, Acc { frames: 0, loose: false, runs: Vec::new(), last: 0, k: -1 }));
                    out.len() - 1
                });
                let p = &mut out[at].1;
                p.frames += 1;
                if self.a.state[i] == SKIMMED {
                    p.loose = true;
                }
                let looked = k - p.k - 1;
                let skipped = i as isize - p.last as isize - 1 - looked;
                match p.runs.last_mut() {
                    Some(r) if looked <= GAP as isize && (skipped <= 0 || self.a.times[i] - self.a.times[p.last] <= span) => r.1 = i + 1,
                    _ => p.runs.push((i, i + 1)),
                }
                p.last = i;
                p.k = k;
            }
            k += 1;
        }
        out.into_iter().map(|(person, p)| Stretches { person, root: self.root(person), frames: p.frames, loose: p.loose, runs: p.runs }).collect()
    }

    /// Frame i's faces, each with who they are (root) and their own detail there.
    pub fn faces(&self, i: usize) -> Vec<Rec> {
        self.a.faces.get(i).cloned().flatten().unwrap_or_default().into_iter().map(|mut r| {
            r.detail = self.detail_at(i, r.person);
            r.person = self.root(r.person);
            r
        }).collect()
    }

    /// Frame i's faces get their own detail, in the order they were found.
    pub fn set_details(&mut self, i: usize, details: &[f64]) {
        if let Some(recs) = self.a.faces[i].as_mut() {
            for (r, d) in recs.iter_mut().zip(details) {
                r.detail = Some(*d);
            }
        }
    }

    /// A person's own detail at frame i: the median over the frames within DETAIL_SPAN where they
    /// have one, so the enhancer's restore doesn't flicker with noise in the measure.
    fn detail_at(&self, i: usize, person: usize) -> Option<f64> {
        let hi = (i + DETAIL_SPAN).min(self.a.faces.len().saturating_sub(1));
        let mut v: Vec<f64> = (i.saturating_sub(DETAIL_SPAN)..=hi)
            .flat_map(|j| self.a.faces[j].iter().flatten())
            .filter(|r| r.person == person)
            .filter_map(|r| r.detail)
            .collect();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    }

    /// how much of frames [lo, hi) a pass has looked at, 0 to 1
    pub fn coverage(&self, lo: usize, hi: usize) -> f64 {
        let hi = hi.min(self.a.state.len());
        if hi <= lo {
            return 1.0;
        }
        self.a.state[lo..hi].iter().filter(|&&s| s == LOOKED).count() as f64 / (hi - lo) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_faces_detail_is_the_median_around_its_frame() {
        let mut s = Scan::new((0..40).map(|i| i as f64 / 30.0).collect(), Vec::new());
        let (a, b) = (s.new_person(), s.new_person());
        let rec = |person, detail| Rec { bbox: [0.0; 4], score: 1.0, kps: [[0.0; 2]; 5], person, detail };
        for i in 0..40 {
            // (one frame far off, which the median ignores)
            let d = if i == 20 { 99.0 } else { 10.0 + (i % 3) as f64 };
            s.a.faces[i] = Some(vec![rec(a, Some(d)), rec(b, Some(50.0))]);
        }
        s.a.faces[21].as_mut().unwrap()[0].detail = None;
        let got = s.faces(20);
        assert_eq!(got[0].detail, Some(11.0));
        assert_eq!(got[1].detail, Some(50.0));
        // (the first frame has fewer neighbours, and nobody measured means no value)
        assert!(s.faces(0)[0].detail.is_some());
        s.a.faces[39] = Some(vec![rec(s.new_person(), None)]);
        assert_eq!(s.faces(39)[0].detail, None);
    }

    fn face(x: f64, y: f64) -> Face {
        Face { bbox: [x, y, x + 100.0, y + 100.0], score: 0.9, kps: [[x + 30.0, y + 40.0], [x + 70.0, y + 40.0], [x + 50.0, y + 60.0], [x + 35.0, y + 80.0], [x + 65.0, y + 80.0]] }
    }

    /// a unit embedding along axis `who`, a little off it
    fn embedding(who: usize, wobble: f32) -> Vec<f32> {
        let mut e = vec![0.0; 16];
        e[who] = 1.0;
        e[(who + 1) % 16] = wobble;
        normalize(&e)
    }

    fn video(n: usize, key_every: usize) -> Scan {
        let times: Vec<f64> = (0..n).map(|i| i as f64 / 30.0).collect();
        let keys: Vec<usize> = (0..n).step_by(key_every).collect();
        Scan::new(times, keys)
    }

    /// Embeds by where the face is: faces left of x = 500 are person 0's, right of it person 1's.
    fn by_place(faces: &[Face]) -> Result<Vec<Vec<f32>>, String> {
        Ok(faces.iter().map(|f| embedding(if f.bbox[0] < 500.0 { 0 } else { 1 }, 0.05)).collect())
    }

    /// A 9-minute video, fully looked at, two people on screen: the people list the app takes
    /// in every 250 ms while the scan runs, and the save, stay quick.
    #[test]
    fn a_long_video_stays_quick() {
        let n = 9 * 60 * 30;
        let mut s = video(n, 60);
        let t = std::time::Instant::now();
        s.begin_pass(0);
        for i in 0..n {
            let x = (i % 300) as f64;
            s.look(i, Some(vec![face(100.0 + x, 100.0), face(700.0 + x, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(n);
        let pass = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        s.regroup();
        let st = s.stretches();
        let people = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let saved = serde_json::to_vec(&s.a).unwrap();
        let save = t.elapsed().as_secs_f64() * 1e3;
        println!("{n} frames: pass {pass:.0} ms ({:.3} ms a frame), people {people:.1} ms, save {save:.0} ms, {} MB", pass / n as f64, saved.len() / 1_000_000);
        assert_eq!(st.iter().filter(|p| p.frames > 1000).count(), 2);
        assert!(people < 50.0, "the people list took {people:.1} ms");
    }

    #[test]
    fn a_face_moving_slowly_stays_one_person() {
        let mut s = video(60, 30);
        s.begin_pass(0);
        for i in 0..60 {
            s.look(i, Some(vec![face(100.0 + i as f64, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(60);
        let st = s.stretches();
        assert_eq!(st.len(), 1);
        assert_eq!(st[0].runs, vec![(0, 60)]);
        assert_eq!(st[0].frames, 60);
    }

    #[test]
    fn two_faces_on_screen_together_are_two_people() {
        let mut s = video(30, 30);
        s.begin_pass(0);
        for i in 0..30 {
            s.look(i, Some(vec![face(100.0, 100.0), face(700.0, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(30);
        let roots: HashSet<usize> = s.faces(10).iter().map(|r| r.person).collect();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn someone_coming_back_is_found_to_be_the_same_person() {
        // on screen, gone for longer than GAP, back somewhere else: a new tracklet, the same embedding
        let mut s = video(90, 30);
        let mut same = |faces: &[Face]| -> Result<Vec<Vec<f32>>, String> { Ok(faces.iter().map(|_| embedding(3, 0.02)).collect()) };
        s.begin_pass(0);
        for i in 0..90 {
            let found = if (30..60).contains(&i) { vec![] } else if i < 30 { vec![face(100.0, 100.0)] } else { vec![face(600.0, 300.0)] };
            s.look(i, Some(found), &mut same).unwrap();
        }
        s.end_pass(90);
        assert_eq!(s.faces(10)[0].person, s.faces(80)[0].person);
        let st: Vec<Stretches> = s.stretches().into_iter().filter(|x| x.root == s.faces(10)[0].person).collect();
        let runs: Vec<(usize, usize)> = st.iter().flat_map(|x| x.runs.clone()).collect();
        assert!(runs.contains(&(0, 30)) && runs.contains(&(60, 90)), "{runs:?}");
    }

    #[test]
    fn the_skim_names_people_the_pass_then_follows() {
        let mut s = video(90, 30);
        for k in [0, 30, 60] {
            s.glance(k, vec![face(100.0, 100.0)], &mut by_place).unwrap();
        }
        assert_eq!(s.a.state[30], SKIMMED);
        let skimmed = s.faces(30)[0].person;
        // a pass over the middle: its faces are the skim's person
        s.begin_pass(25);
        for i in 25..40 {
            s.look(i, Some(vec![face(100.0, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(40);
        assert_eq!(s.faces(26)[0].person, skimmed);
        assert_eq!(s.a.state[30], LOOKED);
        // the skimmed keyframes bridge to one stretch over the whole video
        let st = s.stretches();
        assert_eq!(st.iter().filter(|x| x.root == skimmed).flat_map(|x| x.runs.clone()).count(), 1);
    }

    #[test]
    fn a_second_pass_goes_on_from_the_first() {
        let mut s = video(60, 30);
        s.begin_pass(0);
        for i in 0..30 {
            s.look(i, Some(vec![face(100.0, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(30);
        s.begin_pass(30);
        for i in 30..60 {
            s.look(i, Some(vec![face(102.0, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(60);
        assert_eq!(s.faces(5)[0].person, s.faces(55)[0].person);
        assert_eq!(s.coverage(0, 60), 1.0);
    }

    #[test]
    fn the_analysis_saves_and_comes_back() {
        let mut s = video(30, 30);
        s.begin_pass(0);
        for i in 0..30 {
            s.look(i, Some(vec![face(100.0, 100.0)]), &mut by_place).unwrap();
        }
        s.end_pass(30);
        let json = serde_json::to_vec(&s.a).unwrap();
        let back = Scan::from(serde_json::from_slice(&json).unwrap());
        assert_eq!(back.stretches(), s.stretches());
    }

    #[test]
    fn the_skim_goes_coarse_to_fine_near_the_playhead_first() {
        let s = video(30 * 60, 30);
        // a keyframe a second, skimmed every 2 s at the finest
        let order = s.skim_order(900, 0, 1800, 2.0);
        assert_eq!(order.len(), 30);
        assert!(order.iter().take(3).any(|&k| (k as isize - 900).abs() <= 60), "{:?}", &order[..5]);
        let unique: HashSet<usize> = order.iter().copied().collect();
        assert_eq!(unique.len(), order.len());
    }
}
