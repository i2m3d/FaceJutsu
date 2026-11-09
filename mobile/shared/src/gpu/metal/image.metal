
// Image kernels: resampling 8-bit images into network inputs, pasting swapped faces back,
// copying frames; and a matrix-vector product for the small fully connected layers. 8-bit
// images are RGB or BGRA (as video frames come), rows `stride` bytes apart.

struct ResampleParams {
    float m[6];             // output pixel -> source pixel, pixel centers at integers
    uint h, w;              // the source
    uint oh, ow;            // the output
    uint n;                 // n x n samples per output pixel
    uint valid_w, valid_h;  // outside this the output counts as black
    float mul, add;         // the output is value * mul + add
    uint stride, bgra;      // the source's row bytes, and whether it's BGRA (else RGB)
    uint k, si, sj;         // Detail: the output is sub-crop (si, sj) of a crop k times its size,
                            // output pixel (x, y) its pixel (x k + si, y k + sj); the map is the crop's
};

static float3 rgb8_at(device const uchar *src, uint stride, uint bgra, uint y, uint x) {
    if (bgra) {
        // rows of BGRA pixels are 4-byte aligned: one load
        const uchar4 q = *(device const uchar4 *)(src + y * stride + x * 4);
        return float3(q.z, q.y, q.x);
    }
    device const uchar *q = src + y * stride + x * 3;
    return float3(q[0], q[1], q[2]);
}

// Bilinear, edges replicated.
static float3 rgb8_bilinear(device const uchar *src, constant ResampleParams &p, float sx, float sy) {
    sx = clamp(sx, 0.0f, float(p.w - 1));
    sy = clamp(sy, 0.0f, float(p.h - 1));
    const float fx = floor(sx), fy = floor(sy);
    const float tx = sx - fx, ty = sy - fy;
    const uint x0 = uint(fx), y0 = uint(fy), x1 = min(x0 + 1, p.w - 1), y1 = min(y0 + 1, p.h - 1);
    const float3 top = rgb8_at(src, p.stride, p.bgra, y0, x0) * (1 - tx) + rgb8_at(src, p.stride, p.bgra, y0, x1) * tx;
    const float3 bottom = rgb8_at(src, p.stride, p.bgra, y1, x0) * (1 - tx) + rgb8_at(src, p.stride, p.bgra, y1, x1) * tx;
    return top * (1 - ty) + bottom * ty;
}

// Affine resampling of an 8-bit RGB image into f16 [oh, ow, 3]; when the map shrinks, each
// output pixel averages n x n bilinear samples. One thread per output pixel.
kernel void resample_rgb8(device const uchar *src [[buffer(0)]],
                          device half *dst [[buffer(1)]],
                          constant ResampleParams &p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.oh * p.ow)
        return;
    const uint x = (i % p.ow) * p.k + p.si, y = (i / p.ow) * p.k + p.sj;
    float3 acc = 0;
    if (x < p.valid_w && y < p.valid_h) {
        for (uint j = 0; j < p.n; j++)
            for (uint k = 0; k < p.n; k++) {
                const float ox = float(x) + (float(k) + 0.5f) / float(p.n) - 0.5f;
                const float oy = float(y) + (float(j) + 0.5f) / float(p.n) - 0.5f;
                acc += rgb8_bilinear(src, p, p.m[0] * ox + p.m[1] * oy + p.m[2], p.m[3] * ox + p.m[4] * oy + p.m[5]);
            }
        acc /= float(p.n * p.n);
    }
    const float3 v = acc * p.mul + p.add;
    dst[i * 3] = half(v.x);
    dst[i * 3 + 1] = half(v.y);
    dst[i * 3 + 2] = half(v.z);
}

struct MatvecParams {
    uint rows, cols;
    float slope;  // leaky ReLU after (1: none)
};

// out[r] = bias[r] + W[r] . x, W f16 [rows, cols]: one simdgroup per row, 8 rows per threadgroup.
kernel void matvec_f32(device const half *wt [[buffer(0)]], device const float *x [[buffer(1)]],
                       device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],
                       constant MatvecParams &p [[buffer(4)]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint tg [[threadgroup_position_in_grid]]) {
    const uint row = tg * 8 + sg;
    if (row >= p.rows)
        return;
    float s = 0;
    for (uint i = lane; i < p.cols; i += 32)
        s += float(wt[row * p.cols + i]) * x[i];
    s = simd_sum(s) + bias[row];
    if (lane == 0)
        out[row] = s < 0 ? s * p.slope : s;
}

kernel void matvec_f16(device const half *wt [[buffer(0)]], device const half *x [[buffer(1)]],
                       device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],
                       constant MatvecParams &p [[buffer(4)]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint tg [[threadgroup_position_in_grid]]) {
    const uint row = tg * 8 + sg;
    if (row >= p.rows)
        return;
    float s = 0;
    for (uint i = lane; i < p.cols; i += 32)
        s += float(wt[row * p.cols + i]) * float(x[i]);
    s = simd_sum(s) + bias[row];
    if (lane == 0)
        out[row] = s < 0 ? s * p.slope : s;
}

struct PasteParams {
    float m[6];   // frame pixel -> crop pixel
    uint x1, y1;  // the box of frame pixels the crop covers
    uint bw, bh;
    uint stride, bgra;  // the frame's row bytes, and whether it's BGRA (else RGB)
    uint s;       // the crop's size
    uint k;       // Detail: the crop is k x k sub-crops of s / k px, pixel (y k + j, x k + i) in
                  // sub-crop j k + i
    float alpha;  // the mask times this
    uint add;     // the crop is a change to the frame (0..1 units, not clamped), added to it
};

static float crop_at(device const float *crop, constant PasteParams &p, uint y, uint x, uint ch) {
    const uint sub = p.s / p.k, s = (y % p.k) * p.k + x % p.k;
    const float v = crop[((s * sub + y / p.k) * sub + x / p.k) * 3 + ch];
    return (p.add ? v : clamp(v, 0.0f, 1.0f)) * 255.0f;
}

// Blends a swapped crop (f32 [s, s, 3] in 0..1, clamped here) into an 8-bit frame, in place,
// through the inverse alignment: bilinear crop lookups with edges replicated, the edge mask (the
// outer product of `profile`) zero outside the crop, the result rounded. With `add` the crop is a
// change, added. One thread per frame pixel in the box.
kernel void paste(device uchar *frame [[buffer(0)]],
                  device const float *crop [[buffer(1)]],
                  device const float *profile [[buffer(2)]],
                  constant PasteParams &p [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= p.bw * p.bh)
        return;
    const uint px = p.x1 + i % p.bw, py = p.y1 + i / p.bw;
    const float fx = float(px), fy = float(py);
    const float sx = p.m[0] * fx + p.m[1] * fy + p.m[2], sy = p.m[3] * fx + p.m[4] * fy + p.m[5];
    const float last = float(p.s - 1);
    // the mask: taps outside the crop are zero
    const float mx0 = floor(sx), my0 = floor(sy), mtx = sx - mx0, mty = sy - my0;
    float a = 0;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
            const float x = mx0 + float(dx), y = my0 + float(dy);
            if (x < 0 || y < 0 || x > last || y > last)
                continue;
            const float wgt = (dx ? mtx : 1 - mtx) * (dy ? mty : 1 - mty);
            a += wgt * profile[uint(y)] * profile[uint(x)];
        }
    a = clamp(a, 0.0f, 1.0f) * p.alpha;
    if (a == 0)
        return;
    // the crop: edges replicated
    const float cx = clamp(sx, 0.0f, last), cy = clamp(sy, 0.0f, last);
    const float x0 = floor(cx), y0 = floor(cy), tx = cx - x0, ty = cy - y0;
    const uint ix0 = uint(x0), iy0 = uint(y0), ix1 = min(ix0 + 1, p.s - 1), iy1 = min(iy0 + 1, p.s - 1);
    float3 c;
    for (uint ch = 0; ch < 3; ch++) {
        const float top = crop_at(crop, p, iy0, ix0, ch) * (1 - tx) + crop_at(crop, p, iy0, ix1, ch) * tx;
        const float bottom = crop_at(crop, p, iy1, ix0, ch) * (1 - tx) + crop_at(crop, p, iy1, ix1, ch) * tx;
        c[ch] = top * (1 - ty) + bottom * ty;
    }
    device uchar *q = frame + py * p.stride + px * (p.bgra ? 4 : 3);
    const float3 old = p.bgra ? float3(q[2], q[1], q[0]) : float3(q[0], q[1], q[2]);
    const uchar3 v = uchar3(clamp(rint(p.add ? old + c * a : old * (1 - a) + c * a), 0.0f, 255.0f));
    if (p.bgra) {
        q[0] = v.z;
        q[1] = v.y;
        q[2] = v.x;
    } else {
        q[0] = v.x;
        q[1] = v.y;
        q[2] = v.z;
    }
}

struct CopyParams {
    uint rows, row_bytes;      // what to copy: rows of row_bytes
    uint src_stride, dst_stride;
};

// Copies rows of bytes between images with different row strides; with `copy_rows32`, in 4-byte
// words (row bytes and strides a multiple of 4). One thread per byte or word.
kernel void copy_rows(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                      constant CopyParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.row_bytes)
        return;
    const uint r = i / p.row_bytes, c = i % p.row_bytes;
    dst[r * p.dst_stride + c] = src[r * p.src_stride + c];
}

kernel void copy_rows32(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                        constant CopyParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint words = p.row_bytes / 4;
    if (i >= p.rows * words)
        return;
    const uint r = i / words, c = i % words;
    dst[r * (p.dst_stride / 4) + c] = src[r * (p.src_stride / 4) + c];
}

struct HostParams {
    uint h, w, sw, c;
};

// An activation (rows stored `sw` wide) to compact f32, for reading on the CPU.
kernel void act_to_f32(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],
                       constant HostParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w * p.c)
        return;
    const uint ch = i % p.c, px = (i / p.c) % p.w, py = i / (p.c * p.w);
    dst[i] = float(src[(py * p.sw + px) * p.c + ch]);
}
