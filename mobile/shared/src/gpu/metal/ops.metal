// The networks' kernels besides the convs. Activations are f16 NHWC (batch 1); a row may be
// stored wider than the image (`sw`), as the convs' wide tiles leave it.

#include <metal_stdlib>
using namespace metal;

struct PadParams {
    uint h, w, sw;  // the source and its stored row width
    uint c, cp;     // its channels, and the padded copy's (the rest are zero)
    uint lo, hi;    // the border before and after, on each axis
    uint reflect;
    uint scaled;    // multiply channel ch by scale[ch] (a modulated conv's style)
};

// Copies an image into a buffer with a border: zeros, or reflected without repeating the edge.
// One thread per destination value.
kernel void pad(device const half *src [[buffer(0)]],
                device half *dst [[buffer(1)]],
                device const float *scale [[buffer(2)]],
                constant PadParams &p [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    const uint pw = p.w + p.lo + p.hi, ph = p.h + p.lo + p.hi;
    if (i >= ph * pw * p.cp)
        return;
    const uint ch = i % p.cp, px = (i / p.cp) % pw, py = i / (p.cp * pw);
    int sy = int(py) - int(p.lo), sx = int(px) - int(p.lo);
    half v = 0;
    if (ch < p.c) {
        if (p.reflect) {
            sy = abs(sy);
            sx = abs(sx);
            sy = sy >= int(p.h) ? 2 * (int(p.h) - 1) - sy : sy;
            sx = sx >= int(p.w) ? 2 * (int(p.w) - 1) - sx : sx;
            v = src[(uint(sy) * p.sw + uint(sx)) * p.c + ch];
        } else if (sy >= 0 && sx >= 0 && sy < int(p.h) && sx < int(p.w)) {
            v = src[(uint(sy) * p.sw + uint(sx)) * p.c + ch];
        }
        if (p.scaled)
            v = half(float(v) * scale[ch]);
    }
    dst[i] = v;
}

struct NormParams {
    uint h, w, sw, c;
    uint relu;      // apply ReLU after the style
    uint residual;  // add `res` (compact [h, w, c]) at the end
};

// Per channel: the mean and 1/sqrt(variance + 1e-8) over the image, in f32. A threadgroup of
// 32 x 32 threads takes 32 channels: each column of threads one channel (neighbouring threads
// read neighbouring channels), its 32 threads splitting the pixels and adding up their parts in
// threadgroup memory.
kernel void instance_stats(device const half *x [[buffer(0)]],
                           device float2 *stats [[buffer(1)]],
                           constant NormParams &p [[buffer(2)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float part[32][33];
    threadgroup float mean[32];
    const uint ch = tg.x * 32 + tid.x, n = p.h * p.w;
    const bool live = ch < p.c;
    float sum = 0;
    if (live)
        for (uint i = tid.y; i < n; i += 32)
            sum += float(x[((i / p.w) * p.sw + i % p.w) * p.c + ch]);
    part[tid.y][tid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.y == 0) {
        float s = 0;
        for (uint j = 0; j < 32; j++)
            s += part[j][tid.x];
        mean[tid.x] = s / float(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = mean[tid.x];
    float sq = 0;
    if (live)
        for (uint i = tid.y; i < n; i += 32) {
            const float d = float(x[((i / p.w) * p.sw + i % p.w) * p.c + ch]) - m;
            sq += d * d;
        }
    part[tid.y][tid.x] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.y == 0 && live) {
        float s = 0;
        for (uint j = 0; j < 32; j++)
            s += part[j][tid.x];
        stats[ch] = float2(m, rsqrt(s / float(n) + 1e-8f));
    }
}

// AdaIN: normalizes each channel with its stats and applies the style (scale in the first c
// values, shift in the next c), then optionally ReLU and a residual. Writes compact [h, w, c].
kernel void adain(device const half *x [[buffer(0)]],
                  device const float2 *stats [[buffer(1)]],
                  device const float *style [[buffer(2)]],
                  device const half *res [[buffer(3)]],
                  device half *out [[buffer(4)]],
                  constant NormParams &p [[buffer(5)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w * p.c)
        return;
    const uint ch = i % p.c, px = (i / p.c) % p.w, py = i / (p.c * p.w);
    const float2 s = stats[ch];
    float v = (float(x[(py * p.sw + px) * p.c + ch]) - s.x) * s.y * style[ch] + style[p.c + ch];
    if (p.relu)
        v = max(v, 0.0f);
    if (p.residual)
        v += float(res[i]);
    out[i] = half(v);
}

struct UpParams {
    uint h, w, c;  // the source, compact
};

static float texel(device const half *src, constant UpParams &p, uint y, uint x, uint ch) {
    return float(src[(y * p.w + x) * p.c + ch]);
}

// Bilinear upsampling by 2 (half-pixel centers, edges clamped). One thread per output value.
kernel void upsample2(device const half *src [[buffer(0)]],
                      device half *dst [[buffer(1)]],
                      constant UpParams &p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    const uint ow = 2 * p.w, oh = 2 * p.h;
    if (i >= oh * ow * p.c)
        return;
    const uint ch = i % p.c, ox = (i / p.c) % ow, oy = i / (p.c * ow);
    const float fy = max((float(oy) + 0.5f) * 0.5f - 0.5f, 0.0f);
    const float fx = max((float(ox) + 0.5f) * 0.5f - 0.5f, 0.0f);
    const uint y0 = uint(fy), x0 = uint(fx);
    const uint y1 = min(y0 + 1, p.h - 1), x1 = min(x0 + 1, p.w - 1);
    const float ty = fy - float(y0), tx = fx - float(x0);
    const float top = texel(src, p, y0, x0, ch) * (1 - tx) + texel(src, p, y0, x1, ch) * tx;
    const float bottom = texel(src, p, y1, x0, ch) * (1 - tx) + texel(src, p, y1, x1, ch) * tx;
    const float v = top * (1 - ty) + bottom * ty;
    dst[i] = half(v);
}

struct FinishParams {
    uint h, w, sw, c;
    uint mode;  // 0: (tanh + 1) / 2, 1: sigmoid
};

// A network's output as f32 compact [h, w, c]: the swapper's (tanh(x) + 1) / 2, or the occlusion
// mask's sigmoid.
kernel void finish(device const half *x [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   constant FinishParams &p [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w * p.c)
        return;
    const uint ch = i % p.c, px = (i / p.c) % p.w, py = i / (p.c * p.w);
    const float v = float(x[(py * p.sw + px) * p.c + ch]);
    out[i] = p.mode == 1 ? 1 / (1 + precise::exp(-v)) : (precise::tanh(v) + 1) * 0.5f;
}

struct AffineParams {
    uint h, w, c;
    uint sw_a, sw_b;  // stored row widths of a and b
    uint scale;       // multiply a by scale[c] and add shift[c]
    uint add;         // add b
    uint act;         // 0 none, 1 ReLU, 2 PReLU with alpha[c]
};

// out = act(a * scale + shift + b), per channel scale, shift and PReLU slope. Writes compact
// [h, w, c]. Covers batch norm (folded), residual adds and the activations after them.
kernel void affine_act(device const half *a [[buffer(0)]],
                       device const half *b [[buffer(1)]],
                       device const float *scale [[buffer(2)]],
                       device const float *shift [[buffer(3)]],
                       device const float *alpha [[buffer(4)]],
                       device half *out [[buffer(5)]],
                       constant AffineParams &p [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w * p.c)
        return;
    const uint ch = i % p.c, px = (i / p.c) % p.w, py = i / (p.c * p.w);
    float v = float(a[(py * p.sw_a + px) * p.c + ch]);
    if (p.scale)
        v = v * scale[ch] + shift[ch];
    if (p.add)
        v += float(b[(py * p.sw_b + px) * p.c + ch]);
    if (p.act == 1)
        v = max(v, 0.0f);
    else if (p.act == 2)
        v = v < 0 ? v * alpha[ch] : v;
    out[i] = half(v);
}

struct PoolParams {
    uint h, w, sw, c;  // the source
    uint avg;          // average (windows past the edge count only the pixels inside), else max
};

// 2x2 pooling with stride 2, the output rounded up. Writes compact.
kernel void pool2(device const half *x [[buffer(0)]],
                  device half *out [[buffer(1)]],
                  constant PoolParams &p [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    const uint oh = (p.h + 1) / 2, ow = (p.w + 1) / 2;
    if (i >= oh * ow * p.c)
        return;
    const uint ch = i % p.c, ox = (i / p.c) % ow, oy = i / (p.c * ow);
    float acc = p.avg ? 0.0f : -INFINITY;
    uint n = 0;
    for (uint dy = 0; dy < 2; dy++)
        for (uint dx = 0; dx < 2; dx++) {
            const uint y = 2 * oy + dy, x0 = 2 * ox + dx;
            if (y >= p.h || x0 >= p.w)
                continue;
            const float v = float(x[(y * p.sw + x0) * p.c + ch]);
            acc = p.avg ? acc + v : max(acc, v);
            n++;
        }
    out[i] = half(p.avg ? acc / float(n) : acc);
}

struct UpAddParams {
    uint h, w, sw, c;  // a, the output's size
    uint bh, bw, bsw;  // b, smaller
};

// out = a + b upsampled to a's size by nearest neighbour. Writes compact.
kernel void up_add(device const half *a [[buffer(0)]],
                   device const half *b [[buffer(1)]],
                   device half *out [[buffer(2)]],
                   constant UpAddParams &p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w * p.c)
        return;
    const uint ch = i % p.c, px = (i / p.c) % p.w, py = i / (p.c * p.w);
    const uint by = py * p.bh / p.h, bx = px * p.bw / p.w;
    out[i] = half(float(a[(py * p.sw + px) * p.c + ch]) + float(b[(by * p.bsw + bx) * p.c + ch]));
}
