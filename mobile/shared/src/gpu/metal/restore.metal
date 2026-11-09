// The enhancer's (GPEN) and the occlusion mask's (XSeg) kernels, compiled on their own: if they
// fail to compile on a device, only Enhance and Objects are unavailable. Activations are f16 NHWC
// (batch 1), rows stored `sw` wide where noted, as ops.metal's.

#include <metal_stdlib>
using namespace metal;

// ------------------------------------------------------------------------------ the enhancer

// StyleGAN's [1, 3, 3, 1] filter per axis: / 8 for the encoder's blur, / 4 (gain 4 in 2D) after a
// zero-inserting 2x upsample
constant float FIR[4] = {0.125f, 0.375f, 0.375f, 0.125f};
constant float FIR_UP[4] = {0.25f, 0.75f, 0.75f, 0.25f};

struct FirParams {
    uint h, w, sw, c;  // the source
};

// The enhancer encoder's blur before each stride-2 conv: [h, w, c] -> compact [h + 1, w + 1, c],
// the 4x4 filter over the source zero-padded by 2.
kernel void fir_down(device const half *x [[buffer(0)]],
                     device half *y [[buffer(1)]],
                     constant FirParams &p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    const uint ow = p.w + 1, oh = p.h + 1;
    if (i >= oh * ow * p.c)
        return;
    const uint ch = i % p.c, ox = (i / p.c) % ow, oy = i / (p.c * ow);
    float acc = 0;
    for (uint ty = 0; ty < 4; ty++) {
        const int sy = int(oy + ty) - 2;
        if (sy < 0 || sy >= int(p.h))
            continue;
        for (uint tx = 0; tx < 4; tx++) {
            const int sx = int(ox + tx) - 2;
            if (sx < 0 || sx >= int(p.w))
                continue;
            acc += float(x[(uint(sy) * p.sw + uint(sx)) * p.c + ch]) * FIR[ty] * FIR[tx];
        }
    }
    y[i] = half(acc);
}

struct StyledParams {
    uint h, w, c;  // the output's size, and the conv's channels (the output has 2c)
    uint ysw;      // the conv output's stored row width
    uint fsw;      // the feature map's
    uint up;       // the conv is the polyphase transposed one
    float noise;
};

// A modulated conv's output through the enhancer's activation: compact [h, w, 2c] =
// LeakyReLU_0.2([y * sqrt 2 / sqrt(d + 1e-8) | noise * feat] + act). up: y is the polyphase
// transposed conv [h/2 + 1, w/2 + 1, 4c] (its pixel (2a + py, 2b + px) is y's (a, b), channels
// (py * 2 + px) * c..), blurred by the [1, 3, 3, 1] filter (gain 4, zero pad 1) on the way.
kernel void styled_act(device const half *y [[buffer(0)]],
                       device const float *d [[buffer(1)]],
                       device const float *act [[buffer(2)]],
                       device const half *feat [[buffer(3)]],
                       device half *out [[buffer(4)]],
                       constant StyledParams &p [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    const uint c2 = 2 * p.c;
    if (i >= p.h * p.w * c2)
        return;
    const uint ch = i % c2, x = (i / c2) % p.w, yy = i / (c2 * p.w);
    float v;
    if (ch < p.c) {
        if (p.up) {
            v = 0;
            for (uint ty = 0; ty < 4; ty++) {
                const int my = int(yy + ty) - 1;
                if (my < 0)
                    continue;
                for (uint tx = 0; tx < 4; tx++) {
                    const int mx = int(x + tx) - 1;
                    if (mx < 0)
                        continue;
                    const uint q = (uint(my) & 1) * 2 + (uint(mx) & 1);
                    v += float(y[((uint(my) / 2) * p.ysw + uint(mx) / 2) * 4 * p.c + q * p.c + ch]) * FIR_UP[ty] * FIR_UP[tx];
                }
            }
        } else {
            v = float(y[(yy * p.ysw + x) * p.c + ch]);
        }
        v = v * rsqrt(d[ch] + 1e-8f) * M_SQRT2_F + act[ch];
    } else {
        v = p.noise * float(feat[(yy * p.fsw + x) * p.c + ch - p.c]) + act[ch];
    }
    out[i] = half(v < 0 ? 0.2f * v : v);
}

struct RgbParams {
    uint h, w, sw, c;  // the input
    uint skip;         // add the image before, upsampled
};

// The enhancer's modulated 1x1 conv to RGB, f32 compact [h, w, 3] = (x * s) @ wt^T + bias (wt
// [3, c]), plus with skip that image ([h/2, w/2, 3]) upsampled 2x: zero insertion and the
// [1, 3, 3, 1] filter, weights 1/4 and 3/4 per axis, zeros outside. One thread per pixel.
kernel void to_rgb(device const half *x [[buffer(0)]],
                   device const float *s [[buffer(1)]],
                   device const float *wt [[buffer(2)]],
                   device const float *bias [[buffer(3)]],
                   device const float *skip [[buffer(4)]],
                   device float *out [[buffer(5)]],
                   constant RgbParams &p [[buffer(6)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p.h * p.w)
        return;
    const uint px = i % p.w, py = i / p.w;
    device const half *xs = x + (py * p.sw + px) * p.c;
    float3 a = float3(bias[0], bias[1], bias[2]);
    for (uint c = 0; c < p.c; c++) {
        const float v = float(xs[c]) * s[c];
        a += v * float3(wt[c], wt[p.c + c], wt[2 * p.c + c]);
    }
    if (p.skip) {
        const int sh = int(p.h / 2), sw = int(p.w / 2);
        const int y0 = (int(py) + 1) / 2 - 1, x0 = (int(px) + 1) / 2 - 1;  // floor((y - 1) / 2)
        const float wy = (py & 1) == 0 ? 0.75f : 0.25f, wx = (px & 1) == 0 ? 0.75f : 0.25f;
        float3 t[2][2];
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                const int yy = y0 + dy, xx = x0 + dx;
                t[dy][dx] = (yy < 0 || xx < 0 || yy >= sh || xx >= sw) ? float3(0)
                    : float3(skip[(yy * sw + xx) * 3], skip[(yy * sw + xx) * 3 + 1], skip[(yy * sw + xx) * 3 + 2]);
            }
        a += mix(mix(t[0][0], t[0][1], wx), mix(t[1][0], t[1][1], wx), wy);
    }
    out[i * 3] = a.x;
    out[i * 3 + 1] = a.y;
    out[i * 3 + 2] = a.z;
}

// Pixel norm of a vector in place: x / sqrt(mean(x^2) + 1e-8). One simdgroup.
kernel void pixel_norm(device float *x [[buffer(0)]],
                       constant uint &n [[buffer(1)]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float s = 0;
    for (uint i = lane; i < n; i += 32)
        s += x[i] * x[i];
    const float k = rsqrt(simd_sum(s) / float(n) + 1e-8f);
    for (uint i = lane; i < n; i += 32)
        x[i] *= k;
}

kernel void square_values(device const float *x [[buffer(0)]],
                   device float *y [[buffer(1)]],
                   constant uint &n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n)
        y[i] = x[i] * x[i];
}

// ------------------------------------------------------------------------------ the occlusion mask

struct FrnParams {
    uint h, w, sw;
    uint c;       // channels
    uint groups;  // > 1: a polyphase transposed conv, whose groups of c channels are the same ones
    float eps;
};

// FRN's statistics folded into max(x * S + T, U) per channel: S = gamma / sqrt(mean(x^2) + eps),
// T = beta, U = tau (frn: [gamma, beta, tau]). A threadgroup of 32 x 32 threads takes 32 channels,
// as instance_stats.
kernel void frn_stats(device const half *x [[buffer(0)]],
                      device const float *frn [[buffer(1)]],
                      device float4 *pro [[buffer(2)]],
                      constant FrnParams &p [[buffer(3)]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float part[32][33];
    const uint ch = tg.x * 32 + tid.x, n = p.h * p.w * p.groups, gc = p.groups * p.c;
    float sq = 0;
    if (ch < p.c)
        for (uint i = tid.y; i < n; i += 32) {
            const uint px = i / p.groups, g = i % p.groups;
            const float v = float(x[((px / p.w) * p.sw + px % p.w) * gc + g * p.c + ch]);
            sq += v * v;
        }
    part[tid.y][tid.x] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.y == 0 && ch < p.c) {
        float s = 0;
        for (uint j = 0; j < 32; j++)
            s += part[j][tid.x];
        pro[ch] = float4(frn[ch] * rsqrt(s / float(n) + p.eps), frn[p.c + ch], frn[2 * p.c + ch], 0);
    }
}

struct FrnOutParams {
    uint oh, ow;   // the output's size
    uint c, cs;    // its channels from x, and from skip after them
    uint xsw;      // x's stored row width
    uint shuffle;  // x is a polyphase transposed conv [oh/2, ow/2, 4c]
};

// FRN + TLU with frn_stats' values, compact [oh, ow, c + cs]; skip (compact [oh, ow, cs]) appended.
kernel void frn_out(device const half *x [[buffer(0)]],
                    device const float4 *pro [[buffer(1)]],
                    device const half *skip [[buffer(2)]],
                    device half *out [[buffer(3)]],
                    constant FrnOutParams &p [[buffer(4)]],
                    uint i [[thread_position_in_grid]]) {
    const uint ct = p.c + p.cs;
    if (i >= p.oh * p.ow * ct)
        return;
    const uint ch = i % ct, xx = (i / ct) % p.ow, yy = i / (ct * p.ow);
    if (ch >= p.c) {
        out[i] = skip[(yy * p.ow + xx) * p.cs + ch - p.c];
        return;
    }
    const uint src = p.shuffle ? ((yy / 2) * p.xsw + xx / 2) * 4 * p.c + ((yy & 1) * 2 + (xx & 1)) * p.c + ch
                               : (yy * p.xsw + xx) * p.c + ch;
    const float4 q = pro[ch];
    out[i] = half(max(float(x[src]) * q.x + q.y, q.z));
}

struct BlurDownParams {
    uint h, w, sw, c;  // the source
    uint taps;
    float k[4];        // normalized
};

// XSeg's blurred stride-2 downsampling: compact [h/2, w/2, c], the separable filter over the
// source zero-padded by 1 before: output pixel o reads 2o - 1...
kernel void blur_down(device const half *x [[buffer(0)]],
                      device half *y [[buffer(1)]],
                      constant BlurDownParams &p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    const uint oh = p.h / 2, ow = p.w / 2;
    if (i >= oh * ow * p.c)
        return;
    const uint ch = i % p.c, ox = (i / p.c) % ow, oy = i / (p.c * ow);
    float acc = 0;
    for (uint ty = 0; ty < p.taps; ty++) {
        const int sy = 2 * int(oy) + int(ty) - 1;
        if (sy < 0 || sy >= int(p.h))
            continue;
        for (uint tx = 0; tx < p.taps; tx++) {
            const int sx = 2 * int(ox) + int(tx) - 1;
            if (sx < 0 || sx >= int(p.w))
                continue;
            acc += float(x[(uint(sy) * p.sw + uint(sx)) * p.c + ch]) * p.k[ty] * p.k[tx];
        }
    }
    y[i] = half(acc);
}

// f32 values to f16 (the bottleneck's output, as an activation).
kernel void to_half(device const float *x [[buffer(0)]],
                    device half *y [[buffer(1)]],
                    constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n)
        y[i] = half(x[i]);
}

struct MaskParams {
    uint n;         // the mask is n x n
    uint r;         // the disk's radius (morph), or the taps' (2r + 1, blur)
    uint erode;     // morph: the least instead of the most
    uint vertical;  // blur: along y
    uint finish;    // blur: then (clamp(v, 0.5, 1) - 0.5) * 2
};

// Grows (the most in a disk of radius r around each pixel) or shrinks what's bright in an n x n
// f32 mask. What's outside doesn't count.
kernel void mask_morph(device const float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       constant MaskParams &p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.n)
        return;
    const int n = int(p.n), r = int(p.r), px = int(i % p.n), py = int(i / p.n);
    float v = x[i];
    for (int yy = max(py - r, 0); yy <= min(py + r, n - 1); yy++) {
        const int w = int(sqrt(float(r * r - (yy - py) * (yy - py))));
        for (int xx = max(px - w, 0); xx <= min(px + w, n - 1); xx++)
            v = p.erode ? min(v, x[yy * n + xx]) : max(v, x[yy * n + xx]);
    }
    y[i] = v;
}

// One axis of a Gaussian blur of an n x n f32 mask (reflect-101 borders), taps [2r + 1].
kernel void mask_blur(device const float *x [[buffer(0)]],
                      device const float *taps [[buffer(1)]],
                      device float *y [[buffer(2)]],
                      constant MaskParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.n)
        return;
    const int n = int(p.n);
    const int pos = int(p.vertical ? i / p.n : i % p.n);
    const int along = p.vertical ? n : 1;
    const int base = int(i) - pos * along;
    float v = 0;
    for (uint t = 0; t <= 2 * p.r; t++) {
        int j = abs(pos + int(t) - int(p.r));
        if (j >= n)
            j = 2 * (n - 1) - j;
        v += x[base + j * along] * taps[t];
    }
    y[i] = p.finish ? (clamp(v, 0.5f, 1.0f) - 0.5f) * 2 : v;
}

// ------------------------------------------------------------------------------ pasting

struct UnoccludeParams {
    float m[6];          // frame pixel -> mask pixel
    uint x1, y1, bw, bh; // the box of frame pixels the mask covers
    uint stride, bgra;   // the frame's
    uint ostride, obgra; // the original's
    uint n;              // the mask is n x n
};

static float mask_at(device const float *mask, uint n, float x, float y) {
    return (x < 0 || y < 0 || x > float(n - 1) || y > float(n - 1)) ? 1.0f : mask[uint(y) * n + uint(x)];
}

// What covers a face put back, the last step: frame = orig + (frame - orig) * a, a the occlusion
// mask sampled bilinear through its map (1 outside it), orig the frame before any swap. One
// thread per frame pixel in the box.
kernel void unocclude(device uchar *frame [[buffer(0)]],
                      device const uchar *orig [[buffer(1)]],
                      device const float *mask [[buffer(2)]],
                      constant UnoccludeParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.bw * p.bh)
        return;
    const uint px = p.x1 + i % p.bw, py = p.y1 + i / p.bw;
    const float fx = float(px), fy = float(py);
    const float sx = p.m[0] * fx + p.m[1] * fy + p.m[2], sy = p.m[3] * fx + p.m[4] * fy + p.m[5];
    const float x0 = floor(sx), y0 = floor(sy), tx = sx - x0, ty = sy - y0;
    const float a = mix(mix(mask_at(mask, p.n, x0, y0), mask_at(mask, p.n, x0 + 1, y0), tx),
                        mix(mask_at(mask, p.n, x0, y0 + 1), mask_at(mask, p.n, x0 + 1, y0 + 1), tx), ty);
    if (a >= 1)
        return;
    device uchar *q = frame + py * p.stride + px * (p.bgra ? 4 : 3);
    device const uchar *o = orig + py * p.ostride + px * (p.obgra ? 4 : 3);
    const float3 f = p.bgra ? float3(q[2], q[1], q[0]) : float3(q[0], q[1], q[2]);
    const float3 g = p.obgra ? float3(o[2], o[1], o[0]) : float3(o[0], o[1], o[2]);
    const uchar3 v = uchar3(clamp(rint(g + (f - g) * a), 0.0f, 255.0f));
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

struct BandParams {
    uint n;    // the enhancer's crop
    uint lo;   // the band's size
    uint k;    // samples per axis
    float s;   // n / lo
};

static float3 band_px(device const half *c, device const float *g, uint n, int x, int y) {
    const uint i = (uint(clamp(y, 0, int(n) - 1)) * n + uint(clamp(x, 0, int(n) - 1))) * 3;
    return float3(c[i], c[i + 1], c[i + 2]) - float3(g[i], g[i + 1], g[i + 2]);
}

// The enhancer's detail band: the part of c - g (its input and output, [n, n, 3]) below lo px,
// as f32 [lo, lo, 3]: per pixel the average of k x k bilinear samples, edges replicated.
kernel void band_down(device const half *c [[buffer(0)]],
                      device const float *g [[buffer(1)]],
                      device float *d [[buffer(2)]],
                      constant BandParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.lo * p.lo)
        return;
    const float ox = float(i % p.lo), oy = float(i / p.lo), kn = float(p.k);
    float3 acc = 0;
    for (uint j = 0; j < p.k; j++)
        for (uint t = 0; t < p.k; t++) {
            const float sx = p.s * (ox + (float(t) + 0.5f) / kn - 0.5f), sy = p.s * (oy + (float(j) + 0.5f) / kn - 0.5f);
            const float fx = floor(sx), fy = floor(sy);
            const int x0 = int(fx), y0 = int(fy);
            acc += mix(mix(band_px(c, g, p.n, x0, y0), band_px(c, g, p.n, x0 + 1, y0), sx - fx),
                       mix(band_px(c, g, p.n, x0, y0 + 1), band_px(c, g, p.n, x0 + 1, y0 + 1), sx - fx), sy - fy);
        }
    acc /= kn * kn;
    d[i * 3] = acc.x;
    d[i * 3 + 1] = acc.y;
    d[i * 3 + 2] = acc.z;
}

static float3 d_px(device const float *d, uint lo, int x, int y) {
    const uint i = (uint(clamp(y, 0, int(lo) - 1)) * lo + uint(clamp(x, 0, int(lo) - 1))) * 3;
    return float3(d[i], d[i + 1], d[i + 2]);
}

// The enhancer's output g ([n, n, 3] in [-1, 1]) for the paste, in 0..1 units: (g + 1) / 2, or
// with the band d (lo > 0) its change to the input c, (g + up(d) - c) / 2, up(d) the band
// bilinearly enlarged to n (edges replicated).
kernel void enhance_out(device const float *g [[buffer(0)]],
                        device const float *d [[buffer(1)]],
                        device const half *c [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant BandParams &p [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.n)
        return;
    float3 v = float3(g[i * 3], g[i * 3 + 1], g[i * 3 + 2]);
    if (p.lo > 0) {
        const float r = float(p.lo) / float(p.n);
        const float sx = float(i % p.n) * r, sy = float(i / p.n) * r;
        const float fx = floor(sx), fy = floor(sy);
        const int x0 = int(fx), y0 = int(fy);
        v += mix(mix(d_px(d, p.lo, x0, y0), d_px(d, p.lo, x0 + 1, y0), sx - fx),
                 mix(d_px(d, p.lo, x0, y0 + 1), d_px(d, p.lo, x0 + 1, y0 + 1), sx - fx), sy - fy);
        v = (v - float3(c[i * 3], c[i * 3 + 1], c[i * 3 + 2])) * 0.5f;
    } else {
        v = v * 0.5f + 0.5f;
    }
    out[i * 3] = v.x;
    out[i * 3 + 1] = v.y;
    out[i * 3 + 2] = v.z;
}
