// A k x k conv on Metal 4's tensor ops (Metal Performance Primitives), which can use the matrix
// hardware of newer GPUs.
//
// The input is stored already padded (zeros or reflected, as the layer needs), so for each tap
// the input pixels under a run of output pixels are one strided block: the conv is k*k matmuls
// accumulated in a cooperative tensor, each output pixels x Cin times Cin x output channels. A
// threadgroup of 4 simdgroups computes up to BM output pixels by up to BN output channels; tiles
// at the ends and the last channels are trimmed through the tensors' extents. Bias and the
// activation (leaky ReLU, slope 1 for none) are applied in registers before the store.
//
// The host defines BM and BN, and may define:
//   WIDE   1 for stride 1: output rows are computed as wide as the padded input, so a tile runs
//          across rows (consecutive output pixels read consecutive input pixels). The extra
//          k - 1 columns per row are junk, and land where the next layer's padding goes.
//          Otherwise a tile stays within one output row.
//   SPLIT  n > 1 splits the taps over n threadgroups (grid z), each storing f32 partial sums;
//          `reduce` then adds them up with the bias and activation.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef WIDE
#define WIDE 0
#endif
#ifndef SPLIT
#define SPLIT 1
#endif

struct Params {
    uint in_w;  // the padded input's width
    uint out_h, out_w;
    uint m;     // output pixels as stored: out_h * (WIDE ? in_w : out_w)
    uint cin, cout;
    uint k, stride;
    float slope;
};

using T2 = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
using F2 = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

kernel void conv(device const half *x [[buffer(0)]],      // [padded H, padded W, Cin]
                 device const half *wt [[buffer(1)]],     // [Cout, k, k, Cin]
                 device const float *bias [[buffer(2)]],  // [Cout]
                 device half *y [[buffer(3)]],            // [out H, out W or padded W, Cout]
                 constant Params &p [[buffer(4)]],
                 device float *part [[buffer(5)]],        // [SPLIT, m, Cout] when split
                 uint3 tg [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    const int cin = int(p.cin), k = int(p.k), s = int(p.stride), depth = k * k * cin;
    const int in_w = int(p.in_w), n0 = int(tg.y) * BN, cols = min(BN, int(p.cout) - n0);
#if WIDE
    // wide output pixel m reads input pixel m + dy * in_w + dx; the last k - 1 would read past
    // the input and are junk anyway
    const int m0 = int(tg.x) * BM;
    const int rows = min(BM, int(p.out_h) * in_w - (k - 1) - m0);
    const int a_at = m0, a_step = cin, out_at = m0;
#else
    const int tiles = (int(p.out_w) + BM - 1) / BM;
    const int row = int(tg.x) / tiles, col = (int(tg.x) % tiles) * BM;
    const int rows = min(BM, int(p.out_w) - col);
    const int a_at = row * s * in_w + col * s, a_step = s * cin, out_at = row * int(p.out_w) + col;
#endif
    device half *xs = const_cast<device half *>(x) + a_at * cin;
    device half *ws = const_cast<device half *>(wt) + n0 * depth;

    // operands as (inner, outer): output pixels x channels, and the weights read transposed
    T2 a0(xs, dextents<int32_t, 2>(cin, rows), array<int32_t, 2>{1, a_step});
    T2 b0(ws, dextents<int32_t, 2>(cin, cols), array<int32_t, 2>{1, depth});
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        acc[i] = 0.0f;

    const int taps = k * k, per = (taps + SPLIT - 1) / SPLIT;
    const int t0 = int(tg.z) * per, t1 = min(taps, t0 + per);
    for (int t = t0; t < t1; t++) {
        const int dy = t / k, dx = t % k;
        T2 a(xs + (dy * in_w + dx) * cin, dextents<int32_t, 2>(cin, rows), array<int32_t, 2>{1, a_step});
        T2 b(ws + t * cin, dextents<int32_t, 2>(cin, cols), array<int32_t, 2>{1, depth});
        op.run(a, b, acc);
    }

#if SPLIT > 1
    F2 o(part + (int(tg.z) * int(p.m) + out_at) * int(p.cout) + n0, dextents<int32_t, 2>(cols, rows),
         array<int32_t, 2>{1, int(p.cout)});
    acc.store(o);
#else
    auto out = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), half>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const int n = n0 + int(acc.get_multidimensional_index(i)[0]);
            const float v = acc[i] + (n < int(p.cout) ? bias[n] : 0.0f);
            out[i] = half(v < 0 ? v * p.slope : v);
        }
    }
    T2 o(y + out_at * int(p.cout) + n0, dextents<int32_t, 2>(cols, rows), array<int32_t, 2>{1, int(p.cout)});
    out.store(o);
#endif
}

// Adds up the split partial sums, with the bias and activation.
kernel void reduce(device const float *part [[buffer(0)]],  // [SPLIT, m, Cout]
                   device const float *bias [[buffer(1)]],
                   device half *y [[buffer(2)]],
                   constant Params &p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    const uint total = p.m * p.cout;
    if (i >= total)
        return;
    float v = bias[i % p.cout];
    for (uint z = 0; z < SPLIT; z++)
        v += part[z * total + i];
    y[i] = half(v < 0 ? v * p.slope : v);
}
