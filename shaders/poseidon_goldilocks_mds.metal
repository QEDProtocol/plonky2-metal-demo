/*
  Modified from https://github.com/andrewmilson/ministark/blob/dfd2a7db386ed03e545d37382d56dcc1dc742caa/gpu/src/metal/hash_shaders.h.metal

  
  The MIT License (MIT)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/


#ifndef poseidon_goldilocks_mds
#define poseidon_goldilocks_mds
#include <metal_stdlib>
#include "goldilocks.metal"
#include "u128.h.metal"
namespace GoldilocksField {
// TODO: use pair from standard library (can't figure out how to import)
template <class t1, class t2>
struct pair {
  t1 a;
  t2 b;
};

inline ulong2 ifft2_real(long2 in) {
    return ulong2((ulong) (in.x + in.y), (ulong) (in.x - in.y));
}

inline ulong4 ifft4_real(long4 in) {
    ulong2 z0 = ifft2_real(long2(in.x + in.w, in.y));
    ulong2 z1 = ifft2_real(long2(in.x - in.w, -in.z));
    return ulong4(z0.x, z1.x, z0.y, z1.y);
}

inline long2 fft2_real(ulong2 in) {
    return long2((long) (in.x + in.y), (long) in.x - (long) in.y);
}

inline long4 fft4_real(ulong4 in) {
    long2 z0 = fft2_real(ulong2(in.x, in.z));
    long2 z1 = fft2_real(ulong2(in.y, in.w));
    return long4(z0.x + z1.x, z0.y, -z1.y, z0.x - z1.x);
}

constant const long3 MDS_FREQ_BLOCK_ONE = long3(16, 32, 16);
constant const pair<long3, long3> MDS_FREQ_BLOCK_TWO = { .a = long3(2, -4, 16), .b = long3(-1, 1, 1) };
constant const long3 MDS_FREQ_BLOCK_THREE = long3(-1, -8, 2);

inline long3 block1(long3 in) {
    return long3(
        in.x * MDS_FREQ_BLOCK_ONE.x + in.y * MDS_FREQ_BLOCK_ONE.z + in.z * MDS_FREQ_BLOCK_ONE.y,
        in.x * MDS_FREQ_BLOCK_ONE.y + in.y * MDS_FREQ_BLOCK_ONE.x + in.z * MDS_FREQ_BLOCK_ONE.z,
        in.x * MDS_FREQ_BLOCK_ONE.z + in.y * MDS_FREQ_BLOCK_ONE.y + in.z * MDS_FREQ_BLOCK_ONE.x
    );
}

inline pair<long3, long3> block2(pair<long3, long3> in) {
    long x0s = in.a.x + in.b.x;
    long x1s = in.a.y + in.b.y;
    long x2s = in.a.z + in.b.z;
    long y0s = MDS_FREQ_BLOCK_TWO.a.x + MDS_FREQ_BLOCK_TWO.b.x;
    long y1s = MDS_FREQ_BLOCK_TWO.a.y + MDS_FREQ_BLOCK_TWO.b.y;
    long y2s = MDS_FREQ_BLOCK_TWO.a.z + MDS_FREQ_BLOCK_TWO.b.z;

    // Compute x0​y0 ​− ix1​y2​ − ix2​y1​ using Karatsuba for complex numbers multiplication
    long2 m0 = long2(in.a.x * MDS_FREQ_BLOCK_TWO.a.x, in.b.x * MDS_FREQ_BLOCK_TWO.b.x);
    long2 m1 = long2(in.a.y * MDS_FREQ_BLOCK_TWO.a.z, in.b.y * MDS_FREQ_BLOCK_TWO.b.z);
    long2 m2 = long2(in.a.z * MDS_FREQ_BLOCK_TWO.a.y, in.b.z * MDS_FREQ_BLOCK_TWO.b.y);
    long z0r = (m0.x - m0.y) + (x1s * y2s - m1.x - m1.y) + (x2s * y1s - m2.x - m2.y);
    long z0i = (x0s * y0s - m0.x - m0.y) + (-m1.x + m1.y) + (-m2.x + m2.y);
    long2 z0 = long2(z0r, z0i);

    // Compute x0​y1​ + x1​y0​ − ix2​y2 using Karatsuba for complex numbers multiplication
    m0 = long2(in.a.x * MDS_FREQ_BLOCK_TWO.a.y, in.b.x * MDS_FREQ_BLOCK_TWO.b.y);
    m1 = long2(in.a.y * MDS_FREQ_BLOCK_TWO.a.x, in.b.y * MDS_FREQ_BLOCK_TWO.b.x);
    m2 = long2(in.a.z * MDS_FREQ_BLOCK_TWO.a.z, in.b.z * MDS_FREQ_BLOCK_TWO.b.z);
    long z1r = (m0.x - m0.y) + (m1.x - m1.y) + (x2s * y2s - m2.x - m2.y);
    long z1i = (x0s * y1s - m0.x - m0.y) + (x1s * y0s - m1.x - m1.y) + (-m2.x + m2.y);
    long2 z1 = long2(z1r, z1i);

    // Compute x0​y2​ + x1​y1 ​+ x2​y0​ using Karatsuba for complex numbers multiplication
    m0 = long2(in.a.x * MDS_FREQ_BLOCK_TWO.a.z, in.b.x * MDS_FREQ_BLOCK_TWO.b.z);
    m1 = long2(in.a.y * MDS_FREQ_BLOCK_TWO.a.y, in.b.y * MDS_FREQ_BLOCK_TWO.b.y);
    m2 = long2(in.a.z * MDS_FREQ_BLOCK_TWO.a.x, in.b.z * MDS_FREQ_BLOCK_TWO.b.x);
    long z2r = (m0.x - m0.y) + (m1.x - m1.y) + (m2.x - m2.y);
    long z2i = (x0s * y2s - m0.x - m0.y) + (x1s * y1s - m1.x - m1.y) + (x2s * y0s - m2.x - m2.y);
    long2 z2 = long2(z2r, z2i);

    return { .a = long3(z0.x, z1.x, z2.x), .b = long3(z0.y, z1.y, z2.y) };
}

inline long3 block3(long3 in) {
    return long3(
        in.x * MDS_FREQ_BLOCK_THREE.x - in.y * MDS_FREQ_BLOCK_THREE.z - in.z * MDS_FREQ_BLOCK_THREE.y,
        in.x * MDS_FREQ_BLOCK_THREE.y + in.y * MDS_FREQ_BLOCK_THREE.x - in.z * MDS_FREQ_BLOCK_THREE.z,
        in.x * MDS_FREQ_BLOCK_THREE.z + in.y * MDS_FREQ_BLOCK_THREE.y + in.z * MDS_FREQ_BLOCK_THREE.x
    );
}

// Adapted from Miden: 
// https://github.com/0xPolygonMiden/crypto/blob/main/src/hash/rpo/mds_freq.rs
inline void mds_multiply_freq(unsigned long state[12]) {
    long4 u0 = fft4_real(ulong4(state[0], state[3], state[6], state[9]));
    long4 u1 = fft4_real(ulong4(state[1], state[4], state[7], state[10]));
    long4 u2 = fft4_real(ulong4(state[2], state[5], state[8], state[11]));

    long3 v0 = block1(long3(u0.x, u1.x, u2.x));
    pair<long3, long3> v1 = block2({ .a = long3(u0.y, u1.y, u2.y), .b = long3(u0.z, u1.z, u2.z) });
    long3 v2 = block3(long3(u0.w, u1.w, u2.w));

    ulong4 s0 = ifft4_real(long4(v0.x, v1.a.x, v1.b.x, v2.x));
    ulong4 s1 = ifft4_real(long4(v0.y, v1.a.y, v1.b.y, v2.y));
    ulong4 s2 = ifft4_real(long4(v0.z, v1.a.z, v1.b.z, v2.z));

    state[0] = s0.x;
    state[1] = s1.x;
    state[2] = s2.x;
    state[3] = s0.y;
    state[4] = s1.y;
    state[5] = s2.y;
    state[6] = s0.z;
    state[7] = s1.z;
    state[8] = s2.z;
    state[9] = s0.w;
    state[10] = s1.w;
    state[11] = s2.w;
}

inline void apply_mds_freq(thread Fp* shared, unsigned local_state_offset) {
    unsigned long state_l[12];
    unsigned long state_h[12];

#pragma unroll
    for (unsigned j = 0; j < 12; j++) {
        Fp element = shared[local_state_offset + j];
        unsigned long s = (unsigned long) element;
        state_l[j] = s & 0xFFFFFFFF;
        state_h[j] = s >> 32;
    }

    mds_multiply_freq(state_l);
    mds_multiply_freq(state_h);

        u128 s = u128(state_l[0]) + (u128(state_h[0]) << 32);
        s.accumulate_mul_2_ulong(static_cast<ulong>(shared[0]), 8);
        ulong reduced = reduce128(s.high, s.low);
        shared[local_state_offset] = Fp(reduced<GOLDILOCKS_PRIME?reduced:(reduced-GOLDILOCKS_PRIME));
#pragma unroll
    for (unsigned j = 1; j < 12; j++) {
        s = u128(state_l[j]) + (u128(state_h[j]) << 32);
        reduced = reduce128(s.high, s.low);
        shared[local_state_offset + j] = Fp(reduced<GOLDILOCKS_PRIME?reduced:(reduced-GOLDILOCKS_PRIME));
    }

}

}
#endif