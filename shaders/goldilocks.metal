/*
  The original felt_u64.h.metal was written by andrewmilson for the ministark project:
  https://github.com/andrewmilson/ministark/blob/dfd2a7db386ed03e545d37382d56dcc1dc742caa/gpu/src/metal/felt_u64.h.metal
  
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
#ifndef goldilocks_field
#define goldilocks_field


namespace GoldilocksField
{
    static const constant unsigned long GOLDILOCKS_PRIME = 18446744069414584321;

    // helper for mul
    inline unsigned long reduce128(unsigned long x_hi, unsigned long x_lo){
        unsigned long x_hi_hi = x_hi>>32;
        unsigned long x_hi_lo = x_hi & 0xffffffff;
        unsigned long t0 = x_lo - x_hi_hi;
        if(t0 > x_lo){
            t0 -= 0xffffffff;
        }
        unsigned long t1 = x_hi_lo*0xffffffff;
        unsigned long t2 = t0 + t1;
        if(t2 < t1){
            t2 += 0xffffffff;
        }
        return t2;
    }

    // Prime field
    class Fp
    {
    public:
        Fp() = default;
        constexpr Fp(unsigned long v) : inner(v) {}

        constexpr Fp operator+(const Fp rhs) const
        {
            return Fp(add(inner, rhs.inner));
        }

        constexpr explicit operator unsigned long() const
        {
            return inner;
        }

        Fp operator*(const Fp rhs) const
        {
            return Fp(mul(inner, rhs.inner));
        }

        // used for S-box in the Rescue Prime Optimized hash function
        Fp pow7() {
            unsigned long t2 = mul(inner, inner);
            return mul(mul(mul(t2, t2), t2), inner);
        }

        Fp inverse() 
        {
            unsigned long t2 = exp_acc<1>(inner, inner);
            unsigned long t3 = exp_acc<1>(t2, inner);
            unsigned long t6 = exp_acc<3>(t3, t3);
            unsigned long t12 = exp_acc<6>(t6, t6);
            unsigned long t24 = exp_acc<12>(t12, t12);
            unsigned long t30 = exp_acc<6>(t24, t6);
            unsigned long t31 = exp_acc<1>(t30, inner);
            unsigned long t63 = exp_acc<32>(t31, t31);
            unsigned long inv = exp_acc<1>(t63, inner);
            return Fp(inv);
        }


    private:
        unsigned long inner;

        // Field modulus `p = 2^64 - 2^32 + 1`
        constexpr static const constant unsigned long N = 18446744069414584321;

        // Square of auxiliary modulus R for Montgomery reduction `R2 ≡ (2^64)^2 mod p`
        constexpr static const constant unsigned long R2 = 18446744065119617025;

        template<unsigned N_ACC>
        inline unsigned long exp_acc(unsigned long element, const unsigned long tail) const {
#pragma unroll
            for (unsigned i = 0; i < N_ACC; i++) {
                element = mul(element, element);
            }
            return mul(element, tail);
        }

        template<unsigned N_ACC>
        inline unsigned long sqn(unsigned long element) const {
#pragma unroll
            for (unsigned i = 0; i < N_ACC; i++) {
                element = mul(element, element);
            }
            return element;
        }

        inline unsigned long add(const unsigned long a, const unsigned long b) const
        {
            ulong a_plus_b = a+b;

            if(a_plus_b < a){
                ulong c = a_plus_b + 0xffffffff;
                if(c<a_plus_b){
                    a_plus_b = c+0xffffffff;
                }else{
                  a_plus_b = c;
                }
            }

            // a_plus_b will be at most 36893488138829168640
            // so we can subtract instead of using modulo
            return a_plus_b < N ? a_plus_b : (a_plus_b - N);
        }

        inline unsigned long mul(const unsigned long lhs, const unsigned long rhs) const
        {
            // since lhs = lhs_hi * 2^32 + lhs_lo and rhs = rhs_hi * 2^32 + rhs_lo
            // we can compute lhs * rhs = 
            // lhs_hi * rhs_hi * 2^64 + (lhs_hi * rhs_lo + rhs_hi * lhs_lo)*2^32 + lhs_lo * rhs_lo
            unsigned long a = lhs >> 32;
            unsigned long b = lhs & 0xFFFFFFFF;
            unsigned long c = rhs >> 32;
            unsigned long d = rhs & 0xFFFFFFFF;

            unsigned long ad = a * d;
            unsigned long bd = b * d;

            unsigned long adbc = ad + (b * c);
            unsigned long adbc_carry = adbc < ad ? 1 : 0;

            unsigned long product_lo = bd + (adbc << 32);
            unsigned long product_lo_carry = product_lo < bd ? 1 : 0;
            unsigned long product_hi = (a * c) + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;
            return reduce128(product_hi, product_lo) % N;
        }
    };

}

#endif /* goldilocks_field */