/*
  u128.h.metal was written by andrewmilson for the ministark project:
  https://github.com/andrewmilson/ministark/blob/dfd2a7db386ed03e545d37382d56dcc1dc742caa/gpu/src/metal/u128.h.metal
  
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
#ifndef u128_h
#define u128_h

class u128
{
public:
    u128() = default;
    constexpr u128(int l) : low(l), high(0) {}
    constexpr u128(unsigned long l) : low(l), high(0) {}
    constexpr u128(bool b) : low(b), high(0) {}
    constexpr u128(unsigned long h, unsigned long l) : low(l), high(h) {}

    constexpr u128 operator+(const u128 rhs) const
    {
        return u128(high + rhs.high + ((low + rhs.low) < low), low + rhs.low);
    }

    constexpr u128 operator+=(const u128 rhs)
    {
        *this = *this + rhs;
        return *this;
    }
    

    constexpr inline u128 operator-(const u128 rhs) const
    {
        return u128(high - rhs.high - ((low - rhs.low) > low), low - rhs.low);
    }

    constexpr u128 operator-=(const u128 rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr bool operator==(const u128 rhs) const
    {
        return high == rhs.high && low == rhs.low;
    }

    constexpr bool operator!=(const u128 rhs) const
    {
        return !(*this == rhs);
    }

    constexpr bool operator<(const u128 rhs) const
    {
        return ((high == rhs.high) && (low < rhs.low)) || (high < rhs.high);
    }

    constexpr u128 operator&(const u128 rhs) const
    {
        return u128(high & rhs.high, low & rhs.low);
    }

    constexpr u128 operator|(const u128 rhs) const
    {
        return u128(high | rhs.high, low | rhs.low);
    }

    constexpr bool operator>(const u128 rhs) const
    {
        return ((high == rhs.high) && (low > rhs.low)) || (high > rhs.high);
    }

    constexpr bool operator>=(const u128 rhs) const
    {
        return !(*this < rhs);
    }

    constexpr bool operator<=(const u128 rhs) const
    {
        return !(*this > rhs);
    }

    constexpr inline u128 operator>>(unsigned shift) const
    {
        // TODO: reduce branch conditions
        if (shift >= 128)
        {
            return u128(0);
        }
        else if (shift == 64)
        {
            return u128(0, high);
        }
        else if (shift == 0)
        {
            return *this;
        }
        else if (shift < 64)
        {
            return u128(high >> shift, (high << (64 - shift)) | (low >> shift));
        }
        else if ((128 > shift) && (shift > 64))
        {
            return u128(0, (high >> (shift - 64)));
        }
        else
        {
            return u128(0);
        }
    }

    constexpr inline u128 operator<<(unsigned shift) const
    {
        // TODO: reduce branch conditions
        if (shift >= 128)
        {
            return u128(0);
        }
        else if (shift == 64)
        {
            return u128(low, 0);
        }
        else if (shift == 0)
        {
            return *this;
        }
        else if (shift < 64)
        {
            return u128((high << shift) | (low >> (64 - shift)), low << shift);
        }
        else if ((128 > shift) && (shift > 64))
        {
            return u128((low >> (shift - 64)), 0);
        }
        else
        {
            return u128(0);
        }
    }

    constexpr u128 operator>>=(unsigned rhs)
    {
        *this = *this >> rhs;
        return *this;
    }

    u128 operator*(const bool rhs) const
    {
        return u128(high * rhs, low * rhs);
    }

    u128 operator*(const u128 rhs) const
    {
        unsigned long t_low_high = metal::mulhi(low, rhs.high);
        unsigned long t_high = metal::mulhi(low, rhs.low);
        unsigned long t_high_low = metal::mulhi(high, rhs.low);
        unsigned long t_low = low * rhs.low;
        return u128(t_low_high + t_high_low + t_high, t_low);
    }

    u128 operator*=(const u128 rhs)
    {
        *this = *this * rhs;
        return *this;
    }
    void accumulate_mul_2_ulong(const ulong a, const ulong b){
        *this = *this + u128(metal::mulhi(a, b), a*b);
    }

    // TODO: Could get better performance with  smaller limb size
    // Not sure what word size is for M1 GPU
#ifdef __LITTLE_ENDIAN__
    unsigned long low;
    unsigned long high;
#endif
#ifdef __BIG_ENDIAN__
    unsigned long high;
    unsigned long low;
#endif
};

#endif /* u128_h */
