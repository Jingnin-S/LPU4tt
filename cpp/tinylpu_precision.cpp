#include "tinylpu_precision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kMaxRows = 64;
constexpr std::size_t kMaxK = 4096;
constexpr std::size_t kMaxConversionElements = 1u << 20;

thread_local char g_last_error[256] = {0};

void clear_error() noexcept { g_last_error[0] = '\0'; }

void set_error(const char *message) noexcept {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

uint32_t float_bits(float value) noexcept {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(uint32_t bits) noexcept {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t round_shift_rne(uint32_t value, unsigned shift) noexcept {
  if (shift == 0) {
    return value;
  }
  // The significand supplied to this helper has at most 24 bits.  For a
  // larger shift it is strictly below half of the first output unit.
  if (shift >= 32) {
    return 0;
  }
  const uint32_t quotient = value >> shift;
  const uint32_t remainder = value & ((uint32_t{1} << shift) - 1u);
  const uint32_t halfway = uint32_t{1} << (shift - 1u);
  if (remainder > halfway ||
      (remainder == halfway && (quotient & 1u) != 0u)) {
    return quotient + 1u;
  }
  return quotient;
}

uint16_t float_to_fp16(float value) noexcept {
  const uint32_t source = float_bits(value);
  const uint32_t sign = (source >> 16) & 0x8000u;
  const uint32_t exponent = (source >> 23) & 0xffu;
  const uint32_t mantissa = source & 0x007fffffu;

  if (exponent == 0xffu) {
    if (mantissa == 0u) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    // Preserve NaN rather than allowing a payload of zero to become infinity.
    uint32_t half_mantissa = mantissa >> 13;
    if (half_mantissa == 0u) {
      half_mantissa = 1u;
    }
    return static_cast<uint16_t>(sign | 0x7c00u | half_mantissa);
  }

  if (exponent == 0u) {
    if (mantissa == 0u) {
      return static_cast<uint16_t>(sign);
    }
    // Binary32 subnormals have an effective significand exponent of -149.
    const uint32_t rounded = round_shift_rne(mantissa, 125u);
    return static_cast<uint16_t>(sign | rounded);
  }

  if (exponent < 113u) {
    // Values below 2^-14 become half subnormals.  The hidden one is present
    // because this branch contains only binary32 normal inputs.
    const uint32_t significand = mantissa | 0x00800000u;
    const unsigned shift = 126u - exponent;
    const uint32_t rounded = round_shift_rne(significand, shift);
    return static_cast<uint16_t>(sign | rounded);
  }

  if (exponent >= 143u) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  uint32_t half_exponent = exponent - 112u;
  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half_mantissa & 1u) != 0u)) {
    ++half_mantissa;
  }
  if (half_mantissa == 0x400u) {
    half_mantissa = 0u;
    ++half_exponent;
    if (half_exponent >= 31u) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (half_exponent << 10) | half_mantissa);
}

float fp16_to_float(uint16_t value) noexcept {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;

  if (exponent == 0u) {
    if (mantissa == 0u) {
      return bits_float(sign);
    }
    int32_t unbiased_exponent = -14;
    while ((mantissa & 0x0400u) == 0u) {
      mantissa <<= 1;
      --unbiased_exponent;
    }
    mantissa &= 0x03ffu;
    const uint32_t float_exponent =
        static_cast<uint32_t>(unbiased_exponent + 127);
    return bits_float(sign | (float_exponent << 23) | (mantissa << 13));
  }
  if (exponent == 31u) {
    return bits_float(sign | 0x7f800000u | (mantissa << 13));
  }
  return bits_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

uint16_t float_to_bf16(float value) noexcept {
  const uint32_t source = float_bits(value);
  const uint32_t exponent = (source >> 23) & 0xffu;
  const uint32_t mantissa = source & 0x007fffffu;
  uint32_t upper = source >> 16;

  if (exponent == 0xffu && mantissa != 0u) {
    // A bfloat16 NaN must retain a nonzero payload even if the upper payload
    // bits of the binary32 source are zero.
    if ((upper & 0x007fu) == 0u) {
      upper |= 1u;
    }
    return static_cast<uint16_t>(upper);
  }

  const uint32_t lower = source & 0xffffu;
  if (lower > 0x8000u ||
      (lower == 0x8000u && (upper & 1u) != 0u)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_float(uint16_t value) noexcept {
  return bits_float(static_cast<uint32_t>(value) << 16);
}

bool valid_format(int format) noexcept {
  return format >= TLPU_PRECISION_INT4 && format <= TLPU_PRECISION_FP32;
}

bool is_integer_format(int format) noexcept {
  return format == TLPU_PRECISION_INT4 || format == TLPU_PRECISION_INT8;
}

bool valid_integer_inputs(const float *values, std::size_t count,
                          int format) noexcept {
  const float minimum = format == TLPU_PRECISION_INT4 ? -8.0f : -128.0f;
  const float maximum = format == TLPU_PRECISION_INT4 ? 7.0f : 127.0f;
  for (std::size_t index = 0; index < count; ++index) {
    const float value = values[index];
    if (!std::isfinite(value) || std::trunc(value) != value ||
        value < minimum || value > maximum) {
      return false;
    }
  }
  return true;
}

int64_t wrap_signed(int64_t value, int bits) noexcept {
  const uint64_t modulus = uint64_t{1} << bits;
  const uint64_t masked = static_cast<uint64_t>(value) & (modulus - 1u);
  if ((masked & (uint64_t{1} << (bits - 1))) != 0u) {
    return static_cast<int64_t>(masked) - static_cast<int64_t>(modulus);
  }
  return static_cast<int64_t>(masked);
}

float quantize_input(float value, int format) noexcept {
  if (format == TLPU_PRECISION_FP16) {
    return fp16_to_float(float_to_fp16(value));
  }
  if (format == TLPU_PRECISION_BF16) {
    return bf16_to_float(float_to_bf16(value));
  }
  return value;
}

bool validate_gemv_arguments(int format, int accumulator_bits,
                             const float *a, const float *w,
                             std::size_t rows, std::size_t k, double *out,
                             uint64_t *overflow_events,
                             uint64_t *nonfinite_products,
                             uint64_t *nonfinite_accumulators) noexcept {
  if (!valid_format(format)) {
    set_error("format must be INT4, INT8, FP16, BF16, or FP32");
    return false;
  }
  if (rows == 0u || rows > kMaxRows || k == 0u || k > kMaxK) {
    set_error("rows must be 1..64 and k must be 1..4096");
    return false;
  }
  if (a == nullptr || w == nullptr || out == nullptr ||
      overflow_events == nullptr || nonfinite_products == nullptr ||
      nonfinite_accumulators == nullptr) {
    set_error("gemv pointers must be non-null");
    return false;
  }
  if (is_integer_format(format)) {
    if (accumulator_bits != 16 && accumulator_bits != 32) {
      set_error("integer accumulator_bits must be 16 or 32");
      return false;
    }
  } else if (accumulator_bits != 32) {
    set_error("floating-point accumulator_bits must be 32");
    return false;
  }
  if (rows > std::numeric_limits<std::size_t>::max() / k) {
    set_error("rows*k overflows size_t");
    return false;
  }
  return true;
}

}  // namespace

extern "C" int tlpu_precision_gemv(
    int format, int accumulator_bits, const float *a, const float *w,
    std::size_t rows, std::size_t k, double *out, uint64_t *overflow_events,
    uint64_t *nonfinite_products, uint64_t *nonfinite_accumulators) {
  clear_error();
  try {
    if (!validate_gemv_arguments(
            format, accumulator_bits, a, w, rows, k, out, overflow_events,
            nonfinite_products, nonfinite_accumulators)) {
      return -1;
    }

    const std::size_t weight_count = rows * k;
    if (is_integer_format(format) &&
        (!valid_integer_inputs(a, k, format) ||
         !valid_integer_inputs(w, weight_count, format))) {
      set_error("integer inputs must be finite integral values in format range");
      return -1;
    }

    *overflow_events = 0u;
    *nonfinite_products = 0u;
    *nonfinite_accumulators = 0u;

    if (is_integer_format(format)) {
      const int64_t minimum =
          -(int64_t{1} << (accumulator_bits - 1));
      const int64_t maximum = (int64_t{1} << (accumulator_bits - 1)) - 1;
      for (std::size_t row = 0; row < rows; ++row) {
        int64_t accumulator = 0;
        for (std::size_t index = 0; index < k; ++index) {
          const int64_t left = static_cast<int64_t>(a[index]);
          const int64_t right = static_cast<int64_t>(w[row * k + index]);
          const int64_t product = left * right;
          const int64_t wide_sum = accumulator + product;
          if (wide_sum < minimum || wide_sum > maximum) {
            ++(*overflow_events);
          }
          accumulator = wrap_signed(wide_sum, accumulator_bits);
        }
        out[row] = static_cast<double>(accumulator);
      }
      return 0;
    }

    for (std::size_t row = 0; row < rows; ++row) {
      float accumulator = 0.0f;
      for (std::size_t index = 0; index < k; ++index) {
        const float left = quantize_input(a[index], format);
        const float right = quantize_input(w[row * k + index], format);
        const float product = left * right;
        if (!std::isfinite(product)) {
          ++(*nonfinite_products);
        }
        // Keep this as an explicit multiply followed by add.  The target is
        // built with -ffp-contract=off so this cannot become an FMA.
        accumulator = product + accumulator;
        if (!std::isfinite(accumulator)) {
          ++(*nonfinite_accumulators);
        }
      }
      out[row] = static_cast<double>(accumulator);
    }
    return 0;
  } catch (...) {
    set_error("unexpected exception in tlpu_precision_gemv");
    return -1;
  }
}

extern "C" int tlpu_precision_convert(int format, const float *src,
                                        std::size_t n, float *dst,
                                        uint16_t *bits) {
  clear_error();
  try {
    if (format != TLPU_PRECISION_FP16 && format != TLPU_PRECISION_BF16) {
      set_error("convert format must be FP16 or BF16");
      return -1;
    }
    if (n > kMaxConversionElements) {
      set_error("conversion element count exceeds safety limit");
      return -1;
    }
    if (n != 0u && (src == nullptr || dst == nullptr || bits == nullptr)) {
      set_error("convert pointers must be non-null when n is nonzero");
      return -1;
    }
    for (std::size_t index = 0; index < n; ++index) {
      const uint16_t encoded =
          format == TLPU_PRECISION_FP16 ? float_to_fp16(src[index])
                                        : float_to_bf16(src[index]);
      bits[index] = encoded;
      dst[index] = format == TLPU_PRECISION_FP16 ? fp16_to_float(encoded)
                                                 : bf16_to_float(encoded);
    }
    return 0;
  } catch (...) {
    set_error("unexpected exception in tlpu_precision_convert");
    return -1;
  }
}

extern "C" const char *tlpu_precision_last_error(void) {
  return g_last_error;
}
