#include "tinylpu_precision.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int64_t wrap_signed(int64_t value, int bits) {
  const uint64_t modulus = uint64_t{1} << bits;
  const uint64_t masked = static_cast<uint64_t>(value) & (modulus - 1u);
  if ((masked & (uint64_t{1} << (bits - 1))) != 0u) {
    return static_cast<int64_t>(masked) - static_cast<int64_t>(modulus);
  }
  return static_cast<int64_t>(masked);
}

struct IntegerOracle {
  int64_t value = 0;
  uint64_t overflow_events = 0;
};

IntegerOracle integer_oracle(const std::vector<float> &a,
                             const std::vector<float> &w, int bits) {
  IntegerOracle result;
  const int64_t minimum = -(int64_t{1} << (bits - 1));
  const int64_t maximum = (int64_t{1} << (bits - 1)) - 1;
  for (std::size_t index = 0; index < a.size(); ++index) {
    const int64_t wide = result.value +
                         static_cast<int64_t>(a[index]) *
                             static_cast<int64_t>(w[index]);
    if (wide < minimum || wide > maximum) {
      ++result.overflow_events;
    }
    result.value = wrap_signed(wide, bits);
  }
  return result;
}

void test_integer_gemv() {
  const float a[] = {1.0f, -2.0f, 3.0f, 7.0f};
  const float w[] = {1.0f, 2.0f, -1.0f, 4.0f,
                     -8.0f, 7.0f, 1.0f, -2.0f};
  double out[2] = {0.0, 0.0};
  uint64_t overflow = 99;
  uint64_t nonfinite_products = 99;
  uint64_t nonfinite_accumulators = 99;
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 2, 4, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) == 0,
          "INT4 ACC16 gemv succeeds");
  require(out[0] == 22.0 && out[1] == -33.0,
          "INT4 ACC16 results are exact signed sums");
  require(overflow == 0 && nonfinite_products == 0 &&
              nonfinite_accumulators == 0,
          "integer diagnostics are zero without overflow");

  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 32, a, w, 2, 4, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) == 0,
          "INT4 ACC32 gemv succeeds");
  require(out[0] == 22.0 && out[1] == -33.0,
          "INT4 ACC32 agrees with ACC16 in-range result");

  const std::size_t k = 4096;
  std::vector<float> int4_a(k, 7.0f);
  std::vector<float> int4_w(k, 7.0f);
  const IntegerOracle int4_expected = integer_oracle(int4_a, int4_w, 16);
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, int4_a.data(),
                              int4_w.data(), 1, k, out, &overflow,
                              &nonfinite_products, &nonfinite_accumulators) ==
              0,
          "INT4 ACC16 wrapping gemv succeeds");
  require(out[0] == static_cast<double>(int4_expected.value) &&
              overflow == int4_expected.overflow_events && overflow > 0,
          "INT4 ACC16 wrap and every pre-wrap event match oracle");

  std::vector<float> int8_a(k, 127.0f);
  std::vector<float> int8_w(k, 127.0f);
  const IntegerOracle int8_expected = integer_oracle(int8_a, int8_w, 16);
  require(tlpu_precision_gemv(TLPU_PRECISION_INT8, 16, int8_a.data(),
                              int8_w.data(), 1, k, out, &overflow,
                              &nonfinite_products, &nonfinite_accumulators) ==
              0,
          "INT8 ACC16 wrapping gemv succeeds");
  require(out[0] == static_cast<double>(int8_expected.value) &&
              overflow == int8_expected.overflow_events && overflow > 0,
          "INT8 ACC16 wrap and diagnostics match oracle");

  require(tlpu_precision_gemv(TLPU_PRECISION_INT8, 32, int8_a.data(),
                              int8_w.data(), 1, k, out, &overflow,
                              &nonfinite_products, &nonfinite_accumulators) ==
              0,
          "INT8 ACC32 succeeds");
  require(overflow == 0 && out[0] == 127.0 * 127.0 * 4096.0,
          "INT8 ACC32 has no overflow for the bounded experiment");
}

void test_integer_validation_and_recovery() {
  float a[] = {1.0f, -2.0f};
  float w[] = {3.0f, 4.0f};
  double out[1] = {0.0};
  uint64_t overflow = 0;
  uint64_t nonfinite_products = 0;
  uint64_t nonfinite_accumulators = 0;

  a[0] = 1.5f;
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 1, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "non-integral INT4 input is rejected");
  require(std::strlen(tlpu_precision_last_error()) != 0,
          "invalid integer input records an error");

  a[0] = 1.0f;
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 1, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) == 0,
          "valid call succeeds after invalid input");
  require(std::strlen(tlpu_precision_last_error()) == 0 && out[0] == -5.0,
          "successful recovery clears the error");

  a[0] = 8.0f;
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 1, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "out-of-range INT4 input is rejected");
  a[0] = 1.0f;
  require(tlpu_precision_gemv(TLPU_PRECISION_INT8, 16, a, w, 1, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) == 0,
          "INT8 accepts the same valid values after error");

  require(tlpu_precision_gemv(99, 16, a, w, 1, 2, out, &overflow,
                              &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "unknown format is rejected");
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 0, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "zero rows are rejected");
  require(tlpu_precision_gemv(TLPU_PRECISION_INT4, 16, a, w, 1, 2, out,
                              nullptr, &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "null diagnostics pointer is rejected");
}

void check_conversion(int format, const std::vector<float> &source,
                      const std::vector<uint16_t> &expected,
                      const std::string &label) {
  std::vector<float> decoded(source.size(), 0.0f);
  std::vector<uint16_t> bits(source.size(), 0u);
  require(tlpu_precision_convert(format, source.data(), source.size(),
                                 decoded.data(), bits.data()) == 0,
          label + " conversion succeeds");
  for (std::size_t index = 0; index < source.size(); ++index) {
    require(bits[index] == expected[index], label + " encoded bits");
  }
}

void test_fp16_conversion() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  uint16_t expected_nan = static_cast<uint16_t>(float_bits(nan) >> 13);
  if ((expected_nan & 0x03ffu) == 0u) {
    expected_nan = 1u;
  }
  expected_nan = static_cast<uint16_t>(0x7c00u | (expected_nan & 0x03ffu));
  const std::vector<float> source = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      std::ldexp(1.0f, -24),
      3.0f * std::ldexp(1.0f, -25),
      std::ldexp(1.0f, -14),
      65504.0f,
      70000.0f,
      std::numeric_limits<float>::infinity(),
      nan,
  };
  check_conversion(
      TLPU_PRECISION_FP16, source,
      {0x0000u, 0x8000u, 0x3c00u, 0xbc00u, 0x0001u, 0x0002u, 0x0400u,
       0x7bffu, 0x7c00u, 0x7c00u, expected_nan},
      "FP16");

  std::vector<float> decoded(source.size(), 0.0f);
  std::vector<uint16_t> bits(source.size(), 0u);
  require(tlpu_precision_convert(TLPU_PRECISION_FP16, source.data(),
                                 source.size(), decoded.data(), bits.data()) ==
              0,
          "FP16 decoded output succeeds");
  require(float_bits(decoded[0]) == 0x00000000u &&
              float_bits(decoded[1]) == 0x80000000u,
          "FP16 preserves positive and negative zero");
  require(float_bits(decoded[2]) == float_bits(1.0f) &&
              float_bits(decoded[7]) == float_bits(65504.0f),
          "FP16 decoded normals are exact");
  require(std::isinf(decoded[8]) && std::isinf(decoded[9]) &&
              std::isnan(decoded[10]),
          "FP16 decoded overflow, infinity, and NaN are classified");
}

void test_bf16_conversion() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  uint16_t expected_nan = static_cast<uint16_t>(float_bits(nan) >> 16);
  if ((expected_nan & 0x007fu) == 0u) {
    expected_nan = static_cast<uint16_t>(expected_nan | 1u);
  }
  const std::vector<float> source = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      1.0f + std::ldexp(1.0f, -8),
      1.0f + 3.0f * std::ldexp(1.0f, -9),
      std::ldexp(1.0f, -134),
      3.0f * std::ldexp(1.0f, -134),
      std::ldexp(1.0f, -126),
      std::numeric_limits<float>::infinity(),
      nan,
  };
  check_conversion(
      TLPU_PRECISION_BF16, source,
      {0x0000u, 0x8000u, 0x3f80u, 0xbf80u, 0x3f80u, 0x3f81u, 0x0000u,
       0x0002u, 0x0080u, 0x7f80u, expected_nan},
      "BF16");

  std::vector<float> decoded(source.size(), 0.0f);
  std::vector<uint16_t> bits(source.size(), 0u);
  require(tlpu_precision_convert(TLPU_PRECISION_BF16, source.data(),
                                 source.size(), decoded.data(), bits.data()) ==
              0,
          "BF16 decoded output succeeds");
  require(float_bits(decoded[0]) == 0x00000000u &&
              float_bits(decoded[1]) == 0x80000000u,
          "BF16 preserves positive and negative zero");
  require(float_bits(decoded[8]) == float_bits(std::ldexp(1.0f, -126)),
          "BF16 preserves the minimum normal");
  require(std::isinf(decoded[9]) && std::isnan(decoded[10]),
          "BF16 decoded infinity and NaN are classified");
}

void test_floating_gemv() {
  const float a[] = {1.5f, -2.0f};
  const float w[] = {2.0f, 4.0f};
  double out[1] = {0.0};
  uint64_t overflow = 0;
  uint64_t nonfinite_products = 0;
  uint64_t nonfinite_accumulators = 0;

  for (int format : {TLPU_PRECISION_FP16, TLPU_PRECISION_BF16,
                     TLPU_PRECISION_FP32}) {
    require(tlpu_precision_gemv(
                format, 32, a, w, 1, 2, out, &overflow, &nonfinite_products,
                &nonfinite_accumulators) == 0,
            "finite floating gemv succeeds");
    require(out[0] == -5.0 && overflow == 0 && nonfinite_products == 0 &&
                nonfinite_accumulators == 0,
            "finite floating result and diagnostics are exact");
  }

  const float nonfinite_a[] = {std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()};
  const float finite_w[] = {1.0f, 1.0f};
  require(tlpu_precision_gemv(
              TLPU_PRECISION_FP16, 32, nonfinite_a, finite_w, 1, 2, out,
              &overflow, &nonfinite_products, &nonfinite_accumulators) == 0,
          "nonfinite FP16 gemv succeeds diagnostically");
  require(std::isnan(out[0]) && overflow == 0 && nonfinite_products == 2 &&
              nonfinite_accumulators == 2,
          "nonfinite product and accumulator counts are per term");

  require(tlpu_precision_gemv(TLPU_PRECISION_FP16, 16, a, w, 1, 2, out,
                              &overflow, &nonfinite_products,
                              &nonfinite_accumulators) != 0,
          "floating accumulator width other than 32 is rejected");

  std::vector<float> empty;
  require(tlpu_precision_convert(TLPU_PRECISION_FP16, nullptr, 0, nullptr,
                                 nullptr) == 0,
          "zero-length conversion is a no-op");
  require(tlpu_precision_convert(TLPU_PRECISION_FP32, empty.data(), 0,
                                 empty.data(), nullptr) != 0,
          "conversion rejects unsupported format even for zero length");
}

}  // namespace

int main() {
  test_integer_gemv();
  test_integer_validation_and_recovery();
  test_fp16_conversion();
  test_bf16_conversion();
  test_floating_gemv();
  std::cout << "precision tests passed\n";
  return 0;
}
