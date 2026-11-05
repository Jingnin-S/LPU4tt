#ifndef TINY_LPU_PRECISION_H
#define TINY_LPU_PRECISION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Input/storage format selectors accepted by the C API. */
enum tlpu_precision_format {
  TLPU_PRECISION_INT4 = 0,
  TLPU_PRECISION_INT8 = 1,
  TLPU_PRECISION_FP16 = 2,
  TLPU_PRECISION_BF16 = 3,
  TLPU_PRECISION_FP32 = 4
};

/*
 * Compute rows independent dot products.  `a` has k elements and `w` is a
 * row-major rows-by-k matrix.  Integer inputs are supplied as exactly
 * representable float values so this ABI can be consumed by ctypes without
 * a second integer-pointer entry point.  Integer accumulators wrap as signed
 * two's-complement values; overflow_events counts each pre-wrap addition.
 * Floating-point products and additions are float32 operations (no FMA).
 * All output and diagnostic pointers must be non-null.  The caller owns all
 * pointed-to storage.  Returns 0 on success and -1 on validation/internal
 * failure; use tlpu_precision_last_error() for the thread-local message.
 */
int tlpu_precision_gemv(
    int format,
    int accumulator_bits,
    const float *a,
    const float *w,
    size_t rows,
    size_t k,
    double *out,
    uint64_t *overflow_events,
    uint64_t *nonfinite_products,
    uint64_t *nonfinite_accumulators);

/*
 * Convert FP32 source values to IEEE binary16 or bfloat16 using round-to-
 * nearest-even.  `dst` receives the decoded FP32 value represented by the
 * 16-bit result and `bits` receives its wire encoding.  Returns 0 on success.
 */
int tlpu_precision_convert(
    int format,
    const float *src,
    size_t n,
    float *dst,
    uint16_t *bits);

/* Empty after a successful call; otherwise a thread-local diagnostic string. */
const char *tlpu_precision_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
