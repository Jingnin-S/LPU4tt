#ifndef TINY_LPU_ISA_H
#define TINY_LPU_ISA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Execute one independent Tiny-LPU device byte stream.  The command stream
 * contains LOAD/STORE/MVM/END commands and `loads` contains the literals
 * consumed by LOAD commands in command order.  The returned JSON pointer is
 * thread-local and remains valid until the next call on the same thread.
 * On failure the return value is NULL and tlpu_isa_last_error() describes the
 * validation/execution error.  No exception crosses this C ABI.
 */
const char *tlpu_isa_run(
    const unsigned char *commands,
    size_t command_bytes,
    const unsigned char *loads,
    size_t load_bytes,
    unsigned scratch_bytes,
    unsigned lanes,
    unsigned depth,
    unsigned accumulator_bits,
    unsigned memory_byte_bw,
    unsigned encoding,
    unsigned enable_int8);

const char *tlpu_isa_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
