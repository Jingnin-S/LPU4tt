#ifndef TINY_LPU_ARCHITECTURE_H
#define TINY_LPU_ARCHITECTURE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Analyze one ISA byte stream as a conservative resource schedule.  This
 * interface intentionally does not execute LOAD payloads or initialize
 * scratchpad state; numerical execution and initialization checks remain the
 * responsibility of tlpu_isa_run().  The returned JSON pointer is owned by
 * the calling thread and is invalidated by the next call on that thread.
 * NULL indicates rejection and tlpu_arch_last_error() reports the reason.
 */
const char *tlpu_arch_analyze(
    const unsigned char *commands, size_t command_bytes,
    unsigned encoding, unsigned lanes, unsigned depth, unsigned acc_bits,
    unsigned banks, unsigned flat_read_ports, unsigned read_latency,
    unsigned copy_bytes_per_cycle);

const char *tlpu_arch_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
