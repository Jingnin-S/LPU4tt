#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Cycle model for the synthesizable INT4/ACC32 byte-stream device.
 * Input: RX byte[7:0], rx_valid8, tx_ready9, start10, rst_n11, ena12.
 * Output after rising edge: uo_out[7:0], uio_out[15:8], uio_oe[23:16].
 * Events describe the just-sampled edge: RX accepted bit0, TX accepted bit1,
 * accepted TX byte[15:8]. Reset/start abort has priority over transfers.
 */
void *tlpu_device_create(unsigned capacity, unsigned lanes);
uint32_t tlpu_device_tick(void *device, uint32_t inputs);
uint32_t tlpu_device_peek(void *device, uint32_t inputs);
uint32_t tlpu_device_events(void *device);
const char *tlpu_device_debug(void *device);
const char *tlpu_device_last_error(void);
void tlpu_device_destroy(void *device);
#ifdef __cplusplus
}
#endif
