#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque process-local binding for the Tiny-LPU plan interpreter.
 *
 * A handle is owned by the creating thread and is not safe for concurrent
 * calls.  The program and weights are loaded once by tlpu_plan_create().
 * Each successful tlpu_plan_run() starts a fresh simulation from the loaded
 * immutable program, so a previous input/result cannot affect the next run.
 * The returned UTF-8 JSON pointer belongs to the handle and remains valid
 * until the next run (successful or failed) or tlpu_plan_destroy().
 *
 * All C++ exceptions are contained by the implementation.  On failure,
 * create/run return NULL and tlpu_plan_last_error() exposes the message.
 * Passing NULL to tlpu_plan_last_error() reads the thread-local error from
 * the most recent failed create (or a NULL-handle call).
 */
typedef struct tlpu_plan_handle tlpu_plan_handle;

void* tlpu_plan_create(const char* program_path, const char* weights_path);

const char* tlpu_plan_run(void* handle,
                          const int32_t* input,
                          size_t input_count);

const char* tlpu_plan_last_error(void* handle);

void tlpu_plan_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

