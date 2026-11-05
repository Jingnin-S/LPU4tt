#include "tinylpu_plan_c.h"

#include "tinylpu_plan.hpp"

#include <exception>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

struct tlpu_plan_handle {
  tinylpu_plan::Program program;
  std::string report;
  std::string error;
};

namespace {

thread_local std::string g_last_error;

const char* exception_message() noexcept {
  try {
    throw;
  } catch (const std::exception& error) {
    return error.what();
  } catch (...) {
    return "unknown C++ exception";
  }
}

void set_thread_error(const char* message) noexcept {
  try {
    g_last_error = message == nullptr ? "unknown error" : message;
  } catch (...) {
    // Avoid allowing an allocation failure while recording an error to cross
    // the C boundary.  An empty message is still a valid failure indication.
    g_last_error.clear();
  }
}

void set_handle_error(tlpu_plan_handle* handle, const char* message) noexcept {
  if (handle == nullptr) {
    set_thread_error(message);
    return;
  }
  try {
    handle->error = message == nullptr ? "unknown error" : message;
  } catch (...) {
    handle->error.clear();
  }
}

}  // namespace

extern "C" void* tlpu_plan_create(const char* program_path,
                                    const char* weights_path) {
  try {
    if (program_path == nullptr || weights_path == nullptr) {
      set_thread_error("program_path and weights_path must be non-null");
      return nullptr;
    }
    auto program = tinylpu_plan::load_program(program_path, weights_path);
    auto* handle = new tlpu_plan_handle();
    handle->program = std::move(program);
    handle->error.clear();
    g_last_error.clear();
    return handle;
  } catch (...) {
    set_thread_error(exception_message());
    return nullptr;
  }
}

extern "C" const char* tlpu_plan_run(void* opaque_handle,
                                       const int32_t* input,
                                       size_t input_count) {
  auto* handle = static_cast<tlpu_plan_handle*>(opaque_handle);
  if (handle == nullptr) {
    set_thread_error("handle must be non-null");
    return nullptr;
  }

  // Invalidate the prior report before validating or simulating this call.
  // This makes the returned pointer lifetime explicit even on failure.
  handle->report.clear();
  handle->error.clear();

  try {
    if (input_count != handle->program.header.input_elements) {
      throw std::invalid_argument("input_count does not match program input_elements");
    }
    if (input == nullptr && input_count != 0) {
      throw std::invalid_argument("input must be non-null when input_count is non-zero");
    }

    std::vector<int> values;
    values.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index) {
      values.push_back(static_cast<int>(input[index]));
    }

    // simulate() creates fresh execution state on every invocation.  The
    // loaded program is immutable after create(), so repeated runs are clean
    // and do not reuse scratchpad, validity bits, or result tensors.
    const auto report = tinylpu_plan::simulate(handle->program, values);
    handle->report = tinylpu_plan::report_json(report);
    return handle->report.c_str();
  } catch (...) {
    set_handle_error(handle, exception_message());
    return nullptr;
  }
}

extern "C" const char* tlpu_plan_last_error(void* opaque_handle) {
  if (opaque_handle == nullptr) {
    return g_last_error.c_str();
  }
  auto* handle = static_cast<tlpu_plan_handle*>(opaque_handle);
  return handle->error.c_str();
}

extern "C" void tlpu_plan_destroy(void* opaque_handle) {
  auto* handle = static_cast<tlpu_plan_handle*>(opaque_handle);
  delete handle;
}

