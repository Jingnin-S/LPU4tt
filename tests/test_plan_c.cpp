#include "tinylpu_plan_c.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_u32_le(std::ofstream& stream, std::uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte) {
    stream.put(static_cast<char>((value >> (8u * byte)) & 0xffu));
  }
}

void write_fixture(const std::filesystem::path& program_path,
                   const std::filesystem::path& weights_path) {
  std::ofstream program(program_path, std::ios::binary | std::ios::trunc);
  assert(program);
  const char magic[] = "TLPUv001";
  program.write(magic, 8);

  // ABI v1: lanes=2, depth=3, A/B ports=1/2, 8-byte scratchpad,
  // runtime tensor0 input and one INT16 output tensor.
  const std::vector<std::uint32_t> header = {
      1, 2, 3, 1, 2, 8, 2, 4, 1, 1, 6, 2};
  for (const std::uint32_t value : header) {
    write_u32_le(program, value);
  }

  const std::vector<std::uint32_t> instructions = {
      // ALLOC_TENSOR id=1, count=1, bits=16
      1, 1, 1, 16, 0, 0, 0, 0,
      // LOAD_W blob_offset=0, byte_count=2, scratch address=0
      3, 0, 2, 0, 0, 0, 0, 0,
      // LOAD_A tensor=0, offset=0, count=4, scratch address=2
      2, 0, 0, 4, 2, 0, 0, 0,
      // GEMV A=2, W=0, Y=4, rows=1, K=4
      4, 2, 0, 4, 1, 4, 0, 0,
      // STORE_Y Y=4, tensor=1, offset=0, count=1
      5, 4, 1, 0, 1, 0, 0, 0,
      // END output tensor=1
      7, 1, 0, 0, 0, 0, 0, 0};
  for (const std::uint32_t value : instructions) {
    write_u32_le(program, value);
  }
  program.close();
  assert(program);

  std::ofstream weights(weights_path, std::ios::binary | std::ios::trunc);
  assert(weights);
  // INT4 weights [1, -2, 3, 4], low nibble first.
  weights.put(static_cast<char>(0xe1));
  weights.put(static_cast<char>(0x43));
  weights.close();
  assert(weights);
}

void require_json(const char* json, const std::string& fragment) {
  assert(json != nullptr);
  assert(std::string(json).find(fragment) != std::string::npos);
}

void require_error(void* handle, const std::string& fragment) {
  const char* error = tlpu_plan_last_error(handle);
  assert(error != nullptr);
  assert(std::string(error).find(fragment) != std::string::npos);
}

}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    "tinylpu_plan_c_binding_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const auto program_path = root / "program.tlp";
  const auto weights_path = root / "weights.bin";
  write_fixture(program_path, weights_path);

  // NULL create/run paths must fail through the C boundary, never throw.
  assert(tlpu_plan_create(nullptr, weights_path.string().c_str()) == nullptr);
  require_error(nullptr, "non-null");
  assert(tlpu_plan_run(nullptr, nullptr, 0) == nullptr);
  require_error(nullptr, "handle");
  tlpu_plan_destroy(nullptr);

  const auto missing = root / "missing.tlp";
  assert(tlpu_plan_create(missing.string().c_str(), weights_path.string().c_str()) ==
         nullptr);
  require_error(nullptr, "cannot open");

  void* handle = tlpu_plan_create(program_path.string().c_str(),
                                  weights_path.string().c_str());
  assert(handle != nullptr);
  assert(std::string(tlpu_plan_last_error(handle)).empty());

  const std::vector<std::int32_t> first = {1, 2, 3, 4};
  const std::vector<std::int32_t> second = {-8, 7, 0, 1};

  // A bad call invalidates the prior report but must not poison the handle.
  assert(tlpu_plan_run(handle, nullptr, first.size()) == nullptr);
  require_error(handle, "input must be non-null");
  assert(tlpu_plan_run(handle, first.data(), first.size()) != nullptr);
  require_json(tlpu_plan_run(handle, first.data(), first.size()),
               "\"result\":[22]");
  assert(std::string(tlpu_plan_last_error(handle)).empty());

  // Different input proves a fresh simulation; repeating the first input
  // proves that the loaded program can be reused after a prior result.
  require_json(tlpu_plan_run(handle, second.data(), second.size()),
               "\"result\":[-18]");
  require_json(tlpu_plan_run(handle, first.data(), first.size()),
               "\"result\":[22]");

  assert(tlpu_plan_run(handle, first.data(), first.size() - 1) == nullptr);
  require_error(handle, "input_count");
  require_json(tlpu_plan_run(handle, first.data(), first.size()),
               "\"result\":[22]");

  tlpu_plan_destroy(handle);
  std::filesystem::remove_all(root);
  std::cout << "plan C binding tests passed\n";
  return 0;
}

