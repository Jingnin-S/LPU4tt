#include "tinylpu_isa.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

std::uint64_t json_uint(const std::string &json, const std::string &key) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t start = json.find(needle);
  expect(start != std::string::npos, "JSON contains " + key);
  std::size_t end = start + needle.size();
  while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
    ++end;
  }
  expect(end > start + needle.size(), "JSON number for " + key);
  return std::stoull(json.substr(start + needle.size(),
                                 end - start - needle.size()));
}

std::string json_string(const std::string &json, const std::string &key) {
  const std::string needle = "\"" + key + "\":\"";
  const std::size_t start = json.find(needle);
  expect(start != std::string::npos, "JSON contains string " + key);
  const std::size_t value_start = start + needle.size();
  const std::size_t end = json.find('"', value_start);
  expect(end != std::string::npos, "JSON string terminates for " + key);
  return json.substr(value_start, end - value_start);
}

void append_load(std::vector<unsigned char> &stream, unsigned encoding,
                 unsigned address, unsigned count) {
  expect(count >= 1u && count <= 256u, "test LOAD count range");
  stream.push_back(1u);
  stream.push_back(0u);
  stream.push_back(static_cast<unsigned char>(address));
  stream.push_back(static_cast<unsigned char>(count - 1u));
  if (encoding == 8u) {
    stream.insert(stream.end(), 4u, 0u);
  }
}

void append_store(std::vector<unsigned char> &stream, unsigned encoding,
                  unsigned address, unsigned count) {
  expect(count >= 1u && count <= 256u, "test STORE count range");
  stream.push_back(2u);
  stream.push_back(0u);
  stream.push_back(static_cast<unsigned char>(address));
  stream.push_back(static_cast<unsigned char>(count - 1u));
  if (encoding == 8u) {
    stream.insert(stream.end(), 4u, 0u);
  }
}

void append_mvm(std::vector<unsigned char> &stream, unsigned flags,
                unsigned a, unsigned w, unsigned y, unsigned rows,
                unsigned k, unsigned stride) {
  expect(rows >= 1u && rows <= 16u && k >= 1u && k <= 64u && stride <= 255u,
         "test MVM field range");
  stream.push_back(3u);
  stream.push_back(static_cast<unsigned char>(flags));
  stream.push_back(static_cast<unsigned char>(a));
  stream.push_back(static_cast<unsigned char>(w));
  stream.push_back(static_cast<unsigned char>(y));
  stream.push_back(static_cast<unsigned char>(rows - 1u));
  stream.push_back(static_cast<unsigned char>(k - 1u));
  stream.push_back(static_cast<unsigned char>(stride));
}

void append_end(std::vector<unsigned char> &stream, unsigned encoding) {
  stream.push_back(15u);
  stream.insert(stream.end(), encoding - 1u, 0u);
}

std::vector<unsigned char> pack4(std::initializer_list<int> values) {
  std::vector<unsigned char> result((values.size() + 1u) / 2u, 0u);
  std::size_t index = 0;
  for (int value : values) {
    expect(value >= -8 && value <= 7, "test INT4 range");
    result[index / 2u] |= static_cast<unsigned char>(value & 0x0f)
                           << (4u * (index % 2u));
    ++index;
  }
  return result;
}

std::vector<unsigned char> pack4_repeated(int value, unsigned count) {
  expect(value >= -8 && value <= 7, "repeated INT4 range");
  std::vector<unsigned char> result((count + 1u) / 2u, 0u);
  const unsigned char nibble = static_cast<unsigned char>(value & 0x0f);
  for (unsigned index = 0; index < count; ++index) {
    result[index / 2u] |= static_cast<unsigned char>(nibble <<
                                                     (4u * (index % 2u)));
  }
  return result;
}

std::string run(const std::vector<unsigned char> &commands,
                const std::vector<unsigned char> &loads, unsigned scratch = 64,
                unsigned lanes = 2, unsigned depth = 3,
                unsigned accumulator_bits = 32, unsigned memory_bw = 2,
                unsigned encoding = 4, unsigned enable_int8 = 0) {
  const char *result = tlpu_isa_run(
      commands.empty() ? nullptr : commands.data(), commands.size(),
      loads.empty() ? nullptr : loads.data(), loads.size(), scratch, lanes,
      depth, accumulator_bits, memory_bw, encoding, enable_int8);
  expect(result != nullptr, std::string("ISA run succeeds: ") +
                              tlpu_isa_last_error());
  return std::string(result);
}

void expect_error(const std::vector<unsigned char> &commands,
                  const std::vector<unsigned char> &loads,
                  const std::string &label, unsigned scratch = 64,
                  unsigned lanes = 2, unsigned depth = 3,
                  unsigned accumulator_bits = 32, unsigned memory_bw = 2,
                  unsigned encoding = 4, unsigned enable_int8 = 0) {
  const char *result = tlpu_isa_run(
      commands.empty() ? nullptr : commands.data(), commands.size(),
      loads.empty() ? nullptr : loads.data(), loads.size(), scratch, lanes,
      depth, accumulator_bits, memory_bw, encoding, enable_int8);
  expect(result == nullptr, label + " is rejected");
  expect(std::string(tlpu_isa_last_error()).size() != 0,
         label + " records last_error");
}

void test_fixed_and_mixed_encoding() {
  std::vector<unsigned char> loads = pack4({1, 2, 3, 4});
  const std::vector<unsigned char> weights = pack4({1, -1, 2, -2});
  loads.insert(loads.end(), weights.begin(), weights.end());

  for (unsigned encoding : {4u, 8u}) {
    std::vector<unsigned char> commands;
    append_load(commands, encoding, 0, 2);
    append_load(commands, encoding, 8, 2);
    append_mvm(commands, 0, 0, 8, 16, 1, 4, 2);
    append_store(commands, encoding, 16, 2);
    append_end(commands, encoding);
    const std::string report = run(commands, loads, 32, 2, 3, 16, 2,
                                   encoding, 0);
    expect(json_string(report, "stores_hex") == "fdff",
           "fixed/mixed output is signed little endian");
    expect(json_uint(report, "command_count") == 5 &&
               json_uint(report, "mvm_count") == 1 &&
               json_uint(report, "macs") == 4,
           "command and MAC counts");
    expect(json_uint(report, "command_bytes") == (encoding == 4 ? 24 : 40),
           "actual command byte count follows encoding");
    expect(json_uint(report, "load_bytes") == 4 &&
               json_uint(report, "store_bytes") == 2,
           "payload byte accounting");
    expect(json_uint(report, "core_cycles") == 6 &&
               json_uint(report, "scratch_cycles") == 4 &&
               json_uint(report, "decode_cycles") == 5 &&
               json_uint(report, "device_cycles") == 15,
           "fixed timing accounting");
    expect(json_uint(report, "peak_scratch_end") == 18,
           "peak scratch end tracks output range");
  }
}

void test_group_tail_and_accumulate() {
  // Three rows with two lanes exercises a full group followed by a tail.
  std::vector<unsigned char> commands;
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 3);
  append_mvm(commands, 0, 0, 8, 16, 3, 1, 1);
  append_store(commands, 4, 16, 6);
  append_end(commands, 4);
  const std::vector<unsigned char> loads = {0x02u, 0x01u, 0xffu, 0x03u};
  const std::string report = run(commands, loads, 32, 2, 3, 16);
  expect(json_string(report, "stores_hex") == "0200feff0600",
         "tail group writes each row result");
  expect(json_uint(report, "macs") == 3 && json_uint(report, "core_cycles") == 6,
         "tail group core cycles are explicitly drained");

  // Two K=1 MVMs reuse Y as a raw ACC16 partial sum.
  commands.clear();
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_mvm(commands, 0, 0, 8, 16, 1, 1, 1);
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_mvm(commands, 1, 0, 8, 16, 1, 1, 1);
  append_store(commands, 4, 16, 2);
  append_end(commands, 4);
  const std::vector<unsigned char> accum_loads = {0x02u, 0x03u, 0x04u, 0x03u};
  const std::string accum_report = run(commands, accum_loads, 32, 1, 2, 16);
  expect(json_string(accum_report, "stores_hex") == "1200",
         "ACCUM reads runtime partial sum and adds next product");
  expect(json_uint(accum_report, "mvm_count") == 2 &&
             json_uint(accum_report, "macs") == 2,
         "ACCUM command counts");
}

void test_sat16_and_int8_profile() {
  // Seed an ACC32 partial sum at 65536, then SAT16 it without computing a
  // precomputed result in the model.
  std::vector<unsigned char> commands;
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_load(commands, 4, 16, 4);
  append_mvm(commands, 3, 0, 8, 16, 1, 1, 1);
  append_store(commands, 4, 16, 2);
  append_end(commands, 4);
  const std::vector<unsigned char> loads = {0x00u, 0x00u, 0x00u,
                                            0x00u, 0x00u, 0x01u};
  const std::string report = run(commands, loads, 32, 1, 3, 32);
  expect(json_string(report, "stores_hex") == "ff7f" &&
             json_uint(report, "saturation_events") == 1,
         "SAT16 clamps an ACC32 result and counts changed output");

  // Format bit 1 (INT8) is rejected unless the profile explicitly enables it.
  commands.clear();
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_mvm(commands, 4, 0, 8, 16, 1, 1, 1);
  append_store(commands, 4, 16, 4);
  append_end(commands, 4);
  const std::vector<unsigned char> int8_loads = {2u, 3u};
  expect_error(commands, int8_loads, "disabled INT8 profile", 32, 1, 3, 32,
               2, 4, 0);
  const std::string int8_report = run(commands, int8_loads, 32, 1, 3, 32,
                                      2, 4, 1);
  expect(json_string(int8_report, "stores_hex") == "06000000" &&
             json_uint(int8_report, "macs") == 1,
         "enabled INT8 profile executes packed bytes");
}

void test_long_k_and_runtime_wrap() {
  const std::vector<unsigned char> a = pack4_repeated(-8, 64);
  const std::vector<unsigned char> w = pack4_repeated(-8, 64);
  std::vector<unsigned char> loads = a;
  loads.insert(loads.end(), w.begin(), w.end());
  std::vector<unsigned char> commands;
  append_load(commands, 4, 0, 32);
  append_load(commands, 4, 64, 32);
  append_mvm(commands, 0, 0, 64, 112, 1, 64, 32);
  append_store(commands, 4, 112, 2);
  append_end(commands, 4);
  const std::string report = run(commands, loads, 128, 2, 4, 16);
  expect(json_string(report, "stores_hex") == "0010" &&
             json_uint(report, "overflow_events") == 0 &&
             json_uint(report, "macs") == 64,
         "K64 INT4 raw ACC16 remains in range");
  expect(json_uint(report, "core_cycles") == 67,
         "K64 pipeline includes depth-1 drain");

  // Seeded ACCUM explicitly covers both signed ACC32 wrap boundaries.  This
  // is separate from the zero-initialized K<=4096 proof used by the driver.
  std::vector<unsigned char> seeded_commands;
  append_load(seeded_commands, 4, 0, 1);
  append_load(seeded_commands, 4, 8, 1);
  append_load(seeded_commands, 4, 16, 4);
  append_mvm(seeded_commands, 1, 0, 8, 16, 1, 1, 1);
  append_store(seeded_commands, 4, 16, 4);
  append_end(seeded_commands, 4);
  const std::vector<unsigned char> max_seed_loads = {
      0x01u, 0x01u, 0xffu, 0xffu, 0xffu, 0x7fu};
  const std::string max_seed_report =
      run(seeded_commands, max_seed_loads, 32, 1, 3, 32);
  expect(json_string(max_seed_report, "stores_hex") == "00000080" &&
             json_uint(max_seed_report, "overflow_events") == 1,
         "INT32_MAX plus one wraps to INT32_MIN");

  const std::vector<unsigned char> min_seed_loads = {
      0x0fu, 0x01u, 0x00u, 0x00u, 0x00u, 0x80u};
  const std::string min_seed_report =
      run(seeded_commands, min_seed_loads, 32, 1, 3, 32);
  expect(json_string(min_seed_report, "stores_hex") == "ffffff7f" &&
             json_uint(min_seed_report, "overflow_events") == 1,
         "INT32_MIN minus one wraps to INT32_MAX");
}

void test_bad_encoding_and_memory_validation() {
  std::vector<unsigned char> end4 = {15u, 0u, 0u, 0u};
  expect_error({}, {}, "empty command stream");
  expect_error({15u}, {}, "truncated END");
  expect_error({0u, 0u, 0u, 0u}, {}, "invalid opcode");
  expect_error({15u, 1u, 0u, 0u}, {}, "reserved END bit");
  expect_error({1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u}, {},
               "reserved LOAD bit", 64, 2, 3, 32, 2, 8, 0);
  expect_error({15u, 0u, 0u, 0u, 15u, 0u, 0u, 0u}, {},
               "duplicate END", 64, 2, 3, 32, 2, 8, 0);
  expect_error(end4, {0u}, "extra LOAD payload");

  std::vector<unsigned char> commands;
  append_load(commands, 4, 0, 2);
  append_end(commands, 4);
  expect_error(commands, {0u}, "missing LOAD literal");

  commands.clear();
  append_store(commands, 4, 0, 2);
  append_end(commands, 4);
  expect_error(commands, {}, "uninitialized STORE");

  commands.clear();
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_mvm(commands, 0, 0, 8, 0, 1, 1, 1);
  append_end(commands, 4);
  expect_error(commands, {1u, 2u}, "Y aliases A");

  commands.clear();
  append_load(commands, 4, 0, 1);
  append_load(commands, 4, 8, 1);
  append_mvm(commands, 0, 0, 8, 16, 1, 1, 0);
  append_end(commands, 4);
  expect_error(commands, {1u, 2u}, "zero W stride");

  // Configuration errors are rejected before stream execution.
  expect_error(end4, {}, "scratch lower bound", 31);
  expect_error(end4, {}, "lane profile", 64, 3);
  expect_error(end4, {}, "depth profile", 64, 2, 1);
  expect_error(end4, {}, "accumulator profile", 64, 2, 3, 24);
  expect_error(end4, {}, "memory bandwidth profile", 64, 2, 3, 32, 0);
  expect_error(end4, {}, "encoding profile", 64, 2, 3, 32, 2, 5);

  // A failed call must not poison a subsequent valid run.
  const std::string recovered = run(end4, {});
  expect(json_uint(recovered, "command_count") == 1,
         "valid END recovers after validation errors");
}

}  // namespace

int main() {
  test_fixed_and_mixed_encoding();
  test_group_tail_and_accumulate();
  test_sat16_and_int8_profile();
  test_long_k_and_runtime_wrap();
  test_bad_encoding_and_memory_validation();
  std::cout << "ISA tests passed\n";
  return 0;
}
