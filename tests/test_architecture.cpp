#include "tinylpu_architecture.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
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
  const std::size_t begin = start + needle.size();
  std::size_t end = begin;
  while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
    ++end;
  }
  expect(end > begin, "JSON number for " + key);
  return std::stoull(json.substr(begin, end - begin));
}

void append_load(std::vector<unsigned char> &stream, unsigned encoding,
                 unsigned address, unsigned count) {
  expect(count >= 1u && count <= 256u, "test LOAD count");
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
  expect(count >= 1u && count <= 256u, "test STORE count");
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
  expect(rows >= 1u && rows <= 16u && k >= 1u && k <= 64u &&
             stride <= 255u,
         "test MVM fields");
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

std::string run(const std::vector<unsigned char> &commands,
                unsigned encoding = 4u, unsigned lanes = 2u,
                unsigned depth = 3u, unsigned acc_bits = 16u,
                unsigned banks = 0u, unsigned flat_ports = 1u,
                unsigned latency = 0u, unsigned copy_bw = 2u) {
  const char *result = tlpu_arch_analyze(
      commands.data(), commands.size(), encoding, lanes, depth, acc_bits,
      banks, flat_ports, latency, copy_bw);
  expect(result != nullptr,
         std::string("architecture analysis succeeds: ") +
             tlpu_arch_last_error());
  return std::string(result);
}

void expect_error(const std::vector<unsigned char> &commands,
                  const std::string &label, unsigned encoding = 4u,
                  unsigned lanes = 2u, unsigned depth = 3u,
                  unsigned acc_bits = 16u, unsigned banks = 0u,
                  unsigned flat_ports = 1u, unsigned latency = 0u,
                  unsigned copy_bw = 2u) {
  const char *result = tlpu_arch_analyze(
      commands.empty() ? nullptr : commands.data(), commands.size(), encoding,
      lanes, depth, acc_bits, banks, flat_ports, latency, copy_bw);
  expect(result == nullptr, label + " is rejected");
  expect(std::string(tlpu_arch_last_error()).size() != 0,
         label + " records last_error");
}

void test_copy_and_encoding_counts() {
  std::vector<unsigned char> commands;
  append_load(commands, 4u, 0u, 3u);
  append_store(commands, 4u, 8u, 5u);
  append_end(commands, 4u);
  const std::string report = run(commands, 4u, 1u, 2u, 16u, 0u, 1u, 0u,
                                 2u);
  expect(json_uint(report, "command_count") == 3u &&
             json_uint(report, "command_bytes") == 12u,
         "mixed command count and bytes");
  expect(json_uint(report, "load_bytes") == 3u &&
             json_uint(report, "store_bytes") == 5u &&
             json_uint(report, "scratch_cycles") == 5u &&
             json_uint(report, "decode_cycles") == 3u &&
             json_uint(report, "device_cycles") == 8u,
         "LOAD/STORE copy accounting");
  expect(json_uint(report, "peak_scratch_end") == 13u,
         "copy ranges update peak");

  commands.clear();
  append_load(commands, 8u, 0u, 1u);
  append_end(commands, 8u);
  const std::string fixed = run(commands, 8u, 1u, 2u, 16u, 0u, 1u, 0u,
                                1u);
  expect(json_uint(fixed, "command_bytes") == 16u,
         "fixed encoding uses eight bytes for short commands");
}

void test_flat_ports_and_broadcast() {
  // Rows=2, K=3 INT4, lanes=2.  The two W rows are distinct, so every k
  // gathers three bytes.  Port=1/2/3 gives 9/6/3 gather rounds.
  std::vector<unsigned char> commands;
  append_mvm(commands, 0u, 0u, 8u, 128u, 2u, 3u, 2u);
  append_end(commands, 4u);
  const std::string p1 = run(commands, 4u, 2u, 3u, 16u, 0u, 1u);
  expect(json_uint(p1, "operand_read_bytes") == 9u &&
             json_uint(p1, "gather_cycles") == 9u &&
             json_uint(p1, "conflict_cycles") == 6u &&
             json_uint(p1, "core_cycles") == 11u,
         "flat one-port gather count");
  const std::string p2 = run(commands, 4u, 2u, 3u, 16u, 0u, 2u);
  expect(json_uint(p2, "gather_cycles") == 6u &&
             json_uint(p2, "conflict_cycles") == 3u,
         "flat two-port conflicts");
  const std::string p3 = run(commands, 4u, 2u, 3u, 16u, 0u, 3u);
  expect(json_uint(p3, "gather_cycles") == 3u &&
             json_uint(p3, "conflict_cycles") == 0u,
         "flat three-port no conflicts");
  expect(json_uint(p2, "macs") == 6u && json_uint(p2, "mvm_count") == 1u,
         "MAC and MVM counts");

  // A and W intentionally share bytes.  Equal addresses broadcast once,
  // including the repeated low/high nibble byte for K=3.
  commands.clear();
  append_mvm(commands, 0u, 8u, 8u, 128u, 1u, 3u, 2u);
  append_end(commands, 4u);
  const std::string broadcast = run(commands, 4u, 1u, 2u, 16u, 0u, 1u);
  expect(json_uint(broadcast, "operand_read_bytes") == 3u &&
             json_uint(broadcast, "gather_cycles") == 3u &&
             json_uint(broadcast, "conflict_cycles") == 0u,
         "equal byte address broadcast");
}

void test_banks_latency_and_tails() {
  // The A byte and the two W bytes occupy banks {0,0,1}; the banked schedule
  // therefore needs two rounds (the maximum per-bank occupancy), not three
  // flat reads.
  std::vector<unsigned char> commands;
  append_mvm(commands, 0u, 0u, 8u, 128u, 2u, 1u, 1u);
  append_end(commands, 4u);
  const std::string banked = run(commands, 4u, 2u, 2u, 16u, 2u, 5u);
  expect(json_uint(banked, "gather_cycles") == 2u &&
             json_uint(banked, "conflict_cycles") == 1u,
         "bank conflict rounds use per-bank maximum");

  // A tail row group and explicit read latency/depth drain are both counted.
  commands.clear();
  append_mvm(commands, 0u, 0u, 16u, 128u, 3u, 1u, 1u);
  append_end(commands, 4u);
  const std::string tail = run(commands, 4u, 2u, 4u, 32u, 0u, 5u, 2u, 4u);
  expect(json_uint(tail, "macs") == 3u &&
             json_uint(tail, "gather_cycles") == 2u &&
             json_uint(tail, "memory_latency_cycles") == 4u &&
             json_uint(tail, "pipeline_drain_cycles") == 6u &&
             json_uint(tail, "core_cycles") == 12u,
         "tail groups get one latency and drain per group");
}

void test_partial_sum_and_sat16_storage() {
  std::vector<unsigned char> commands;
  // ACCUM|SAT16, ACC32: each two-row group reads 8 bytes and writes 4 bytes;
  // BW=2 gives 4 scratch cycles, with no operand gather in this K=1 case
  // beyond one round.
  append_mvm(commands, 3u, 0u, 16u, 128u, 2u, 1u, 1u);
  append_end(commands, 4u);
  const std::string report = run(commands, 4u, 2u, 2u, 32u, 0u, 5u, 0u,
                                 2u);
  expect(json_uint(report, "scratch_cycles") == 6u &&
             json_uint(report, "core_cycles") == 2u &&
             json_uint(report, "device_cycles") == 10u,
         "partial read and SAT16 output copy phases");
  expect(json_uint(report, "peak_scratch_end") == 136u,
         "ACCUM partial range contributes to peak");
}

void test_invalid_streams_and_recovery() {
  expect_error({}, "empty stream");
  expect_error({15u}, "truncated END");
  expect_error({0u, 0u, 0u, 0u}, "invalid opcode");
  expect_error({15u, 1u, 0u, 0u}, "reserved END field");
  expect_error({15u, 0u, 0u, 0u, 15u, 0u, 0u, 0u},
               "commands after END", 4u);
  expect_error({3u, 0u, 0u, 8u, 128u, 0u, 0u, 0u, 15u, 0u, 0u, 0u},
               "zero stride");

  std::vector<unsigned char> commands;
  append_mvm(commands, 0u, 0u, 8u, 0u, 1u, 1u, 1u);
  append_end(commands, 4u);
  expect_error(commands, "Y aliases A");

  commands.clear();
  append_mvm(commands, 0u, 0u, 255u, 128u, 2u, 1u, 1u);
  append_end(commands, 4u);
  expect_error(commands, "MVM range exceeds scratch", 4u, 1u, 2u, 16u,
               0u, 1u, 0u, 2u);

  commands.clear();
  append_load(commands, 4u, 255u, 2u);
  append_end(commands, 4u);
  expect_error(commands, "LOAD range exceeds scratch");

  // Reserved MVM flags and malformed fixed short-command fields.
  commands = {3u, 0x80u, 0u, 8u, 128u, 0u, 0u, 1u,
              15u, 0u, 0u, 0u};
  expect_error(commands, "reserved MVM flags");
  commands = {1u, 0u, 0u, 0u};
  expect_error(commands, "missing END");

  expect_error({15u, 0u, 0u, 0u}, "invalid lanes", 4u, 3u);
  expect_error({15u, 0u, 0u, 0u}, "invalid banks", 4u, 1u, 2u, 16u,
               3u);
  expect_error({15u, 0u, 0u, 0u}, "invalid flat ports", 4u, 1u, 2u, 16u,
               0u, 0u);
  expect_error({15u, 0u, 0u, 0u}, "invalid latency", 4u, 1u, 2u, 16u,
               0u, 1u, 3u);
  expect_error({15u, 0u, 0u, 0u}, "invalid copy bandwidth", 4u, 1u, 2u,
               16u, 0u, 1u, 0u, 5u);

  const std::string recovered = run({15u, 0u, 0u, 0u});
  expect(json_uint(recovered, "command_count") == 1u &&
             json_uint(recovered, "device_cycles") == 1u,
         "valid call recovers after rejected streams");
}

}  // namespace

int main() {
  test_copy_and_encoding_counts();
  test_flat_ports_and_broadcast();
  test_banks_latency_and_tails();
  test_partial_sum_and_sat16_storage();
  test_invalid_streams_and_recovery();
  std::cout << "Architecture tests passed\n";
  return 0;
}
