#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tinylpu_plan {

constexpr std::uint32_t kAbiVersion = 1;
constexpr std::size_t kMagicBytes = 8;
constexpr std::size_t kHeaderBytes = 56;
constexpr std::size_t kInstructionBytes = 32;
constexpr std::array<std::uint8_t, kMagicBytes> kMagic = {
    {'T', 'L', 'P', 'U', 'v', '0', '0', '1'}};

enum class Opcode : std::uint32_t {
  AllocTensor = 1,
  LoadA = 2,
  LoadW = 3,
  Gemv = 4,
  StoreY = 5,
  HostReluQ = 6,
  End = 7,
};

// The field order is the little-endian <8s12I ABI described in the compiler
// change plan. Do not serialize this C++ struct directly: the loader/writer
// use explicit little-endian words so host alignment cannot change the ABI.
struct Header {
  std::uint32_t abi = kAbiVersion;
  std::uint32_t lanes = 1;
  std::uint32_t depth = 2;
  std::uint32_t a_ports = 1;
  std::uint32_t b_ports = 1;
  std::uint32_t scratchpad_bytes = 0;
  std::uint32_t host_tensor_count = 0;
  std::uint32_t input_elements = 0;
  std::uint32_t output_tensor_id = 0;
  std::uint32_t output_elements = 0;
  std::uint32_t instruction_count = 0;
  std::uint32_t weights_bytes = 0;
};

struct Instruction {
  std::uint32_t opcode = 0;
  std::array<std::uint32_t, 7> args{};
};

struct Program {
  Header header;
  std::vector<Instruction> instructions;
  std::vector<std::uint8_t> weights;
};

// This helper is intentionally exposed for C++ tests and small local fixture
// generators. It writes the exact binary ABI and does not silently repair an
// intentionally malformed fixture.
std::vector<std::uint8_t> encode_program(const Program& program);
void write_program_files(const Program& program,
                         const std::string& program_path,
                         const std::string& weights_path);

// load_program performs strict magic/ABI/length/weights checks and validates
// the static instruction/dataflow contract before returning.
Program load_program(const std::string& program_path,
                     const std::string& weights_path);
void validate_program(const Program& program);

struct TraceEvent {
  std::uint64_t sequence = 0;
  std::string kind;  // command, core, or done
  std::uint32_t instruction = 0;
  std::uint32_t opcode = 0;
  int relative = 0;
  int lane = -1;
  int issue_row = -1;
  int issue_index = -1;
  int commit_row = -1;
  int commit_index = -1;
  bool issue = false;
  bool commit = false;
  bool done = false;
  int issue_product = 0;
  int commit_product = 0;
  int acc = 0;
};

using TraceSink = std::function<void(const TraceEvent&)>;

struct SimulationReport {
  std::uint32_t abi = kAbiVersion;
  Header header;
  std::vector<int> input;
  std::vector<int> result;
  std::uint64_t core_cycles = 0;
  std::uint64_t macs = 0;
  std::uint64_t load_a_bytes = 0;
  std::uint64_t load_w_bytes = 0;
  std::uint64_t store_y_bytes = 0;
  std::uint64_t host_ops = 0;
  std::uint32_t instruction_count = 0;
  std::uint32_t kernel_count = 0;
  std::uint32_t peak_scratch_end = 0;
};

std::vector<int> parse_input_values(const std::string& text);
SimulationReport simulate(const Program& program,
                          const std::vector<int>& input,
                          TraceSink trace_sink = {});
std::string report_json(const SimulationReport& report);
std::string opcode_name(std::uint32_t opcode);

struct CliParseResult {
  bool ok = false;
  bool help = false;
  std::string program_path;
  std::string weights_path;
  std::string input_text;
  std::string trace_path;
  std::string error;
};

CliParseResult parse_cli(int argc, char** argv);
std::string usage();

}  // namespace tinylpu_plan
