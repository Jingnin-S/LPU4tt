#include "tinylpu_plan.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tinylpu_plan::Instruction;
using tinylpu_plan::Opcode;
using tinylpu_plan::Program;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_throw(const std::function<void()>& action, const std::string& message) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("expected rejection: " + message);
}

Instruction instruction(Opcode opcode,
                        std::initializer_list<std::uint32_t> args = {}) {
  Instruction result;
  result.opcode = static_cast<std::uint32_t>(opcode);
  std::size_t index = 0;
  for (std::uint32_t arg : args) {
    if (index >= result.args.size()) {
      throw std::runtime_error("test instruction has too many arguments");
    }
    result.args[index++] = arg;
  }
  return result;
}

Program linear_program() {
  Program program;
  program.header.lanes = 2;
  program.header.depth = 3;
  program.header.a_ports = 1;
  program.header.b_ports = 2;
  program.header.scratchpad_bytes = 64;
  program.header.host_tensor_count = 2;
  program.header.input_elements = 4;
  program.header.output_tensor_id = 1;
  program.header.output_elements = 3;
  program.header.weights_bytes = 6;
  program.instructions = {
      instruction(Opcode::AllocTensor, {1, 3, 16}),
      instruction(Opcode::LoadA, {0, 0, 3, 0}),
      instruction(Opcode::LoadW, {0, 6, 8}),
      instruction(Opcode::Gemv, {0, 8, 16, 3, 3}),
      instruction(Opcode::StoreY, {16, 1, 0, 3}),
      instruction(Opcode::End, {1}),
  };
  program.header.instruction_count =
      static_cast<std::uint32_t>(program.instructions.size());
  // Each GEMV row has ceil(K/2)=2 independent bytes, low nibble first.
  program.weights = {0x21, 0x03, 0x1f, 0x02, 0x07, 0x0e};
  return program;
}

Program reluq_program() {
  Program program;
  program.header.lanes = 4;
  program.header.depth = 2;
  program.header.a_ports = 1;
  program.header.b_ports = 4;
  program.header.scratchpad_bytes = 64;
  program.header.host_tensor_count = 3;
  program.header.input_elements = 3;
  program.header.output_tensor_id = 2;
  program.header.output_elements = 2;
  program.header.weights_bytes = 4;
  program.instructions = {
      instruction(Opcode::AllocTensor, {1, 2, 16}),
      instruction(Opcode::AllocTensor, {2, 2, 4}),
      instruction(Opcode::LoadA, {0, 0, 3, 0}),
      instruction(Opcode::LoadW, {0, 4, 8}),
      instruction(Opcode::Gemv, {0, 8, 16, 2, 3}),
      instruction(Opcode::StoreY, {16, 1, 0, 2}),
      instruction(Opcode::HostReluQ, {1, 2, 2, 3, 2}),
      instruction(Opcode::End, {2}),
  };
  program.header.instruction_count =
      static_cast<std::uint32_t>(program.instructions.size());
  // Row0 [1,1,0] and row1 [7,7,7], with K=3.
  program.weights = {0x11, 0x00, 0x77, 0x07};
  return program;
}

void put_u32_le(std::vector<std::uint8_t>& bytes,
                std::size_t offset,
                std::uint32_t value) {
  expect(offset + 4 <= bytes.size(), "fixture word offset");
  bytes[offset] = static_cast<std::uint8_t>(value & 0xffu);
  bytes[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
  bytes[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
  bytes[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  expect(static_cast<bool>(stream), "open fixture bytes");
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  expect(static_cast<bool>(stream), "write fixture bytes");
}

struct FixturePaths {
  std::filesystem::path program;
  std::filesystem::path weights;
};

FixturePaths write_fixture(const Program& program,
                           const std::filesystem::path& directory,
                           const std::string& stem) {
  FixturePaths paths{directory / (stem + ".tlp"), directory / (stem + ".bin")};
  tinylpu_plan::write_program_files(program, paths.program.string(),
                                    paths.weights.string());
  return paths;
}

void test_linear_and_trace(const std::filesystem::path& directory) {
  const Program program = linear_program();
  const FixturePaths paths = write_fixture(program, directory, "linear");
  const Program loaded =
      tinylpu_plan::load_program(paths.program.string(), paths.weights.string());
  std::vector<tinylpu_plan::TraceEvent> trace;
  const auto first = tinylpu_plan::simulate(
      loaded, {1, -2, 3, 7},
      [&trace](const tinylpu_plan::TraceEvent& event) { trace.push_back(event); });
  expect(first.result == std::vector<int>({6, 3, 1}), "linear result");
  expect(first.core_cycles == 8, "linear gapless cycles");
  expect(first.macs == 9, "linear MAC count");
  expect(first.load_a_bytes == 2 && first.load_w_bytes == 6,
         "linear load byte counts");
  expect(first.store_y_bytes == 6 && first.host_ops == 0,
         "linear store/host counts");
  expect(first.kernel_count == 1 && first.instruction_count == 6,
         "linear instruction/kernel counts");
  expect(first.peak_scratch_end == 22, "linear scratch peak");

  int commands = 0;
  int issues = 0;
  int commits = 0;
  int done_events = 0;
  for (const auto& event : trace) {
    if (event.kind == "command") {
      ++commands;
    }
    if (event.kind == "core") {
      issues += event.issue ? 1 : 0;
      commits += event.commit ? 1 : 0;
    }
    done_events += event.done ? 1 : 0;
  }
  expect(commands == 6, "trace command sequence");
  expect(issues == 9 && commits == 9, "trace issue/commit events");
  expect(done_events >= 1, "trace DONE event");

  const auto changed = tinylpu_plan::simulate(loaded, {2, 1, -1, 7});
  expect(changed.result == std::vector<int>({1, -3, 16}),
         "same program must execute changed input");
  expect(changed.result != first.result, "runtime input must not be precomputed");
}

void test_reluq_and_tail(const std::filesystem::path& directory) {
  const Program program = reluq_program();
  const FixturePaths paths = write_fixture(program, directory, "reluq");
  const Program loaded =
      tinylpu_plan::load_program(paths.program.string(), paths.weights.string());
  const auto result = tinylpu_plan::simulate(loaded, {1, 1, 1});
  expect(result.result == std::vector<int>({2, 7}),
         "HOST_RELU_Q ties-away and clip");
  expect(result.core_cycles == 4, "tail lane core cycles");
  expect(result.macs == 6 && result.host_ops == 2, "reluq counts");
  expect(result.load_a_bytes == 2 && result.load_w_bytes == 4 &&
             result.store_y_bytes == 4,
         "reluq byte counts");
  expect(result.peak_scratch_end == 20, "reluq scratch peak");
  const auto negative = tinylpu_plan::simulate(loaded, {-1, -1, -1});
  expect(negative.result == std::vector<int>({0, 0}),
         "HOST_RELU_Q must apply ReLU before quantization");
}

void test_rejections(const std::filesystem::path& directory) {
  const Program valid = linear_program();
  const FixturePaths paths = write_fixture(valid, directory, "valid");
  const std::vector<std::uint8_t> encoded = tinylpu_plan::encode_program(valid);

  {
    auto bytes = encoded;
    bytes[0] = 'X';
    const auto path = directory / "bad_magic.tlp";
    write_bytes(path, bytes);
    expect_throw([&] { tinylpu_plan::load_program(path.string(), paths.weights.string()); },
                 "bad magic");
  }
  {
    auto bytes = encoded;
    put_u32_le(bytes, 8, 2);
    const auto path = directory / "bad_abi.tlp";
    write_bytes(path, bytes);
    expect_throw([&] { tinylpu_plan::load_program(path.string(), paths.weights.string()); },
                 "bad ABI");
  }
  {
    auto bytes = encoded;
    bytes.pop_back();
    const auto path = directory / "truncated.tlp";
    write_bytes(path, bytes);
    expect_throw([&] { tinylpu_plan::load_program(path.string(), paths.weights.string()); },
                 "truncated program");
  }
  {
    const auto weight_path = directory / "wrong_weights.bin";
    write_bytes(weight_path, {1, 2, 3});
    expect_throw([&] {
      tinylpu_plan::load_program(paths.program.string(), weight_path.string());
    }, "wrong weights length");
  }
  {
    Program unknown = valid;
    unknown.instructions[0].opcode = 99;
    const auto unknown_paths = write_fixture(unknown, directory, "unknown_opcode");
    expect_throw([&] {
      tinylpu_plan::load_program(unknown_paths.program.string(),
                                 unknown_paths.weights.string());
    }, "unknown opcode");
  }
  {
    Program reserved = valid;
    reserved.instructions[0].args[3] = 1;
    const auto reserved_paths = write_fixture(reserved, directory, "reserved_arg");
    expect_throw([&] {
      tinylpu_plan::load_program(reserved_paths.program.string(),
                                 reserved_paths.weights.string());
    }, "reserved argument");
  }
  {
    Program reallocated = valid;
    reallocated.instructions.insert(reallocated.instructions.begin(),
                                    instruction(Opcode::AllocTensor, {0, 4, 4}));
    reallocated.header.instruction_count =
        static_cast<std::uint32_t>(reallocated.instructions.size());
    const auto bad_paths = write_fixture(reallocated, directory, "realloc_input");
    expect_throw([&] {
      tinylpu_plan::load_program(bad_paths.program.string(), bad_paths.weights.string());
    }, "tensor0 reallocation");
  }
  {
    Program uninitialized = valid;
    uninitialized.instructions.erase(uninitialized.instructions.begin() + 1);
    uninitialized.header.instruction_count =
        static_cast<std::uint32_t>(uninitialized.instructions.size());
    const auto bad_paths = write_fixture(uninitialized, directory, "uninitialized");
    expect_throw([&] {
      tinylpu_plan::load_program(bad_paths.program.string(), bad_paths.weights.string());
    }, "uninitialized A");
  }
  {
    Program out_of_bounds = valid;
    out_of_bounds.instructions[3].args[0] = 63;
    const auto bad_paths = write_fixture(out_of_bounds, directory, "oob");
    expect_throw([&] {
      tinylpu_plan::load_program(bad_paths.program.string(), bad_paths.weights.string());
    }, "out-of-bounds GEMV");
  }
  {
    Program no_output = valid;
    no_output.instructions.erase(no_output.instructions.begin() + 4);
    no_output.header.instruction_count =
        static_cast<std::uint32_t>(no_output.instructions.size());
    const auto bad_paths = write_fixture(no_output, directory, "no_output");
    expect_throw([&] {
      tinylpu_plan::load_program(bad_paths.program.string(), bad_paths.weights.string());
    }, "invalid output buffer");
  }
  const Program loaded =
      tinylpu_plan::load_program(paths.program.string(), paths.weights.string());
  expect_throw([&] { tinylpu_plan::simulate(loaded, {1, 2, 3}); },
               "short input");
  expect_throw([&] { tinylpu_plan::simulate(loaded, {1, 2, 3, 8}); },
               "out-of-range input");
  expect_throw([&] { tinylpu_plan::parse_input_values("1,,2"); },
               "empty input element");
  expect_throw([&] { tinylpu_plan::parse_input_values("1,x"); },
               "non-integer input element");
}

}  // namespace

int main() {
  const auto unique = std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("tinylpu_plan_test_" + std::to_string(unique));
  try {
    std::filesystem::create_directories(directory);
    test_linear_and_trace(directory);
    test_reluq_and_tail(directory);
    test_rejections(directory);
    std::filesystem::remove_all(directory);
    std::cout << "plan tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::cerr << "plan test failure: " << error.what() << '\n';
    return 1;
  }
}
