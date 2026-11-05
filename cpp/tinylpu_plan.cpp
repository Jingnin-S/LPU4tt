#include "tinylpu_plan.hpp"

#include "tinylpu_dse.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace tinylpu_plan {
namespace {

constexpr std::uint32_t kMaxTensorElements = 1u << 24;
constexpr std::uint32_t kMaxProgramBytes =
    static_cast<std::uint32_t>(kHeaderBytes + 100000u * kInstructionBytes);
constexpr std::uint32_t kMaxWeightsBytes = 64u * 1024u * 1024u;

enum class ByteRole : std::uint8_t { Empty, A, W, Y };

struct TensorState {
  bool allocated = false;
  std::uint32_t count = 0;
  std::uint32_t bits = 0;
  std::vector<std::int32_t> values;
  std::vector<std::uint8_t> valid;
};

struct ExecutionState {
  std::vector<TensorState> tensors;
  std::vector<std::uint8_t> scratch;
  std::vector<std::uint8_t> initialized;
  std::vector<ByteRole> roles;
  bool ended = false;

  explicit ExecutionState(const Header& header)
      : tensors(header.host_tensor_count),
        scratch(header.scratchpad_bytes, 0),
        initialized(header.scratchpad_bytes, 0),
        roles(header.scratchpad_bytes, ByteRole::Empty) {
    // Tensor 0 is the pre-valid runtime input and therefore is not allocated
    // by an instruction. ALLOC_TENSOR 0 is a deliberate reallocation error.
    TensorState& input = tensors[0];
    input.allocated = true;
    input.count = header.input_elements;
    input.bits = 4;
    input.values.assign(header.input_elements, 0);
    input.valid.assign(header.input_elements, 1);
  }
};

struct Counters {
  std::uint64_t core_cycles = 0;
  std::uint64_t macs = 0;
  std::uint64_t load_a_bytes = 0;
  std::uint64_t load_w_bytes = 0;
  std::uint64_t store_y_bytes = 0;
  std::uint64_t host_ops = 0;
  std::uint32_t kernel_count = 0;
  std::uint32_t peak_scratch_end = 0;
};

std::string describe_instruction(std::size_t index,
                                 std::uint32_t opcode,
                                 const std::string& message) {
  std::ostringstream out;
  out << "instruction " << index << " (" << opcode_name(opcode) << "): "
      << message;
  return out.str();
}

std::vector<std::uint8_t> read_file(const std::string& path,
                                    std::uint64_t max_bytes,
                                    const std::string& what) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + what + " '" + path + "'");
  }
  stream.seekg(0, std::ios::end);
  const std::streamoff end = stream.tellg();
  if (end < 0) {
    throw std::runtime_error("cannot determine size of " + what);
  }
  const std::uint64_t size = static_cast<std::uint64_t>(end);
  if (size > max_bytes) {
    throw std::runtime_error(what + " exceeds safety limit");
  }
  stream.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  if (!data.empty()) {
    stream.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    if (stream.gcount() != static_cast<std::streamsize>(data.size())) {
      throw std::runtime_error("truncated " + what);
    }
  }
  return data;
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& data,
                const std::string& what) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot open " + what + " '" + path + "'");
  }
  if (!data.empty()) {
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
  }
  if (!stream) {
    throw std::runtime_error("cannot write " + what + " '" + path + "'");
  }
}

std::uint32_t read_u32_le(const std::vector<std::uint8_t>& data,
                          std::size_t offset) {
  if (offset > data.size() || data.size() - offset < 4) {
    throw std::runtime_error("truncated little-endian word");
  }
  return static_cast<std::uint32_t>(data[offset]) |
         (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

void append_u32_le(std::vector<std::uint8_t>& data, std::uint32_t value) {
  data.push_back(static_cast<std::uint8_t>(value & 0xffu));
  data.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
  data.push_back(static_cast<std::uint8_t>((value >> 16) & 0xffu));
  data.push_back(static_cast<std::uint8_t>((value >> 24) & 0xffu));
}

std::array<std::uint32_t, 12> header_words(const Header& header) {
  return {header.abi,
          header.lanes,
          header.depth,
          header.a_ports,
          header.b_ports,
          header.scratchpad_bytes,
          header.host_tensor_count,
          header.input_elements,
          header.output_tensor_id,
          header.output_elements,
          header.instruction_count,
          header.weights_bytes};
}

Header decode_header(const std::vector<std::uint8_t>& data) {
  if (data.size() < kHeaderBytes) {
    throw std::runtime_error("program is shorter than the 56-byte header");
  }
  for (std::size_t i = 0; i < kMagicBytes; ++i) {
    if (data[i] != kMagic[i]) {
      throw std::runtime_error("program magic is not TLPUv001");
    }
  }
  Header header;
  header.abi = read_u32_le(data, 8);
  header.lanes = read_u32_le(data, 12);
  header.depth = read_u32_le(data, 16);
  header.a_ports = read_u32_le(data, 20);
  header.b_ports = read_u32_le(data, 24);
  header.scratchpad_bytes = read_u32_le(data, 28);
  header.host_tensor_count = read_u32_le(data, 32);
  header.input_elements = read_u32_le(data, 36);
  header.output_tensor_id = read_u32_le(data, 40);
  header.output_elements = read_u32_le(data, 44);
  header.instruction_count = read_u32_le(data, 48);
  header.weights_bytes = read_u32_le(data, 52);
  return header;
}

void validate_header(const Header& header) {
  if (header.abi != kAbiVersion) {
    throw std::invalid_argument("unsupported plan ABI version");
  }
  if (header.lanes != 1 && header.lanes != 2 && header.lanes != 4) {
    throw std::invalid_argument("lanes must be one of 1,2,4");
  }
  if (header.depth < 2 || header.depth > 4) {
    throw std::invalid_argument("depth must be in [2,4]");
  }
  if (header.a_ports < 1 || header.b_ports < 1) {
    throw std::invalid_argument("A/B read ports must be positive");
  }
  if (header.scratchpad_bytes == 0 || header.scratchpad_bytes > 65536) {
    throw std::invalid_argument("scratchpad_bytes must be in [1,65536]");
  }
  if (header.host_tensor_count == 0 || header.host_tensor_count > 64) {
    throw std::invalid_argument("host_tensor_count must be in [1,64]");
  }
  if (header.input_elements == 0 || header.output_elements == 0 ||
      header.input_elements > kMaxTensorElements ||
      header.output_elements > kMaxTensorElements) {
    throw std::invalid_argument(
        "input/output element counts must be in [1,2^24]");
  }
  if (header.output_tensor_id >= header.host_tensor_count) {
    throw std::invalid_argument("output_tensor_id is outside host tensor table");
  }
  if (header.instruction_count == 0 || header.instruction_count > 100000) {
    throw std::invalid_argument("instruction_count must be in [1,100000]");
  }
  if (header.weights_bytes > kMaxWeightsBytes) {
    throw std::invalid_argument("weights_bytes exceeds safety limit");
  }
}

std::size_t checked_end(std::uint32_t address,
                        std::uint64_t bytes,
                        std::uint32_t capacity,
                        const std::string& what) {
  const std::uint64_t end = static_cast<std::uint64_t>(address) + bytes;
  if (end > capacity) {
    throw std::invalid_argument(what + " exceeds scratchpad");
  }
  return static_cast<std::size_t>(end);
}

bool ranges_overlap(std::uint32_t a,
                    std::uint64_t a_bytes,
                    std::uint32_t b,
                    std::uint64_t b_bytes) {
  const std::uint64_t a_end = static_cast<std::uint64_t>(a) + a_bytes;
  const std::uint64_t b_end = static_cast<std::uint64_t>(b) + b_bytes;
  return static_cast<std::uint64_t>(a) < b_end &&
         static_cast<std::uint64_t>(b) < a_end;
}

void update_peak(Counters* counters, std::uint32_t end) {
  if (counters != nullptr) {
    counters->peak_scratch_end = std::max(counters->peak_scratch_end, end);
  }
}

void mark_role(ExecutionState& state,
               std::uint32_t address,
               std::uint64_t bytes,
               ByteRole role,
               Counters* counters,
               const std::string& what) {
  const std::size_t end = checked_end(address, bytes,
                                      static_cast<std::uint32_t>(state.scratch.size()),
                                      what);
  const std::size_t begin = address;
  for (std::size_t i = begin; i < end; ++i) {
    // LOAD_A/LOAD_W/GEMV are explicit sequential writers. A later writer may
    // retag storage previously used by another role; any stale read is still
    // rejected by require_role before it can execute.
    state.roles[i] = role;
  }
  update_peak(counters, static_cast<std::uint32_t>(end));
}

void require_role(const ExecutionState& state,
                  std::uint32_t address,
                  std::uint64_t bytes,
                  ByteRole role,
                  const std::string& what) {
  const std::size_t end = checked_end(address, bytes,
                                      static_cast<std::uint32_t>(state.scratch.size()),
                                      what);
  for (std::size_t i = address; i < end; ++i) {
    if (state.roles[i] != role || state.initialized[i] == 0) {
      throw std::invalid_argument(what + " reads uninitialized or wrong-role scratch data");
    }
  }
}

TensorState& tensor(ExecutionState& state,
                    std::uint32_t id,
                    std::size_t instruction,
                    std::uint32_t opcode) {
  if (id >= state.tensors.size()) {
    throw std::invalid_argument(describe_instruction(
        instruction, opcode, "tensor id is outside host tensor table"));
  }
  TensorState& result = state.tensors[id];
  if (!result.allocated) {
    throw std::invalid_argument(
        describe_instruction(instruction, opcode, "tensor has not been allocated"));
  }
  return result;
}

void require_bits(const TensorState& value,
                  std::uint32_t bits,
                  std::size_t instruction,
                  std::uint32_t opcode,
                  const std::string& role) {
  if (value.bits != bits) {
    std::ostringstream out;
    out << role << " tensor must be INT" << bits;
    throw std::invalid_argument(describe_instruction(instruction, opcode, out.str()));
  }
}

void require_valid_range(const TensorState& value,
                         std::uint32_t offset,
                         std::uint32_t count,
                         std::size_t instruction,
                         std::uint32_t opcode,
                         const std::string& role) {
  if (count == 0 || static_cast<std::uint64_t>(offset) + count > value.count) {
    throw std::invalid_argument(describe_instruction(
        instruction, opcode, role + " element range is outside tensor"));
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    if (value.valid[offset + i] == 0) {
      throw std::invalid_argument(describe_instruction(
          instruction, opcode, role + " reads an invalid host tensor element"));
    }
  }
}

int decode_int4(std::uint8_t value) {
  const int nibble = value & 0x0f;
  return nibble >= 8 ? nibble - 16 : nibble;
}

std::uint8_t encode_int4(int value) {
  if (value < -8 || value > 7) {
    throw std::invalid_argument("value outside signed INT4 [-8,7]");
  }
  return static_cast<std::uint8_t>(value) & 0x0f;
}

void require_reserved_zero(const Instruction& instruction,
                           std::initializer_list<std::size_t> used,
                           std::size_t index) {
  std::array<bool, 7> is_used{};
  for (std::size_t position : used) {
    if (position >= is_used.size()) {
      throw std::logic_error("internal instruction schema error");
    }
    is_used[position] = true;
  }
  for (std::size_t i = 0; i < instruction.args.size(); ++i) {
    if (!is_used[i] && instruction.args[i] != 0) {
      throw std::invalid_argument(describe_instruction(
          index, instruction.opcode, "reserved argument must be zero"));
    }
  }
}

void allocate_tensor_impl(ExecutionState& state,
                          const Header& header,
                          const Instruction& instruction,
                          std::size_t index,
                          const std::vector<int>* input) {
  require_reserved_zero(instruction, {0, 1, 2}, index);
  const std::uint32_t id = instruction.args[0];
  const std::uint32_t count = instruction.args[1];
  const std::uint32_t bits = instruction.args[2];
  if (id >= state.tensors.size()) {
    throw std::invalid_argument(describe_instruction(
        index, instruction.opcode, "tensor id is outside host tensor table"));
  }
  if (count == 0 || count > kMaxTensorElements) {
    throw std::invalid_argument(describe_instruction(
        index, instruction.opcode, "tensor count is outside supported range"));
  }
  if (bits != 4 && bits != 16) {
    throw std::invalid_argument(describe_instruction(
        index, instruction.opcode, "tensor bits must be 4 or 16"));
  }
  TensorState& result = state.tensors[id];
  if (result.allocated) {
    throw std::invalid_argument(
        describe_instruction(index, instruction.opcode, "tensor is allocated twice"));
  }
  result.allocated = true;
  result.count = count;
  result.bits = bits;
  result.values.assign(count, 0);
  result.valid.assign(count, 0);
  (void)header;
  (void)input;
}

void execute_instruction(const Program& program,
                         ExecutionState& state,
                         std::size_t index,
                         bool runtime,
                         const std::vector<int>* input,
                         Counters* counters,
                         TraceSink trace_sink,
                         std::uint64_t* trace_sequence) {
  if (index >= program.instructions.size()) {
    throw std::logic_error("instruction index outside program");
  }
  const Instruction& instruction = program.instructions[index];
  const Header& header = program.header;
  switch (static_cast<Opcode>(instruction.opcode)) {
    case Opcode::AllocTensor:
      allocate_tensor_impl(state, header, instruction, index, input);
      return;

    case Opcode::LoadA: {
      require_reserved_zero(instruction, {0, 1, 2, 3}, index);
      const std::uint32_t id = instruction.args[0];
      const std::uint32_t offset = instruction.args[1];
      const std::uint32_t count = instruction.args[2];
      const std::uint32_t address = instruction.args[3];
      const TensorState& source = tensor(state, id, index, instruction.opcode);
      require_bits(source, 4, index, instruction.opcode, "LOAD_A source");
      require_valid_range(source, offset, count, index, instruction.opcode,
                          "LOAD_A source");
      const std::uint64_t bytes = (static_cast<std::uint64_t>(count) + 1) / 2;
      mark_role(state, address, bytes, ByteRole::A, counters, "LOAD_A");
      for (std::uint32_t byte = 0; byte < bytes; ++byte) {
        state.scratch[address + byte] = 0;
        state.initialized[address + byte] = 1;
      }
      for (std::uint32_t element = 0; element < count; ++element) {
        const std::uint8_t nibble = encode_int4(source.values[offset + element]);
        const std::size_t destination = address + element / 2;
        if ((element & 1u) == 0) {
          state.scratch[destination] = nibble;
        } else {
          state.scratch[destination] = static_cast<std::uint8_t>(
              state.scratch[destination] | static_cast<std::uint8_t>(nibble << 4));
        }
      }
      if (runtime && counters != nullptr) {
        counters->load_a_bytes += bytes;
      }
      return;
    }

    case Opcode::LoadW: {
      require_reserved_zero(instruction, {0, 1, 2}, index);
      const std::uint32_t blob_offset = instruction.args[0];
      const std::uint32_t byte_count = instruction.args[1];
      const std::uint32_t address = instruction.args[2];
      if (byte_count == 0 ||
          static_cast<std::uint64_t>(blob_offset) + byte_count >
              program.weights.size()) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "weight blob range is outside weights.bin"));
      }
      mark_role(state, address, byte_count, ByteRole::W, counters, "LOAD_W");
      for (std::uint32_t byte = 0; byte < byte_count; ++byte) {
        state.scratch[address + byte] = program.weights[blob_offset + byte];
        state.initialized[address + byte] = 1;
      }
      if (runtime && counters != nullptr) {
        counters->load_w_bytes += byte_count;
      }
      return;
    }

    case Opcode::Gemv: {
      require_reserved_zero(instruction, {0, 1, 2, 3, 4}, index);
      const std::uint32_t a_address = instruction.args[0];
      const std::uint32_t w_address = instruction.args[1];
      const std::uint32_t y_address = instruction.args[2];
      const std::uint32_t rows = instruction.args[3];
      const std::uint32_t k = instruction.args[4];
      if (rows < 1 || rows > 16 || k < 1 || k > 64) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "GEMV rows/k outside DSE range"));
      }
      if (header.a_ports < 1 ||
          header.b_ports < std::min(header.lanes, rows)) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "GEMV read ports do not cover lane group"));
      }
      const std::uint64_t a_bytes = (static_cast<std::uint64_t>(k) + 1) / 2;
      const std::uint64_t row_w_bytes = a_bytes;
      const std::uint64_t w_bytes = row_w_bytes * rows;
      const std::uint64_t y_bytes = static_cast<std::uint64_t>(rows) * 2;
      checked_end(a_address, a_bytes, header.scratchpad_bytes, "GEMV A");
      checked_end(w_address, w_bytes, header.scratchpad_bytes, "GEMV W");
      checked_end(y_address, y_bytes, header.scratchpad_bytes, "GEMV Y");
      if ((y_address & 1u) != 0) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "GEMV Y address must be 2-byte aligned"));
      }
      if (ranges_overlap(a_address, a_bytes, w_address, w_bytes) ||
          ranges_overlap(a_address, a_bytes, y_address, y_bytes) ||
          ranges_overlap(w_address, w_bytes, y_address, y_bytes)) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "GEMV A/W/Y ranges overlap"));
      }
      require_role(state, a_address, a_bytes, ByteRole::A, "GEMV A");
      require_role(state, w_address, w_bytes, ByteRole::W, "GEMV W");
      mark_role(state, y_address, y_bytes, ByteRole::Y, counters, "GEMV Y");
      for (std::uint64_t byte = 0; byte < y_bytes; ++byte) {
        state.scratch[y_address + byte] = 0;
        state.initialized[y_address + byte] = 1;
      }

      if (runtime) {
        std::vector<std::int8_t> a(static_cast<std::size_t>(k));
        std::vector<std::int8_t> weights(static_cast<std::size_t>(rows * k));
        for (std::uint32_t i = 0; i < k; ++i) {
          const std::uint8_t packed = state.scratch[a_address + i / 2];
          a[i] = static_cast<std::int8_t>(decode_int4(
              (i & 1u) == 0 ? packed : static_cast<std::uint8_t>(packed >> 4)));
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
          for (std::uint32_t i = 0; i < k; ++i) {
            const std::uint8_t packed =
                state.scratch[w_address + row * row_w_bytes + i / 2];
            weights[row * k + i] = static_cast<std::int8_t>(decode_int4(
                (i & 1u) == 0 ? packed : static_cast<std::uint8_t>(packed >> 4)));
          }
        }
        tinylpu_dse::Config config;
        config.rows = static_cast<int>(rows);
        config.k = static_cast<int>(k);
        config.lanes = static_cast<int>(header.lanes);
        config.depth = static_cast<int>(header.depth);
        config.row_drain = false;
        tinylpu_dse::DseCore core(config);
        core.set_operands(a, weights);
        core.start();
        while (core.running()) {
          const tinylpu_dse::Tick tick = core.tick();
          if (counters != nullptr) {
            ++counters->core_cycles;
          }
          for (const tinylpu_dse::LaneEvent& source_event : tick.events) {
            if (trace_sink) {
              TraceEvent event;
              event.kind = "core";
              event.instruction = static_cast<std::uint32_t>(index);
              event.opcode = instruction.opcode;
              event.relative = source_event.relative;
              event.lane = source_event.lane;
              event.issue_row = source_event.issue_row;
              event.issue_index = source_event.issue_index;
              event.commit_row = source_event.commit_row;
              event.commit_index = source_event.commit_index;
              event.issue = source_event.issue;
              event.commit = source_event.commit;
              event.done = source_event.done;
              event.issue_product = source_event.issue_product;
              event.commit_product = source_event.commit_product;
              event.acc = source_event.acc;
              event.sequence = trace_sequence == nullptr ? 0 : ++(*trace_sequence);
              trace_sink(event);
            }
          }
          if (tick.done && trace_sink) {
            TraceEvent event;
            event.kind = "done";
            event.instruction = static_cast<std::uint32_t>(index);
            event.opcode = instruction.opcode;
            event.relative = tick.relative;
            event.done = true;
            event.sequence = trace_sequence == nullptr ? 0 : ++(*trace_sequence);
            trace_sink(event);
          }
        }
        const std::vector<std::int16_t>& result = core.results();
        for (std::uint32_t row = 0; row < rows; ++row) {
          const std::uint16_t bits = static_cast<std::uint16_t>(
              result[static_cast<std::size_t>(row)]);
          state.scratch[y_address + row * 2] =
              static_cast<std::uint8_t>(bits & 0xffu);
          state.scratch[y_address + row * 2 + 1] =
              static_cast<std::uint8_t>((bits >> 8) & 0xffu);
        }
        if (counters != nullptr) {
          counters->macs += static_cast<std::uint64_t>(rows) * k;
          ++counters->kernel_count;
        }
      }
      return;
    }

    case Opcode::StoreY: {
      require_reserved_zero(instruction, {0, 1, 2, 3}, index);
      const std::uint32_t y_address = instruction.args[0];
      const std::uint32_t id = instruction.args[1];
      const std::uint32_t offset = instruction.args[2];
      const std::uint32_t count = instruction.args[3];
      TensorState& destination = tensor(state, id, index, instruction.opcode);
      require_bits(destination, 16, index, instruction.opcode, "STORE_Y destination");
      if (count == 0 || static_cast<std::uint64_t>(offset) + count > destination.count) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "STORE_Y destination range is outside tensor"));
      }
      const std::uint64_t bytes = static_cast<std::uint64_t>(count) * 2;
      require_role(state, y_address, bytes, ByteRole::Y, "STORE_Y source");
      for (std::uint32_t element = 0; element < count; ++element) {
        const std::uint32_t address = y_address + element * 2;
        const std::uint16_t bits = static_cast<std::uint16_t>(state.scratch[address]) |
                                   (static_cast<std::uint16_t>(state.scratch[address + 1]) << 8);
        destination.values[offset + element] =
            static_cast<std::int16_t>(bits);
        destination.valid[offset + element] = 1;
      }
      if (runtime && counters != nullptr) {
        counters->store_y_bytes += bytes;
      }
      return;
    }

    case Opcode::HostReluQ: {
      require_reserved_zero(instruction, {0, 1, 2, 3, 4}, index);
      const std::uint32_t src_id = instruction.args[0];
      const std::uint32_t dst_id = instruction.args[1];
      const std::uint32_t count = instruction.args[2];
      const std::uint32_t gain_num = instruction.args[3];
      const std::uint32_t gain_shift = instruction.args[4];
      if (gain_num != 3 || gain_shift != 2) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "HOST_RELU_Q requires gain args 3,2 for 3/4"));
      }
      const TensorState& source = tensor(state, src_id, index, instruction.opcode);
      TensorState& destination = tensor(state, dst_id, index, instruction.opcode);
      require_bits(source, 16, index, instruction.opcode, "HOST_RELU_Q source");
      require_bits(destination, 4, index, instruction.opcode, "HOST_RELU_Q destination");
      require_valid_range(source, 0, count, index, instruction.opcode,
                          "HOST_RELU_Q source");
      if (count == 0 || count > destination.count) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "HOST_RELU_Q count exceeds destination"));
      }
      for (std::uint32_t element = 0; element < count; ++element) {
        const int nonnegative = std::max(0, static_cast<int>(source.values[element]));
        const int denominator = 1 << gain_shift;
        const int rounded = (nonnegative * static_cast<int>(gain_num) +
                             denominator / 2) /
                            denominator;
        destination.values[element] = std::min(7, std::max(0, rounded));
        destination.valid[element] = 1;
      }
      if (runtime && counters != nullptr) {
        counters->host_ops += count;
      }
      return;
    }

    case Opcode::End: {
      require_reserved_zero(instruction, {0}, index);
      if (index + 1 != program.instructions.size()) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "END must be the final instruction"));
      }
      const std::uint32_t id = instruction.args[0];
      if (id != header.output_tensor_id) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "END tensor does not match header output_tensor_id"));
      }
      const TensorState& output = tensor(state, id, index, instruction.opcode);
      if (output.count != header.output_elements) {
        throw std::invalid_argument(describe_instruction(
            index, instruction.opcode, "END tensor count does not match output_elements"));
      }
      for (std::uint8_t valid : output.valid) {
        if (valid == 0) {
          throw std::invalid_argument(describe_instruction(
              index, instruction.opcode, "END reads an invalid output element"));
        }
      }
      state.ended = true;
      return;
    }

    default:
      throw std::invalid_argument(describe_instruction(
          index, instruction.opcode, "unknown opcode"));
  }
}

void emit_command(const Instruction& instruction,
                  std::size_t index,
                  TraceSink sink,
                  std::uint64_t* sequence) {
  if (!sink) {
    return;
  }
  TraceEvent event;
  event.kind = "command";
  event.instruction = static_cast<std::uint32_t>(index);
  event.opcode = instruction.opcode;
  event.sequence = sequence == nullptr ? 0 : ++(*sequence);
  sink(event);
}

}  // namespace

std::string opcode_name(std::uint32_t opcode) {
  switch (static_cast<Opcode>(opcode)) {
    case Opcode::AllocTensor:
      return "ALLOC_TENSOR";
    case Opcode::LoadA:
      return "LOAD_A";
    case Opcode::LoadW:
      return "LOAD_W";
    case Opcode::Gemv:
      return "GEMV";
    case Opcode::StoreY:
      return "STORE_Y";
    case Opcode::HostReluQ:
      return "HOST_RELU_Q";
    case Opcode::End:
      return "END";
    default:
      return "UNKNOWN";
  }
}

std::vector<std::uint8_t> encode_program(const Program& program) {
  if (program.instructions.size() != program.header.instruction_count) {
    throw std::invalid_argument("instruction vector size does not match header");
  }
  std::vector<std::uint8_t> data;
  data.reserve(kHeaderBytes + program.instructions.size() * kInstructionBytes);
  data.insert(data.end(), kMagic.begin(), kMagic.end());
  for (std::uint32_t word : header_words(program.header)) {
    append_u32_le(data, word);
  }
  for (const Instruction& instruction : program.instructions) {
    append_u32_le(data, instruction.opcode);
    for (std::uint32_t arg : instruction.args) {
      append_u32_le(data, arg);
    }
  }
  return data;
}

void write_program_files(const Program& program,
                         const std::string& program_path,
                         const std::string& weights_path) {
  write_file(program_path, encode_program(program), "program");
  write_file(weights_path, program.weights, "weights");
}

Program load_program(const std::string& program_path,
                     const std::string& weights_path) {
  const std::vector<std::uint8_t> data =
      read_file(program_path, kMaxProgramBytes, "program");
  Program program;
  program.header = decode_header(data);
  validate_header(program.header);
  const std::uint64_t expected_size =
      static_cast<std::uint64_t>(kHeaderBytes) +
      static_cast<std::uint64_t>(program.header.instruction_count) *
          kInstructionBytes;
  if (expected_size != data.size()) {
    throw std::runtime_error("program length does not match instruction_count");
  }
  program.instructions.resize(program.header.instruction_count);
  std::size_t offset = kHeaderBytes;
  for (Instruction& instruction : program.instructions) {
    instruction.opcode = read_u32_le(data, offset);
    offset += 4;
    for (std::uint32_t& arg : instruction.args) {
      arg = read_u32_le(data, offset);
      offset += 4;
    }
  }
  program.weights = read_file(weights_path, kMaxWeightsBytes, "weights.bin");
  if (program.weights.size() != program.header.weights_bytes) {
    throw std::runtime_error("weights.bin length does not match header weights_bytes");
  }
  validate_program(program);
  return program;
}

void validate_program(const Program& program) {
  validate_header(program.header);
  if (program.instructions.size() != program.header.instruction_count) {
    throw std::invalid_argument("instruction vector size does not match header");
  }
  if (program.weights.size() != program.header.weights_bytes) {
    throw std::invalid_argument("weights length does not match header");
  }
  if (program.instructions.empty() ||
      program.instructions.back().opcode != static_cast<std::uint32_t>(Opcode::End)) {
    throw std::invalid_argument("program must end with END");
  }
  ExecutionState state(program.header);
  for (std::size_t index = 0; index < program.instructions.size(); ++index) {
    execute_instruction(program, state, index, false, nullptr, nullptr, {}, nullptr);
  }
  if (!state.ended) {
    throw std::invalid_argument("program has no valid END");
  }
}

std::vector<int> parse_input_values(const std::string& text) {
  std::vector<int> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string::npos ? text.size() : comma;
    std::size_t left = begin;
    std::size_t right = end;
    while (left < right && std::isspace(static_cast<unsigned char>(text[left]))) {
      ++left;
    }
    while (right > left && std::isspace(static_cast<unsigned char>(text[right - 1]))) {
      --right;
    }
    if (left == right) {
      throw std::invalid_argument("input contains an empty element");
    }
    std::string token = text.substr(left, right - left);
    std::size_t used = 0;
    int value = 0;
    try {
      value = std::stoi(token, &used, 10);
    } catch (const std::exception&) {
      throw std::invalid_argument("input contains a non-integer element");
    }
    if (used != token.size()) {
      throw std::invalid_argument("input contains a non-integer element");
    }
    values.push_back(value);
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (values.empty()) {
    throw std::invalid_argument("input must contain at least one integer");
  }
  return values;
}

SimulationReport simulate(const Program& program,
                          const std::vector<int>& input,
                          TraceSink trace_sink) {
  validate_program(program);
  if (input.size() != program.header.input_elements) {
    throw std::invalid_argument("input length does not match header input_elements");
  }
  for (int value : input) {
    if (value < -8 || value > 7) {
      throw std::invalid_argument("input value outside signed INT4 [-8,7]");
    }
  }

  ExecutionState state(program.header);
  for (std::size_t i = 0; i < input.size(); ++i) {
    state.tensors[0].values[i] = input[i];
  }
  Counters counters;
  std::uint64_t trace_sequence = 0;
  for (std::size_t index = 0; index < program.instructions.size(); ++index) {
    emit_command(program.instructions[index], index, trace_sink, &trace_sequence);
    execute_instruction(program, state, index, true, &input, &counters,
                        trace_sink, &trace_sequence);
  }
  if (!state.ended) {
    throw std::runtime_error("simulation ended without END");
  }
  const TensorState& output = state.tensors[program.header.output_tensor_id];
  SimulationReport report;
  report.abi = program.header.abi;
  report.header = program.header;
  report.input = input;
  report.result.reserve(output.values.size());
  for (std::int32_t value : output.values) {
    report.result.push_back(static_cast<int>(value));
  }
  report.core_cycles = counters.core_cycles;
  report.macs = counters.macs;
  report.load_a_bytes = counters.load_a_bytes;
  report.load_w_bytes = counters.load_w_bytes;
  report.store_y_bytes = counters.store_y_bytes;
  report.host_ops = counters.host_ops;
  report.instruction_count = program.header.instruction_count;
  report.kernel_count = counters.kernel_count;
  report.peak_scratch_end = counters.peak_scratch_end;
  return report;
}

namespace {

void append_int_array(std::ostringstream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
}

}  // namespace

std::string report_json(const SimulationReport& report) {
  std::ostringstream out;
  out << "{\"abi\":" << report.abi << ",\"target\":{";
  out << "\"lanes\":" << report.header.lanes
      << ",\"depth\":" << report.header.depth
      << ",\"a_ports\":" << report.header.a_ports
      << ",\"b_ports\":" << report.header.b_ports
      << ",\"scratchpad_bytes\":" << report.header.scratchpad_bytes << '}';
  out << ",\"input\":";
  append_int_array(out, report.input);
  out << ",\"result\":";
  append_int_array(out, report.result);
  out << ",\"core_cycles\":" << report.core_cycles
      << ",\"macs\":" << report.macs
      << ",\"load_a_bytes\":" << report.load_a_bytes
      << ",\"load_w_bytes\":" << report.load_w_bytes
      << ",\"store_y_bytes\":" << report.store_y_bytes
      << ",\"host_ops\":" << report.host_ops
      << ",\"instruction_count\":" << report.instruction_count
      << ",\"kernel_count\":" << report.kernel_count
      << ",\"peak_scratch_end\":" << report.peak_scratch_end << '}';
  return out.str();
}

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help") {
      result.help = true;
      result.ok = true;
      return result;
    }
    if (i + 1 >= argc) {
      result.error = "missing value for " + option;
      return result;
    }
    const std::string value = argv[++i];
    if (option == "--program") {
      result.program_path = value;
    } else if (option == "--weights") {
      result.weights_path = value;
    } else if (option == "--input") {
      result.input_text = value;
    } else if (option == "--trace") {
      result.trace_path = value;
    } else {
      result.error = "unknown option " + option;
      return result;
    }
  }
  if (result.program_path.empty() || result.weights_path.empty() ||
      result.input_text.empty()) {
    result.error = "--program, --weights, and --input are required";
    return result;
  }
  result.ok = true;
  return result;
}

std::string usage() {
  return "tinylpu_plan_sim --program PATH --weights PATH --input 'comma signed ints' [--trace FILE]";
}

}  // namespace tinylpu_plan
