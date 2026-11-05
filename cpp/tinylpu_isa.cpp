#include "tinylpu_isa.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kLoad = 1;
constexpr unsigned kStore = 2;
constexpr unsigned kMvm = 3;
constexpr unsigned kEnd = 15;
constexpr unsigned kMaxScratch = 256;

thread_local char g_last_error[256] = {0};
// Keep the returned C string thread-local without relying on a TLS destructor
// (some MinGW runtimes tear down C++ TLS objects after the DLL/CRT boundary).
// The small per-thread allocation is intentionally retained until thread exit.
thread_local std::string *g_json = nullptr;

void clear_error() noexcept { g_last_error[0] = '\0'; }

void set_error(const char *message) noexcept {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

[[noreturn]] void fail(const char *message) {
  set_error(message);
  throw std::runtime_error(message);
}

struct Config {
  unsigned scratch_bytes;
  unsigned lanes;
  unsigned depth;
  unsigned accumulator_bits;
  unsigned memory_byte_bw;
  unsigned encoding;
  unsigned enable_int8;
};

struct Command {
  unsigned op = 0;
  unsigned flags = 0;
  unsigned addr = 0;
  unsigned count = 0;
  unsigned a = 0;
  unsigned w = 0;
  unsigned y = 0;
  unsigned rows = 0;
  unsigned k = 0;
  unsigned stride = 0;
  unsigned input_bits = 0;
  bool accumulate = false;
  bool sat16 = false;
};

struct Report {
  std::string stores_hex;
  std::uint64_t command_count = 0;
  std::uint64_t mvm_count = 0;
  std::uint64_t macs = 0;
  std::uint64_t core_cycles = 0;
  std::uint64_t scratch_cycles = 0;
  std::uint64_t decode_cycles = 0;
  std::uint64_t device_cycles = 0;
  std::uint64_t load_bytes = 0;
  std::uint64_t store_bytes = 0;
  std::uint64_t command_bytes = 0;
  std::uint64_t overflow_events = 0;
  std::uint64_t saturation_events = 0;
  std::uint64_t peak_scratch_end = 0;
};

struct State {
  explicit State(unsigned scratch_bytes)
      : bytes(scratch_bytes, 0), valid(scratch_bytes, false) {}

  std::vector<unsigned char> bytes;
  std::vector<unsigned char> valid;
  std::size_t load_cursor = 0;
  Report report;
};

bool overlaps(std::size_t left, std::size_t left_size, std::size_t right,
              std::size_t right_size) {
  return left < right + right_size && right < left + left_size;
}

std::size_t checked_end(std::size_t start, std::size_t count,
                        std::size_t capacity, const char *what) {
  if (start > capacity || count > capacity - start) {
    fail(what);
  }
  return start + count;
}

std::size_t ceil_div(std::size_t value, unsigned divisor) {
  return (value + divisor - 1u) / divisor;
}

void validate_config(const Config &config) {
  if (config.scratch_bytes < 32u || config.scratch_bytes > kMaxScratch) {
    fail("scratch_bytes must be in [32,256]");
  }
  if (config.lanes != 1u && config.lanes != 2u && config.lanes != 4u) {
    fail("lanes must be 1, 2, or 4");
  }
  if (config.depth < 2u || config.depth > 4u) {
    fail("depth must be in [2,4]");
  }
  if (config.accumulator_bits != 16u && config.accumulator_bits != 32u) {
    fail("accumulator_bits must be 16 or 32");
  }
  if (config.memory_byte_bw == 0u || config.memory_byte_bw > 16u) {
    fail("memory_byte_bw must be in [1,16]");
  }
  if (config.encoding != 4u && config.encoding != 8u) {
    fail("encoding must be 4 or 8");
  }
  if (config.enable_int8 > 1u) {
    fail("enable_int8 must be 0 or 1");
  }
}

std::vector<Command> decode_commands(const unsigned char *data,
                                     std::size_t byte_count,
                                     unsigned encoding) {
  if (data == nullptr || byte_count == 0u) {
    fail("command stream must be nonempty");
  }

  std::vector<Command> commands;
  std::size_t offset = 0;
  bool ended = false;
  while (offset < byte_count) {
    if (ended) {
      fail("trailing commands after END");
    }
    const unsigned op = data[offset];
    const std::size_t length = op == kMvm ? 8u : encoding;
    if (byte_count - offset < length) {
      fail("truncated command encoding");
    }
    const unsigned char *row = data + offset;
    Command command;
    command.op = op;
    if (op == kLoad || op == kStore) {
      if (row[1] != 0u || (encoding == 8u &&
                           (row[4] != 0u || row[5] != 0u || row[6] != 0u ||
                            row[7] != 0u))) {
        fail("reserved LOAD/STORE fields are nonzero");
      }
      command.addr = row[2];
      command.count = static_cast<unsigned>(row[3]) + 1u;
    } else if (op == kMvm) {
      const unsigned flags = row[1];
      if ((flags & 0xf0u) != 0u || ((flags >> 2u) & 3u) > 1u ||
          row[5] > 15u || row[6] > 63u || row[7] == 0u) {
        fail("reserved or invalid MVM fields");
      }
      command.flags = flags;
      command.a = row[2];
      command.w = row[3];
      command.y = row[4];
      command.rows = static_cast<unsigned>(row[5]) + 1u;
      command.k = static_cast<unsigned>(row[6]) + 1u;
      command.stride = row[7];
      command.input_bits = (flags & 4u) != 0u ? 8u : 4u;
      command.accumulate = (flags & 1u) != 0u;
      command.sat16 = (flags & 2u) != 0u;
    } else if (op == kEnd) {
      for (std::size_t index = 1; index < length; ++index) {
        if (row[index] != 0u) {
          fail("reserved END fields are nonzero");
        }
      }
      ended = true;
    } else {
      fail("invalid opcode");
    }
    commands.push_back(command);
    offset += length;
  }
  if (!ended) {
    fail("missing END command");
  }
  return commands;
}

void update_peak(State &state, std::size_t end) {
  state.report.peak_scratch_end =
      std::max(state.report.peak_scratch_end, static_cast<std::uint64_t>(end));
}

void check_valid(const State &state, std::size_t start, std::size_t count,
                 const char *what) {
  for (std::size_t index = 0; index < count; ++index) {
    if (!state.valid[start + index]) {
      fail(what);
    }
  }
}

void mark_valid(State &state, std::size_t start, std::size_t count) {
  std::fill(state.valid.begin() + static_cast<std::ptrdiff_t>(start),
            state.valid.begin() + static_cast<std::ptrdiff_t>(start + count),
            static_cast<unsigned char>(true));
}

std::int64_t sign_extend(std::uint64_t value, unsigned bits) {
  const std::uint64_t sign = std::uint64_t{1} << (bits - 1u);
  const std::uint64_t modulus = std::uint64_t{1} << bits;
  if ((value & sign) != 0u) {
    return static_cast<std::int64_t>(value) -
           static_cast<std::int64_t>(modulus);
  }
  return static_cast<std::int64_t>(value);
}

std::int64_t read_signed(const State &state, std::size_t start,
                         unsigned bytes) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < bytes; ++index) {
    value |= static_cast<std::uint64_t>(state.bytes[start + index])
             << (8u * index);
  }
  return sign_extend(value, bytes * 8u);
}

void write_signed(State &state, std::size_t start, unsigned bytes,
                  std::int64_t value) {
  const std::uint64_t raw = static_cast<std::uint64_t>(value);
  for (unsigned index = 0; index < bytes; ++index) {
    state.bytes[start + index] =
        static_cast<unsigned char>((raw >> (8u * index)) & 0xffu);
  }
  mark_valid(state, start, bytes);
}

std::int64_t wrap_signed(std::int64_t value, unsigned bits) {
  const std::uint64_t modulus = std::uint64_t{1} << bits;
  const std::uint64_t masked = static_cast<std::uint64_t>(value) & (modulus - 1u);
  return sign_extend(masked, bits);
}

std::int64_t unpack_operand(const State &state, std::size_t address,
                            std::size_t index, unsigned input_bits) {
  if (input_bits == 8u) {
    const unsigned raw = state.bytes[address + index];
    return raw >= 128u ? static_cast<std::int64_t>(raw) - 256 : raw;
  }
  const unsigned char packed = state.bytes[address + index / 2u];
  const unsigned nibble = index % 2u == 0u ? packed & 0x0fu : packed >> 4u;
  return nibble >= 8u ? static_cast<std::int64_t>(nibble) - 16 : nibble;
}

void append_hex_byte(std::string &output, unsigned char value) {
  static constexpr char digits[] = "0123456789abcdef";
  output.push_back(digits[value >> 4u]);
  output.push_back(digits[value & 0x0fu]);
}

void execute_load(const Command &command, const unsigned char *loads,
                  std::size_t load_bytes, State &state,
                  const Config &config) {
  const std::size_t end = checked_end(command.addr, command.count,
                                       config.scratch_bytes,
                                       "LOAD range exceeds scratch");
  if (state.load_cursor > load_bytes ||
      command.count > load_bytes - state.load_cursor) {
    fail("LOAD literal payload is missing");
  }
  if (command.count != 0u) {
    std::copy(loads + state.load_cursor, loads + state.load_cursor + command.count,
              state.bytes.begin() + static_cast<std::ptrdiff_t>(command.addr));
    mark_valid(state, command.addr, command.count);
  }
  state.load_cursor += command.count;
  state.report.load_bytes += command.count;
  state.report.scratch_cycles += ceil_div(command.count, config.memory_byte_bw);
  update_peak(state, end);
}

void execute_store(const Command &command, State &state, const Config &config) {
  const std::size_t end = checked_end(command.addr, command.count,
                                       config.scratch_bytes,
                                       "STORE range exceeds scratch");
  check_valid(state, command.addr, command.count,
              "STORE reads uninitialized scratch");
  for (unsigned index = 0; index < command.count; ++index) {
    append_hex_byte(state.report.stores_hex, state.bytes[command.addr + index]);
  }
  state.report.store_bytes += command.count;
  state.report.scratch_cycles += ceil_div(command.count, config.memory_byte_bw);
  update_peak(state, end);
}

struct MvmRanges {
  std::size_t a_bytes;
  std::size_t row_bytes;
  std::size_t w_end;
  std::size_t a_end;
  std::size_t y_read_end;
  std::size_t y_write_end;
  unsigned accumulator_bytes;
  unsigned output_bytes;
};

MvmRanges validate_mvm(const Command &command, const Config &config,
                       const State &state) {
  if (command.input_bits == 8u && config.enable_int8 == 0u) {
    fail("INT8 MVM requires enable_int8 profile");
  }
  const std::size_t row_bits =
      static_cast<std::size_t>(command.k) * command.input_bits;
  const std::size_t row_bytes = (row_bits + 7u) / 8u;
  const std::size_t a_bytes = row_bytes;
  if (command.stride < row_bytes) {
    fail("MVM W stride is shorter than packed row");
  }
  const unsigned accumulator_bytes = config.accumulator_bits / 8u;
  const unsigned output_bytes = command.sat16 ? 2u : accumulator_bytes;
  if (command.y % output_bytes != 0u ||
      (command.accumulate && command.y % accumulator_bytes != 0u)) {
    fail("MVM Y address is not naturally aligned");
  }

  MvmRanges ranges{a_bytes,
                   row_bytes,
                   0,
                   0,
                   0,
                   0,
                   accumulator_bytes,
                   output_bytes};
  ranges.a_end = checked_end(command.a, a_bytes, config.scratch_bytes,
                             "MVM A range exceeds scratch");
  if (command.rows > 0u) {
    const std::size_t last_row =
        static_cast<std::size_t>(command.rows - 1u) * command.stride;
    if (last_row > std::numeric_limits<std::size_t>::max() - row_bytes) {
      fail("MVM W range overflows size_t");
    }
    ranges.w_end = checked_end(command.w, last_row + row_bytes,
                               config.scratch_bytes,
                               "MVM W range exceeds scratch");
  }
  const std::size_t y_read_bytes =
      command.accumulate ? static_cast<std::size_t>(command.rows) * accumulator_bytes
                         : 0u;
  const std::size_t y_write_bytes =
      static_cast<std::size_t>(command.rows) * output_bytes;
  ranges.y_read_end = checked_end(command.y, y_read_bytes,
                                  config.scratch_bytes,
                                  "MVM Y partial range exceeds scratch");
  ranges.y_write_end = checked_end(command.y, y_write_bytes,
                                   config.scratch_bytes,
                                   "MVM Y output range exceeds scratch");
  const std::size_t y_union_end = std::max(ranges.y_read_end, ranges.y_write_end);
  if (overlaps(command.y, y_union_end - command.y, command.a,
               ranges.a_end - command.a) ||
      overlaps(command.y, y_union_end - command.y, command.w,
               ranges.w_end - command.w)) {
    fail("MVM Y read/write range aliases A or W");
  }

  check_valid(state, command.a, ranges.a_bytes,
              "MVM reads uninitialized A scratch");
  for (unsigned row = 0; row < command.rows; ++row) {
    check_valid(state, command.w + static_cast<std::size_t>(row) * command.stride,
                ranges.row_bytes, "MVM reads uninitialized W scratch");
  }
  if (command.accumulate) {
    check_valid(state, command.y, y_read_bytes,
                "MVM reads uninitialized ACCUM partial sums");
  }
  return ranges;
}

void execute_mvm(const Command &command, const Config &config, State &state,
                 const MvmRanges &ranges) {
  const unsigned accumulator_bits = config.accumulator_bits;
  const std::int64_t minimum = -(std::int64_t{1} << (accumulator_bits - 1u));
  const std::int64_t maximum = (std::int64_t{1} << (accumulator_bits - 1u)) - 1;
  const unsigned latency = config.depth - 1u;
  for (unsigned start = 0; start < command.rows; start += config.lanes) {
    const unsigned active = std::min(config.lanes, command.rows - start);
    std::vector<std::int64_t> accumulators(active, 0);
    if (command.accumulate) {
      for (unsigned lane = 0; lane < active; ++lane) {
        accumulators[lane] = read_signed(
            state, command.y + static_cast<std::size_t>(start + lane) *
                          ranges.accumulator_bytes,
            ranges.accumulator_bytes);
      }
      state.report.scratch_cycles +=
          ceil_div(static_cast<std::size_t>(active) * ranges.accumulator_bytes,
                   config.memory_byte_bw);
    }

    std::vector<std::deque<std::int64_t>> pipeline(active);
    const unsigned ticks = command.k + latency;
    for (unsigned tick = 0; tick < ticks; ++tick) {
      if (tick >= latency) {
        for (unsigned lane = 0; lane < active; ++lane) {
          if (pipeline[lane].empty()) {
            fail("internal MVM pipeline underflow");
          }
          const std::int64_t product = pipeline[lane].front();
          pipeline[lane].pop_front();
          const std::int64_t wide = accumulators[lane] + product;
          if (wide < minimum || wide > maximum) {
            ++state.report.overflow_events;
          }
          accumulators[lane] = wrap_signed(wide, accumulator_bits);
        }
      }
      if (tick < command.k) {
        for (unsigned lane = 0; lane < active; ++lane) {
          const std::size_t row = start + lane;
          const std::int64_t left =
              unpack_operand(state, command.a, tick, command.input_bits);
          const std::int64_t right = unpack_operand(
              state, command.w + row * command.stride, tick, command.input_bits);
          pipeline[lane].push_back(left * right);
        }
      }
    }

    for (unsigned lane = 0; lane < active; ++lane) {
      std::int64_t output = accumulators[lane];
      if (command.sat16) {
        const std::int64_t clipped = std::min<std::int64_t>(
            32767, std::max<std::int64_t>(-32768, output));
        if (clipped != output) {
          ++state.report.saturation_events;
        }
        output = clipped;
      }
      write_signed(state,
                   command.y + static_cast<std::size_t>(start + lane) *
                                  ranges.output_bytes,
                   ranges.output_bytes, output);
    }
    state.report.core_cycles += ticks;
    state.report.macs += static_cast<std::uint64_t>(active) * command.k;
    state.report.scratch_cycles +=
        ceil_div(static_cast<std::size_t>(active) * ranges.output_bytes,
                 config.memory_byte_bw);
  }
  update_peak(state, std::max(ranges.y_read_end, ranges.y_write_end));
  update_peak(state, ranges.a_end);
  update_peak(state, ranges.w_end);
}

void execute(const std::vector<Command> &commands, const unsigned char *loads,
             std::size_t load_bytes, const Config &config, State &state) {
  for (const Command &command : commands) {
    ++state.report.command_count;
    switch (command.op) {
      case kLoad:
        execute_load(command, loads, load_bytes, state, config);
        break;
      case kStore:
        execute_store(command, state, config);
        break;
      case kMvm: {
        const MvmRanges ranges = validate_mvm(command, config, state);
        ++state.report.mvm_count;
        execute_mvm(command, config, state, ranges);
        break;
      }
      case kEnd:
        break;
      default:
        fail("internal invalid opcode");
    }
  }
  if (state.load_cursor != load_bytes) {
    fail("extra LOAD literal payload");
  }
  state.report.decode_cycles = state.report.command_count;
  state.report.device_cycles = state.report.decode_cycles +
                               state.report.core_cycles +
                               state.report.scratch_cycles;
  state.report.command_bytes = 0;  // filled by the C entry point
}

std::string make_json(const Report &report, std::size_t command_bytes) {
  std::ostringstream stream;
  stream << "{\"stores_hex\":\"" << report.stores_hex
         << "\",\"command_count\":" << report.command_count
         << ",\"mvm_count\":" << report.mvm_count
         << ",\"macs\":" << report.macs
         << ",\"core_cycles\":" << report.core_cycles
         << ",\"scratch_cycles\":" << report.scratch_cycles
         << ",\"decode_cycles\":" << report.decode_cycles
         << ",\"device_cycles\":" << report.device_cycles
         << ",\"load_bytes\":" << report.load_bytes
         << ",\"store_bytes\":" << report.store_bytes
         << ",\"command_bytes\":" << command_bytes
         << ",\"overflow_events\":" << report.overflow_events
         << ",\"saturation_events\":" << report.saturation_events
         << ",\"peak_scratch_end\":" << report.peak_scratch_end << "}";
  return stream.str();
}

}  // namespace

extern "C" const char *tlpu_isa_run(
    const unsigned char *commands, std::size_t command_bytes,
    const unsigned char *loads, std::size_t load_bytes, unsigned scratch_bytes,
    unsigned lanes, unsigned depth, unsigned accumulator_bits,
    unsigned memory_byte_bw, unsigned encoding, unsigned enable_int8) {
  clear_error();
  try {
    const Config config{scratch_bytes, lanes, depth, accumulator_bits,
                        memory_byte_bw, encoding, enable_int8};
    validate_config(config);
    if (commands == nullptr && command_bytes != 0u) {
      fail("commands pointer is null for nonempty stream");
    }
    if (loads == nullptr && load_bytes != 0u) {
      fail("loads pointer is null for nonempty payload");
    }
    const std::vector<Command> decoded =
        decode_commands(commands, command_bytes, encoding);
    State state(scratch_bytes);
    execute(decoded, loads, load_bytes, config, state);
    if (g_json == nullptr) {
      g_json = new std::string();
    }
    *g_json = make_json(state.report, command_bytes);
    return g_json->c_str();
  } catch (const std::exception &error) {
    if (g_last_error[0] == '\0') {
      set_error(error.what());
    }
    if (g_json != nullptr) {
      g_json->clear();
    }
    return nullptr;
  } catch (...) {
    set_error("unexpected exception in tlpu_isa_run");
    if (g_json != nullptr) {
      g_json->clear();
    }
    return nullptr;
  }
}

extern "C" const char *tlpu_isa_last_error(void) { return g_last_error; }
