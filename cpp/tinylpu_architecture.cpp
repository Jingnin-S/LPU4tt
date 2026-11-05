#include "tinylpu_architecture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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
constexpr std::size_t kScratchBytes = 256;

thread_local char g_last_error[256] = {0};
// Avoid relying on a C++ TLS destructor across shared-library boundaries.
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
  unsigned encoding = 0;
  unsigned lanes = 0;
  unsigned depth = 0;
  unsigned acc_bits = 0;
  unsigned banks = 0;
  unsigned flat_read_ports = 0;
  unsigned read_latency = 0;
  unsigned copy_bytes_per_cycle = 0;
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
  std::uint64_t command_count = 0;
  std::uint64_t command_bytes = 0;
  std::uint64_t mvm_count = 0;
  std::uint64_t macs = 0;
  std::uint64_t load_bytes = 0;
  std::uint64_t store_bytes = 0;
  std::uint64_t operand_read_bytes = 0;
  std::uint64_t gather_cycles = 0;
  std::uint64_t conflict_cycles = 0;
  std::uint64_t memory_latency_cycles = 0;
  std::uint64_t pipeline_drain_cycles = 0;
  std::uint64_t core_cycles = 0;
  std::uint64_t scratch_cycles = 0;
  std::uint64_t decode_cycles = 0;
  std::uint64_t device_cycles = 0;
  std::uint64_t peak_scratch_end = 0;
};

std::size_t ceil_div(std::size_t value, unsigned divisor) {
  return (value + divisor - 1u) / divisor;
}

std::size_t checked_end(std::size_t start, std::size_t count,
                        const char *message) {
  if (start > kScratchBytes || count > kScratchBytes - start) {
    fail(message);
  }
  return start + count;
}

bool overlaps(std::size_t left, std::size_t left_size, std::size_t right,
              std::size_t right_size) {
  return left < right + right_size && right < left + left_size;
}

void validate_config(const Config &config) {
  if (config.encoding != 4u && config.encoding != 8u) {
    fail("encoding must be 4 or 8");
  }
  if (config.lanes != 1u && config.lanes != 2u && config.lanes != 4u) {
    fail("lanes must be 1, 2, or 4");
  }
  if (config.depth < 2u || config.depth > 4u) {
    fail("depth must be in [2,4]");
  }
  if (config.acc_bits != 16u && config.acc_bits != 32u) {
    fail("acc_bits must be 16 or 32");
  }
  if (config.banks != 0u && config.banks != 1u && config.banks != 2u &&
      config.banks != 4u) {
    fail("banks must be one of 0,1,2,4");
  }
  if (config.flat_read_ports < 1u || config.flat_read_ports > 5u) {
    fail("flat_read_ports must be in [1,5]");
  }
  if (config.read_latency > 2u) {
    fail("read_latency must be in [0,2]");
  }
  if (config.copy_bytes_per_cycle < 1u ||
      config.copy_bytes_per_cycle > 4u) {
    fail("copy_bytes_per_cycle must be in [1,4]");
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
      if (row[1] != 0u ||
          (encoding == 8u &&
           (row[4] != 0u || row[5] != 0u || row[6] != 0u || row[7] != 0u))) {
        fail("reserved LOAD/STORE fields are nonzero");
      }
      command.addr = row[2];
      command.count = static_cast<unsigned>(row[3]) + 1u;
    } else if (op == kMvm) {
      const unsigned flags = row[1];
      const unsigned format = (flags >> 2u) & 3u;
      if ((flags & 0xf0u) != 0u || format > 1u || row[5] > 15u ||
          row[6] > 63u || row[7] == 0u) {
        fail("reserved or invalid MVM fields");
      }
      command.flags = flags;
      command.a = row[2];
      command.w = row[3];
      command.y = row[4];
      command.rows = static_cast<unsigned>(row[5]) + 1u;
      command.k = static_cast<unsigned>(row[6]) + 1u;
      command.stride = row[7];
      command.input_bits = format == 0u ? 4u : 8u;
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

void update_peak(Report &report, std::size_t end) {
  report.peak_scratch_end =
      std::max(report.peak_scratch_end, static_cast<std::uint64_t>(end));
}

struct MvmRanges {
  std::size_t a_bytes = 0;
  std::size_t row_bytes = 0;
  std::size_t a_end = 0;
  std::size_t w_end = 0;
  std::size_t y_read_end = 0;
  std::size_t y_write_end = 0;
  unsigned accumulator_bytes = 0;
  unsigned output_bytes = 0;
};

MvmRanges validate_mvm(const Command &command, const Config &config,
                       Report &report) {
  const std::size_t row_bits =
      static_cast<std::size_t>(command.k) * command.input_bits;
  const std::size_t row_bytes = (row_bits + 7u) / 8u;
  if (command.stride < row_bytes) {
    fail("MVM W stride is shorter than packed row");
  }
  const unsigned accumulator_bytes = config.acc_bits / 8u;
  const unsigned output_bytes = command.sat16 ? 2u : accumulator_bytes;
  if (command.y % output_bytes != 0u ||
      (command.accumulate && command.y % accumulator_bytes != 0u)) {
    fail("MVM Y address is not naturally aligned");
  }

  MvmRanges ranges;
  ranges.a_bytes = row_bytes;
  ranges.row_bytes = row_bytes;
  ranges.accumulator_bytes = accumulator_bytes;
  ranges.output_bytes = output_bytes;
  ranges.a_end = checked_end(command.a, row_bytes,
                             "MVM A range exceeds 256B scratch");
  const std::size_t last_row =
      static_cast<std::size_t>(command.rows - 1u) * command.stride;
  if (last_row > std::numeric_limits<std::size_t>::max() - row_bytes) {
    fail("MVM W range overflows size_t");
  }
  ranges.w_end = checked_end(command.w, last_row + row_bytes,
                             "MVM W range exceeds 256B scratch");
  const std::size_t y_read_bytes =
      command.accumulate
          ? static_cast<std::size_t>(command.rows) * accumulator_bytes
          : 0u;
  const std::size_t y_write_bytes =
      static_cast<std::size_t>(command.rows) * output_bytes;
  ranges.y_read_end = checked_end(command.y, y_read_bytes,
                                  "MVM Y partial range exceeds 256B scratch");
  ranges.y_write_end = checked_end(command.y, y_write_bytes,
                                   "MVM Y output range exceeds 256B scratch");
  const std::size_t y_union_end =
      std::max(ranges.y_read_end, ranges.y_write_end);
  if (overlaps(command.y, y_union_end - command.y, command.a,
               ranges.a_end - command.a) ||
      overlaps(command.y, y_union_end - command.y, command.w,
               ranges.w_end - command.w)) {
    fail("MVM Y read/write range aliases A or W");
  }
  update_peak(report, ranges.a_end);
  update_peak(report, ranges.w_end);
  update_peak(report, y_union_end);
  return ranges;
}

std::size_t gather_rounds(const std::vector<std::size_t> &addresses,
                          const Config &config) {
  if (addresses.empty()) {
    fail("MVM gather has no active operand addresses");
  }
  if (config.banks == 0u) {
    return ceil_div(addresses.size(), config.flat_read_ports);
  }
  std::vector<std::size_t> per_bank(config.banks, 0u);
  for (const std::size_t address : addresses) {
    ++per_bank[address % config.banks];
  }
  return *std::max_element(per_bank.begin(), per_bank.end());
}

void analyze_mvm(const Command &command, const Config &config,
                 const MvmRanges &ranges, Report &report) {
  for (unsigned row_start = 0; row_start < command.rows;
       row_start += config.lanes) {
    const unsigned active =
        std::min(config.lanes, command.rows - row_start);
    for (unsigned index = 0; index < command.k; ++index) {
      std::vector<std::size_t> addresses;
      addresses.reserve(static_cast<std::size_t>(active) + 1u);
      const std::size_t byte_index =
          command.input_bits == 4u ? index / 2u : index;
      addresses.push_back(static_cast<std::size_t>(command.a) + byte_index);
      for (unsigned lane = 0; lane < active; ++lane) {
        const std::size_t row = row_start + lane;
        addresses.push_back(static_cast<std::size_t>(command.w) +
                            row * command.stride + byte_index);
      }
      std::sort(addresses.begin(), addresses.end());
      addresses.erase(std::unique(addresses.begin(), addresses.end()),
                      addresses.end());
      report.operand_read_bytes += addresses.size();
      const std::size_t rounds = gather_rounds(addresses, config);
      report.gather_cycles += rounds;
      report.conflict_cycles += rounds - 1u;
    }
    report.memory_latency_cycles += config.read_latency;
    report.pipeline_drain_cycles += config.depth - 1u;

    // Partial-sum reads and final output writes are serialized scratchpad
    // phases.  They are counted per row group, independent of gather ports.
    if (command.accumulate) {
      report.scratch_cycles += ceil_div(
          static_cast<std::size_t>(active) * ranges.accumulator_bytes,
          config.copy_bytes_per_cycle);
    }
    report.scratch_cycles += ceil_div(
        static_cast<std::size_t>(active) * ranges.output_bytes,
        config.copy_bytes_per_cycle);
  }
  report.macs += static_cast<std::uint64_t>(command.rows) * command.k;
}

void analyze(const std::vector<Command> &commands, const Config &config,
             std::size_t command_bytes, Report &report) {
  for (const Command &command : commands) {
    ++report.command_count;
    switch (command.op) {
      case kLoad:
        checked_end(command.addr, command.count,
                    "LOAD range exceeds 256B scratch");
        report.load_bytes += command.count;
        report.scratch_cycles +=
            ceil_div(command.count, config.copy_bytes_per_cycle);
        update_peak(report, static_cast<std::size_t>(command.addr) +
                              command.count);
        break;
      case kStore:
        checked_end(command.addr, command.count,
                    "STORE range exceeds 256B scratch");
        report.store_bytes += command.count;
        report.scratch_cycles +=
            ceil_div(command.count, config.copy_bytes_per_cycle);
        update_peak(report, static_cast<std::size_t>(command.addr) +
                              command.count);
        break;
      case kMvm: {
        const MvmRanges ranges = validate_mvm(command, config, report);
        ++report.mvm_count;
        analyze_mvm(command, config, ranges, report);
        break;
      }
      case kEnd:
        break;
      default:
        fail("internal invalid opcode");
    }
  }
  report.command_bytes = command_bytes;
  report.decode_cycles = report.command_count;
  report.core_cycles = report.gather_cycles + report.memory_latency_cycles +
                       report.pipeline_drain_cycles;
  report.device_cycles = report.decode_cycles + report.core_cycles +
                         report.scratch_cycles;
}

std::string make_json(const Report &report) {
  std::ostringstream stream;
  stream << "{\"command_count\":" << report.command_count
         << ",\"command_bytes\":" << report.command_bytes
         << ",\"mvm_count\":" << report.mvm_count
         << ",\"macs\":" << report.macs
         << ",\"load_bytes\":" << report.load_bytes
         << ",\"store_bytes\":" << report.store_bytes
         << ",\"operand_read_bytes\":" << report.operand_read_bytes
         << ",\"gather_cycles\":" << report.gather_cycles
         << ",\"conflict_cycles\":" << report.conflict_cycles
         << ",\"memory_latency_cycles\":"
         << report.memory_latency_cycles
         << ",\"pipeline_drain_cycles\":"
         << report.pipeline_drain_cycles
         << ",\"core_cycles\":" << report.core_cycles
         << ",\"scratch_cycles\":" << report.scratch_cycles
         << ",\"decode_cycles\":" << report.decode_cycles
         << ",\"device_cycles\":" << report.device_cycles
         << ",\"peak_scratch_end\":" << report.peak_scratch_end << "}";
  return stream.str();
}

}  // namespace

extern "C" const char *tlpu_arch_analyze(
    const unsigned char *commands, std::size_t command_bytes,
    unsigned encoding, unsigned lanes, unsigned depth, unsigned acc_bits,
    unsigned banks, unsigned flat_read_ports, unsigned read_latency,
    unsigned copy_bytes_per_cycle) {
  clear_error();
  try {
    const Config config{encoding, lanes, depth, acc_bits, banks,
                        flat_read_ports, read_latency, copy_bytes_per_cycle};
    validate_config(config);
    if (commands == nullptr && command_bytes != 0u) {
      fail("commands pointer is null for nonempty stream");
    }
    const std::vector<Command> decoded =
        decode_commands(commands, command_bytes, encoding);
    Report report;
    analyze(decoded, config, command_bytes, report);
    if (g_json == nullptr) {
      g_json = new std::string();
    }
    *g_json = make_json(report);
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
    set_error("unexpected exception in tlpu_arch_analyze");
    if (g_json != nullptr) {
      g_json->clear();
    }
    return nullptr;
  }
}

extern "C" const char *tlpu_arch_last_error(void) { return g_last_error; }
