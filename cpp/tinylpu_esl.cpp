#include "tinylpu_esl.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinylpu {
namespace {

// The CLI is an executable timing experiment, so reject configurations that
// would require an impractical or overflowing simulation before any tick runs.
constexpr std::int64_t kMaxExperimentCycles = 10'000'000;

std::string trim(std::string value) {
  const auto first = std::find_if(value.begin(), value.end(), [](unsigned char c) {
    return !std::isspace(c);
  });
  const auto last = std::find_if(value.rbegin(), value.rend(), [](unsigned char c) {
    return !std::isspace(c);
  }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string token = trim(text.substr(begin, end == std::string::npos
                                                        ? std::string::npos
                                                        : end - begin));
    if (token.empty()) {
      throw std::invalid_argument("empty integer in comma-separated list");
    }
    std::size_t consumed = 0;
    int value = 0;
    try {
      value = std::stoi(token, &consumed, 10);
    } catch (const std::exception&) {
      throw std::invalid_argument("invalid integer '" + token + "'");
    }
    if (consumed != token.size()) {
      throw std::invalid_argument("invalid integer '" + token + "'");
    }
    values.push_back(value);
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return values;
}

std::string command_name(CommandKind kind) {
  switch (kind) {
    case CommandKind::Noop:
      return "idle";
    case CommandKind::WriteA:
      return "write_a";
    case CommandKind::WriteW:
      return "write_w";
    case CommandKind::Start:
      return "start";
    case CommandKind::ReadResult:
      return "read_result";
  }
  return "unknown";
}

void append_int_array(std::ostringstream& out, const std::vector<std::int16_t>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << static_cast<int>(values[i]);
  }
  out << ']';
}

}  // namespace

std::string kernel_name(Kernel kernel) {
  return kernel == Kernel::Dot4 ? "dot4" : "gemv2x4";
}

int kernel_rows(Kernel kernel) {
  return kernel == Kernel::Dot4 ? 1 : 2;
}

ValidationResult validate_inputs(Kernel kernel,
                                 const std::vector<int>& a,
                                 const std::vector<int>& weights,
                                 int reuse,
                                 int host_wait) {
  const int rows = kernel_rows(kernel);
  if (a.size() != kK) {
    return {false, "A must contain exactly 4 signed INT4 values"};
  }
  if (weights.size() != static_cast<std::size_t>(rows * kK)) {
    return {false, "W must contain exactly rows*4 signed INT4 values"};
  }
  if (reuse <= 0) {
    return {false, "reuse must be positive"};
  }
  if (host_wait < 0) {
    return {false, "host-wait must be non-negative"};
  }
  const std::int64_t weight_bytes = static_cast<std::int64_t>(rows) * 2;
  const std::int64_t per_transaction = 2 + static_cast<std::int64_t>(host_wait) + 1 +
                                        5 * static_cast<std::int64_t>(rows) + weight_bytes;
  const std::int64_t total_cycles = weight_bytes +
                                    static_cast<std::int64_t>(reuse) * per_transaction;
  if (total_cycles > kMaxExperimentCycles ||
      total_cycles > std::numeric_limits<int>::max()) {
    return {false, "configuration exceeds the 10000000-cycle ESL simulation limit"};
  }
  const auto check_range = [](const std::vector<int>& values, const char* name)
      -> ValidationResult {
    for (int value : values) {
      if (value < -8 || value > 7) {
        return {false, std::string(name) + " value " + std::to_string(value) +
                         " is outside signed INT4 [-8,7]"};
      }
    }
    return {true, {}};
  };
  ValidationResult a_check = check_range(a, "A");
  if (!a_check.ok) {
    return a_check;
  }
  return check_range(weights, "W");
}

std::int8_t decode_int4(std::uint8_t nibble) {
  const std::uint8_t value = nibble & 0x0fU;
  return static_cast<std::int8_t>(value >= 8U ? static_cast<int>(value) - 16
                                               : static_cast<int>(value));
}

std::uint8_t encode_int4(int value) {
  if (value < -8 || value > 7) {
    throw std::out_of_range("signed INT4 value outside [-8,7]");
  }
  return static_cast<std::uint8_t>(value) & 0x0fU;
}

std::uint8_t pack_int4(int low_index, int high_index) {
  return static_cast<std::uint8_t>(encode_int4(low_index) |
                                   (encode_int4(high_index) << 4U));
}

std::array<std::int8_t, 2> unpack_int4(std::uint8_t payload) {
  return {decode_int4(payload & 0x0fU), decode_int4(payload >> 4U)};
}

std::int8_t FunctionalModel::product(int a, int b) {
  if (a < -8 || a > 7 || b < -8 || b > 7) {
    throw std::out_of_range("INT4 product operand outside [-8,7]");
  }
  // The signed INT4 product is guaranteed to fit in signed 8 bits.
  return static_cast<std::int8_t>(a * b);
}

std::vector<std::int16_t> FunctionalModel::evaluate(
    Kernel kernel,
    const Vec4& a,
    const std::vector<std::int8_t>& weights) {
  const int rows = kernel_rows(kernel);
  for (std::int8_t value : a) {
    if (value < -8 || value > 7) {
      throw std::out_of_range("A operand outside signed INT4");
    }
  }
  if (weights.size() != static_cast<std::size_t>(rows * kK)) {
    throw std::invalid_argument("wrong weight count for kernel");
  }
  for (std::int8_t value : weights) {
    if (value < -8 || value > 7) {
      throw std::out_of_range("W operand outside signed INT4");
    }
  }
  std::vector<std::int16_t> result(static_cast<std::size_t>(rows), 0);
  for (int row = 0; row < rows; ++row) {
    int acc = 0;
    for (int index = 0; index < kK; ++index) {
      acc += product(a[static_cast<std::size_t>(index)],
                     weights[static_cast<std::size_t>(row * kK + index)]);
    }
    result[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(acc);
  }
  return result;
}

CycleModel::CycleModel(int rows)
    : rows_(rows),
      weights_(static_cast<std::size_t>(rows * kK), 0),
      results_(static_cast<std::size_t>(rows), 0) {
  if (rows < 1 || rows > 2) {
    throw std::invalid_argument("CycleModel supports one or two rows");
  }
  reset();
}

void CycleModel::set_operands(const Vec4& a, const std::vector<std::int8_t>& weights) {
  if (weights.size() != static_cast<std::size_t>(rows_ * kK)) {
    throw std::invalid_argument("wrong weight count for CycleModel");
  }
  for (std::int8_t value : a) {
    if (value < -8 || value > 7) {
      throw std::out_of_range("A operand outside signed INT4");
    }
  }
  for (std::int8_t value : weights) {
    if (value < -8 || value > 7) {
      throw std::out_of_range("W operand outside signed INT4");
    }
  }
  a_ = a;
  weights_ = weights;
}

void CycleModel::reset() {
  std::fill(results_.begin(), results_.end(), 0);
  acc_ = 0;
  relative_ = 0;
  running_ = false;
  done_ = false;
  pending_valid_ = false;
  pending_row_ = -1;
  pending_index_ = -1;
  pending_product_ = 0;
}

void CycleModel::start() {
  std::fill(results_.begin(), results_.end(), 0);
  acc_ = 0;
  relative_ = 0;
  running_ = true;
  done_ = false;
  pending_valid_ = false;
  pending_row_ = -1;
  pending_index_ = -1;
  pending_product_ = 0;
}

CycleStep CycleModel::tick() {
  CycleStep step;
  step.relative = relative_;
  step.acc = acc_;
  if (!running_) {
    step.done = done_;
    return step;
  }

  step.relative = ++relative_;
  const int row = (relative_ - 1) / 5;
  const int slot = (relative_ - 1) % 5;
  if (slot == 0) {
    // The first issue edge of every row starts a fresh reduction. START also
    // clears row zero, so this assignment is deliberately idempotent there.
    acc_ = 0;
  }

  if (pending_valid_) {
    step.commit = true;
    step.commit_row = pending_row_;
    step.commit_index = pending_index_;
    step.commit_product = static_cast<int>(pending_product_);
    acc_ = static_cast<std::int16_t>(acc_ + pending_product_);
    pending_valid_ = false;
    if (pending_index_ == kK - 1) {
      results_[static_cast<std::size_t>(pending_row_)] = acc_;
      step.store = true;
    }
  }

  if (slot < kK) {
    const int index = slot;
    const int product = static_cast<int>(a_[static_cast<std::size_t>(index)]) *
                        static_cast<int>(weights_[static_cast<std::size_t>(row * kK + index)]);
    step.issue = true;
    step.issue_row = row;
    step.issue_index = index;
    step.issue_product = product;
    pending_valid_ = true;
    pending_row_ = row;
    pending_index_ = index;
    pending_product_ = static_cast<std::int8_t>(product);
  }

  step.acc = acc_;
  if (step.store && row == rows_ - 1) {
    running_ = false;
    done_ = true;
    step.done = true;
  }
  return step;
}

HostModel::HostModel(Kernel kernel)
    : kernel_(kernel),
      rows_(kernel_rows(kernel)),
      core_(rows_),
      weights_(static_cast<std::size_t>(rows_ * kK), 0),
      weight_valid_(static_cast<std::size_t>(rows_ * 2), false),
      results_(static_cast<std::size_t>(rows_), 0) {
  clear_state();
}

bool HostModel::a_valid() const {
  return a_valid_[0] && a_valid_[1];
}

bool HostModel::weights_valid() const {
  return std::all_of(weight_valid_.begin(), weight_valid_.end(), [](bool value) {
    return value;
  });
}

HostResponse HostModel::reject(const std::string& message) {
  HostResponse response;
  response.rejected = true;
  response.error = true;
  response.message = message;
  return response;
}

void HostModel::clear_state() {
  a_.fill(0);
  std::fill(weights_.begin(), weights_.end(), 0);
  std::fill(a_valid_.begin(), a_valid_.end(), false);
  std::fill(weight_valid_.begin(), weight_valid_.end(), false);
  std::fill(results_.begin(), results_.end(), 0);
  busy_ = false;
  done_ = false;
  core_.reset();
}

void HostModel::emit(TraceRow row) {
  if (sink_) {
    sink_(row);
  }
}

HostResponse HostModel::tick(bool rst_n, bool ena, HostCommand command) {
  TraceRow trace;
  trace.cycle = cycle_;
  HostResponse response;
  const bool pre_busy = busy_;

  if (!rst_n) {
    clear_state();
    trace.phase = "reset";
    trace.action = "reset";
    trace.busy_before = pre_busy;
    trace.busy = false;
    trace.done = false;
    emit(trace);
    ++cycle_;
    return response;
  }
  if (!ena) {
    clear_state();
    trace.phase = "disabled";
    trace.action = "ena_low";
    trace.busy_before = pre_busy;
    trace.busy = false;
    trace.done = false;
    emit(trace);
    ++cycle_;
    return response;
  }

  trace.busy_before = pre_busy;
  if (pre_busy) {
    trace.phase = "compute";
    trace.action = "compute";
    if (command.kind != CommandKind::Noop) {
      response = reject("busy");
    }
    const CycleStep step = core_.tick();
    trace.issue = step.issue;
    trace.commit = step.commit;
    trace.row = step.issue ? step.issue_row : step.commit_row;
    trace.index = step.issue ? step.issue_index : step.commit_index;
    trace.commit_row = step.commit_row;
    trace.commit_index = step.commit_index;
    trace.issue_product = step.issue_product;
    trace.commit_product = step.commit_product;
    trace.product = step.commit ? step.commit_product : step.issue_product;
    trace.acc = step.acc;
    if (step.done) {
      busy_ = false;
      done_ = true;
      results_ = core_.results();
      trace.done = true;
    }
    // This reports the pre-tick busy state, making the final commit edge
    // visibly busy even though state becomes DONE after that edge.
    trace.busy = busy_;
    emit(trace);
    ++cycle_;
    return response;
  }

  trace.phase = "idle";
  trace.action = command_name(command.kind);
  switch (command.kind) {
    case CommandKind::Noop:
      break;

    case CommandKind::WriteA: {
      trace.index = command.address;
      if (command.address > 1U) {
        response = reject("invalid A byte address");
        break;
      }
      const auto values = unpack_int4(command.payload);
      const std::size_t base = static_cast<std::size_t>(command.address) * 2U;
      a_[base] = values[0];
      a_[base + 1U] = values[1];
      a_valid_[command.address] = true;
      response.accepted = true;
      break;
    }

    case CommandKind::WriteW: {
      trace.index = command.address;
      if (command.address >= weight_valid_.size()) {
        response = reject("invalid W byte address");
        break;
      }
      const auto values = unpack_int4(command.payload);
      const std::size_t base = static_cast<std::size_t>(command.address) * 2U;
      weights_[base] = values[0];
      weights_[base + 1U] = values[1];
      weight_valid_[command.address] = true;
      response.accepted = true;
      break;
    }

    case CommandKind::Start:
      trace.phase = "setup";
      if (!a_valid() || !weights_valid()) {
        response = reject("START requires all A and W bytes");
        break;
      }
      core_.set_operands(a_, weights_);
      core_.start();
      busy_ = true;
      done_ = false;
      std::fill(results_.begin(), results_.end(), 0);
      response.accepted = true;
      break;

    case CommandKind::ReadResult: {
      trace.phase = "read";
      trace.index = command.address;
      if (!done_) {
        response = reject("result is not DONE");
        break;
      }
      if (command.address >= static_cast<std::uint8_t>(rows_ * 2)) {
        response = reject("invalid result byte address");
        break;
      }
      const int row = command.address / 2;
      const std::uint16_t bits = static_cast<std::uint16_t>(results_[static_cast<std::size_t>(row)]);
      response.data = command.address % 2U == 0U
                          ? static_cast<std::uint8_t>(bits & 0xffU)
                          : static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
      response.accepted = true;
      break;
    }
  }

  trace.busy = busy_;
  trace.done = done_;
  emit(trace);
  ++cycle_;
  return response;
}

ExperimentReport run_experiment(const ExperimentConfig& config) {
  const ValidationResult validation =
      validate_inputs(config.kernel, config.a, config.weights, config.reuse, config.host_wait);
  if (!validation.ok) {
    throw std::invalid_argument(validation.error);
  }

  const int rows = kernel_rows(config.kernel);
  Vec4 a{};
  for (int index = 0; index < kK; ++index) {
    a[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(config.a[static_cast<std::size_t>(index)]);
  }
  std::vector<std::int8_t> weights(config.weights.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<std::int8_t>(config.weights[index]);
  }
  const std::vector<std::int16_t> golden = FunctionalModel::evaluate(config.kernel, a, weights);

  std::ofstream trace_file;
  HostModel host(config.kernel);
  if (!config.trace_path.empty()) {
    trace_file.open(config.trace_path, std::ios::out | std::ios::trunc);
    if (!trace_file) {
      throw std::runtime_error("cannot open trace file '" + config.trace_path + "'");
    }
    trace_file << "cycle,phase,action,row,index,commit_row,commit_index,issue,commit,product,issue_product,commit_product,acc,busy_before,busy,done\n";
    host.set_trace_sink([&trace_file](const TraceRow& row) {
      trace_file << row.cycle << ',' << row.phase << ',' << row.action << ',' << row.row << ','
                 << row.index << ',' << row.commit_row << ',' << row.commit_index << ','
                 << (row.issue ? 1 : 0) << ',' << (row.commit ? 1 : 0) << ',' << row.product << ','
                 << row.issue_product << ',' << row.commit_product << ',' << row.acc << ','
                 << (row.busy_before ? 1 : 0) << ',' << (row.busy ? 1 : 0) << ','
                 << (row.done ? 1 : 0) << '\n';
    });
  }

  const auto require_accept = [](const HostResponse& response, const char* what) {
    if (!response.accepted) {
      throw std::runtime_error(std::string(what) + " rejected: " + response.message);
    }
  };

  const int weight_bytes = rows * 2;
  for (int address = 0; address < weight_bytes; ++address) {
    const int low = config.weights[static_cast<std::size_t>(address * 2)];
    const int high = config.weights[static_cast<std::size_t>(address * 2 + 1)];
    require_accept(host.tick(true, true, {CommandKind::WriteW,
                                          static_cast<std::uint8_t>(address),
                                          pack_int4(low, high)}),
                   "WriteW");
  }

  std::vector<std::int16_t> cycle_result;
  for (int transaction = 0; transaction < config.reuse; ++transaction) {
    require_accept(host.tick(true, true,
                             {CommandKind::WriteA, 0, pack_int4(config.a[0], config.a[1])}),
                   "WriteA[0]");
    require_accept(host.tick(true, true,
                             {CommandKind::WriteA, 1, pack_int4(config.a[2], config.a[3])}),
                   "WriteA[1]");
    for (int wait = 0; wait < config.host_wait; ++wait) {
      host.tick(true, true, {});
    }
    require_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "START");

    int run_cycles = 0;
    while (!host.done()) {
      host.tick(true, true, {});
      ++run_cycles;
      if (run_cycles > 5 * rows) {
        throw std::runtime_error("cycle model failed to reach DONE at its fixed latency");
      }
    }
    if (run_cycles != 5 * rows) {
      throw std::runtime_error("cycle model RUN length does not match 5*M contract");
    }

    cycle_result.assign(static_cast<std::size_t>(rows), 0);
    for (int row = 0; row < rows; ++row) {
      std::uint16_t bits = 0;
      for (int byte = 0; byte < 2; ++byte) {
        const int address = row * 2 + byte;
        const HostResponse response = host.tick(true, true,
                                                {CommandKind::ReadResult,
                                                 static_cast<std::uint8_t>(address), 0});
        require_accept(response, "ReadResult");
        bits |= static_cast<std::uint16_t>(response.data) << (8 * byte);
      }
      cycle_result[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(bits);
    }
    if (cycle_result != golden) {
      throw std::runtime_error("cycle result differs from independent functional result");
    }
  }
  trace_file.flush();

  const int total_cycles = static_cast<int>(host.cycle());
  const std::int64_t expected_total = static_cast<std::int64_t>(weight_bytes) +
                                      static_cast<std::int64_t>(config.reuse) *
                                          (2 + config.host_wait + 1 + 5 * rows + weight_bytes);
  if (static_cast<std::int64_t>(total_cycles) != expected_total) {
    throw std::runtime_error("host cycle count does not match the endpoint contract");
  }

  ExperimentReport report;
  report.kernel = config.kernel;
  report.rows = rows;
  report.reuse = config.reuse;
  report.host_wait = config.host_wait;
  report.functional_result = golden;
  report.cycle_result = cycle_result;
  report.transactions = config.reuse;
  report.macs = kK * rows * config.reuse;
  report.weight_load_cycles = weight_bytes;
  report.input_load_cycles = 2 * config.reuse;
  report.setup_cycles = config.reuse;
  report.compute_cycles = 5 * rows * config.reuse;
  report.readout_cycles = weight_bytes * config.reuse;
  report.host_wait_cycles = config.host_wait * config.reuse;
  report.total_cycles = total_cycles;
  return report;
}

std::string report_json(const ExperimentReport& report) {
  std::ostringstream out;
  out << "{\"kernel\":\"" << kernel_name(report.kernel) << "\","
      << "\"rows\":" << report.rows << ','
      << "\"k\":" << report.k << ','
      << "\"reuse\":" << report.reuse << ','
      << "\"host_wait\":" << report.host_wait << ','
      << "\"functional_result\":";
  append_int_array(out, report.functional_result);
  out << ",\"cycle_result\":";
  append_int_array(out, report.cycle_result);
  out << ",\"transactions\":" << report.transactions
      << ",\"macs\":" << report.macs
      << ",\"weight_load_cycles\":" << report.weight_load_cycles
      << ",\"input_load_cycles\":" << report.input_load_cycles
      << ",\"setup_cycles\":" << report.setup_cycles
      << ",\"compute_cycles\":" << report.compute_cycles
      << ",\"readout_cycles\":" << report.readout_cycles
      << ",\"host_wait_cycles\":" << report.host_wait_cycles
      << ",\"total_cycles\":" << report.total_cycles
      << ",\"peak_macs_per_cycle\":" << report.peak_macs_per_cycle
      << ",\"core_macs_per_cycle\":" << std::fixed << std::setprecision(1)
      << report.core_macs_per_cycle << "}";
  return out.str();
}

std::string usage() {
  return "tinylpu_esl --kernel dot4|gemv2x4 --a '1,2,3,4' "
         "--w '1,2,3,4[,..]' --reuse N --host-wait H [--trace FILE]";
}

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result;
  bool seen_kernel = false;
  bool seen_a = false;
  bool seen_w = false;
  bool seen_reuse = false;
  bool seen_wait = false;
  auto need_value = [&](int& index, const std::string& option) -> std::string {
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    return argv[++index];
  };

  try {
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--help" || option == "-h") {
        result.help = true;
        result.ok = true;
        return result;
      }
      if (option == "--kernel") {
        const std::string value = need_value(index, option);
        if (value == "dot4") {
          result.config.kernel = Kernel::Dot4;
        } else if (value == "gemv2x4") {
          result.config.kernel = Kernel::Gemv2x4;
        } else {
          throw std::invalid_argument("kernel must be dot4 or gemv2x4");
        }
        seen_kernel = true;
      } else if (option == "--a") {
        result.config.a = parse_int_list(need_value(index, option));
        seen_a = true;
      } else if (option == "--w") {
        result.config.weights = parse_int_list(need_value(index, option));
        seen_w = true;
      } else if (option == "--reuse") {
        const std::string value = need_value(index, option);
        std::size_t consumed = 0;
        result.config.reuse = std::stoi(value, &consumed, 10);
        if (consumed != value.size()) {
          throw std::invalid_argument("reuse must be an integer");
        }
        seen_reuse = true;
      } else if (option == "--host-wait") {
        const std::string value = need_value(index, option);
        std::size_t consumed = 0;
        result.config.host_wait = std::stoi(value, &consumed, 10);
        if (consumed != value.size()) {
          throw std::invalid_argument("host-wait must be an integer");
        }
        seen_wait = true;
      } else if (option == "--trace") {
        result.config.trace_path = need_value(index, option);
      } else {
        throw std::invalid_argument("unknown option '" + option + "'");
      }
    }
    if (!seen_kernel || !seen_a || !seen_w || !seen_reuse || !seen_wait) {
      throw std::invalid_argument("kernel, A, W, reuse and host-wait are required");
    }
    const ValidationResult validation = validate_inputs(result.config.kernel,
                                                        result.config.a,
                                                        result.config.weights,
                                                        result.config.reuse,
                                                        result.config.host_wait);
    if (!validation.ok) {
      throw std::invalid_argument(validation.error);
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

}  // namespace tinylpu
