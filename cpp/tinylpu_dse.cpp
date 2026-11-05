#include "tinylpu_dse.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinylpu_dse {
namespace {

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

int parse_scalar(const std::string& text, const char* name) {
  std::size_t consumed = 0;
  int value = 0;
  try {
    value = std::stoi(text, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return value;
}

void append_int_array(std::ostringstream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << values[index];
  }
  out << ']';
}

void append_i16_array(std::ostringstream& out, const std::vector<std::int16_t>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << static_cast<int>(values[index]);
  }
  out << ']';
}

}  // namespace

ValidationResult validate_config(const Config& config) {
  if (config.rows < 1 || config.rows > 16) {
    return {false, "rows must be in [1,16]"};
  }
  if (config.k < 1 || config.k > 64) {
    return {false, "k must be in [1,64]"};
  }
  if (config.lanes != 1 && config.lanes != 2 && config.lanes != 4) {
    return {false, "lanes must be one of 1,2,4"};
  }
  if (config.depth < 2 || config.depth > 4) {
    return {false, "depth must be in [2,4]"};
  }
  return {true, {}};
}

ValidationResult validate_operands(const Config& config,
                                   const std::vector<int>& a,
                                   const std::vector<int>& weights) {
  const ValidationResult config_result = validate_config(config);
  if (!config_result.ok) {
    return config_result;
  }
  if (a.size() != static_cast<std::size_t>(config.k)) {
    return {false, "A must contain exactly K signed INT4 values"};
  }
  if (weights.size() != static_cast<std::size_t>(config.rows * config.k)) {
    return {false, "W must contain exactly M*K signed INT4 values"};
  }
  for (int value : a) {
    if (value < -8 || value > 7) {
      return {false, "A value is outside signed INT4 [-8,7]"};
    }
  }
  for (int value : weights) {
    if (value < -8 || value > 7) {
      return {false, "W value is outside signed INT4 [-8,7]"};
    }
  }
  return {true, {}};
}

std::vector<std::int16_t> golden(const Config& config,
                                 const std::vector<std::int8_t>& a,
                                 const std::vector<std::int8_t>& weights) {
  const ValidationResult config_result = validate_config(config);
  if (!config_result.ok) {
    throw std::invalid_argument(config_result.error);
  }
  if (a.size() != static_cast<std::size_t>(config.k) ||
      weights.size() != static_cast<std::size_t>(config.rows * config.k)) {
    throw std::invalid_argument("wrong operand count for golden");
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
  std::vector<std::int16_t> result(static_cast<std::size_t>(config.rows), 0);
  for (int row = 0; row < config.rows; ++row) {
    int sum = 0;
    for (int index = 0; index < config.k; ++index) {
      sum += static_cast<int>(a[static_cast<std::size_t>(index)]) *
             static_cast<int>(weights[static_cast<std::size_t>(row * config.k + index)]);
    }
    result[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(sum);
  }
  return result;
}

DseCore::DseCore(Config config) : config_(config) {
  const ValidationResult result = validate_config(config_);
  if (!result.ok) {
    throw std::invalid_argument(result.error);
  }
  groups_ = (config_.rows + config_.lanes - 1) / config_.lanes;
  group_stride_ = config_.row_drain ? config_.k + config_.depth - 1 : config_.k;
  a_.assign(static_cast<std::size_t>(config_.k), 0);
  weights_.assign(static_cast<std::size_t>(config_.rows * config_.k), 0);
  accumulators_.assign(static_cast<std::size_t>(config_.lanes), 0);
  results_.assign(static_cast<std::size_t>(config_.rows), 0);
  pipeline_.assign(static_cast<std::size_t>(config_.depth),
                   std::vector<PipeEntry>(static_cast<std::size_t>(config_.lanes)));
  reset();
}

void DseCore::set_operands(const std::vector<std::int8_t>& a,
                           const std::vector<std::int8_t>& weights) {
  if (a.size() != static_cast<std::size_t>(config_.k) ||
      weights.size() != static_cast<std::size_t>(config_.rows * config_.k)) {
    throw std::invalid_argument("wrong operand count for DseCore");
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

void DseCore::reset() {
  std::fill(accumulators_.begin(), accumulators_.end(), 0);
  std::fill(results_.begin(), results_.end(), 0);
  for (auto& slot : pipeline_) {
    for (auto& entry : slot) {
      entry = {};
    }
  }
  relative_ = 0;
  running_ = false;
  done_ = false;
}

void DseCore::start() {
  reset();
  running_ = true;
}

bool DseCore::issue_for(int relative, int lane, int& row, int& index) const {
  if (relative < 1) {
    return false;
  }
  const int zero_based = relative - 1;
  const int group = zero_based / group_stride_;
  const int slot = zero_based % group_stride_;
  if (group >= groups_ || slot >= config_.k) {
    return false;
  }
  row = group * config_.lanes + lane;
  if (row >= config_.rows) {
    return false;
  }
  index = slot;
  return true;
}

Tick DseCore::tick() {
  Tick tick;
  if (!running_) {
    tick.relative = relative_;
    tick.done = done_;
    return tick;
  }

  tick.relative = ++relative_;
  std::vector<LaneEvent> lane_events(static_cast<std::size_t>(config_.lanes));
  std::vector<bool> active(static_cast<std::size_t>(config_.lanes), false);
  for (int lane = 0; lane < config_.lanes; ++lane) {
    lane_events[static_cast<std::size_t>(lane)].relative = relative_;
    lane_events[static_cast<std::size_t>(lane)].lane = lane;
  }

  const int due_slot = relative_ % config_.depth;
  for (auto& entry : pipeline_[static_cast<std::size_t>(due_slot)]) {
    if (!entry.valid) {
      continue;
    }
    LaneEvent& event = lane_events[static_cast<std::size_t>(entry.lane)];
    active[static_cast<std::size_t>(entry.lane)] = true;
    event.commit = true;
    event.commit_row = entry.row;
    event.commit_index = entry.index;
    event.commit_product = entry.product;
    const std::int16_t next_acc = entry.first
                                      ? static_cast<std::int16_t>(entry.product)
                                      : static_cast<std::int16_t>(
                                            accumulators_[static_cast<std::size_t>(entry.lane)] +
                                            entry.product);
    accumulators_[static_cast<std::size_t>(entry.lane)] = next_acc;
    event.acc = next_acc;
    if (entry.last) {
      results_[static_cast<std::size_t>(entry.row)] = next_acc;
      event.store = true;
      if (entry.row == config_.rows - 1) {
        event.done = true;
        tick.done = true;
      }
    }
    entry = {};
  }

  for (int lane = 0; lane < config_.lanes; ++lane) {
    int row = -1;
    int index = -1;
    if (!issue_for(relative_, lane, row, index)) {
      continue;
    }
    LaneEvent& event = lane_events[static_cast<std::size_t>(lane)];
    active[static_cast<std::size_t>(lane)] = true;
    event.issue = true;
    event.issue_row = row;
    event.issue_index = index;
    event.issue_product = static_cast<int>(a_[static_cast<std::size_t>(index)]) *
                          static_cast<int>(weights_[static_cast<std::size_t>(row * config_.k + index)]);
    PipeEntry& destination = pipeline_[static_cast<std::size_t>(
        (relative_ + config_.depth - 1) % config_.depth)][static_cast<std::size_t>(lane)];
    destination.valid = true;
    destination.lane = lane;
    destination.row = row;
    destination.index = index;
    destination.product = event.issue_product;
    destination.first = index == 0;
    destination.last = index == config_.k - 1;
  }

  if (tick.done) {
    running_ = false;
    done_ = true;
  }
  for (int lane = 0; lane < config_.lanes; ++lane) {
    if (active[static_cast<std::size_t>(lane)]) {
      LaneEvent& event = lane_events[static_cast<std::size_t>(lane)];
      if (!event.commit) {
        event.acc = event.issue ? accumulators_[static_cast<std::size_t>(event.lane)] : 0;
      }
      tick.events.push_back(event);
    }
  }
  return tick;
}

std::string report_json(const DseReport& report) {
  std::ostringstream out;
  out << "{\"rows\":" << report.config.rows
      << ",\"k\":" << report.config.k
      << ",\"lanes\":" << report.config.lanes
      << ",\"depth\":" << report.config.depth
      << ",\"row_drain\":" << (report.config.row_drain ? "true" : "false")
      << ",\"macs\":" << report.macs
      << ",\"cycles\":" << report.cycles
      << ",\"issue_events\":" << report.issue_events
      << ",\"commit_events\":" << report.commit_events
      << ",\"a\":";
  append_int_array(out, report.a);
  out << ",\"w\":";
  append_int_array(out, report.weights);
  out << ",\"functional_result\":";
  append_i16_array(out, report.functional_result);
  out << ",\"cycle_result\":";
  append_i16_array(out, report.cycle_result);
  out << ",\"accumulator_bits\":" << report.accumulator_bits << '}';
  return out.str();
}

std::string usage() {
  return "tinylpu_dse_core --rows M --k K --lanes P --depth D --row-drain 0|1 "
         "--seed N [--a comma-list --w comma-list] [--trace FILE]";
}

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result;
  bool seen_rows = false;
  bool seen_k = false;
  bool seen_lanes = false;
  bool seen_depth = false;
  bool seen_drain = false;
  bool seen_a = false;
  bool seen_w = false;
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
      if (option == "--rows") {
        result.config.rows = parse_scalar(need_value(index, option), "rows");
        seen_rows = true;
      } else if (option == "--k") {
        result.config.k = parse_scalar(need_value(index, option), "k");
        seen_k = true;
      } else if (option == "--lanes") {
        result.config.lanes = parse_scalar(need_value(index, option), "lanes");
        seen_lanes = true;
      } else if (option == "--depth") {
        result.config.depth = parse_scalar(need_value(index, option), "depth");
        seen_depth = true;
      } else if (option == "--row-drain") {
        const int value = parse_scalar(need_value(index, option), "row-drain");
        if (value != 0 && value != 1) {
          throw std::invalid_argument("row-drain must be 0 or 1");
        }
        result.config.row_drain = value != 0;
        seen_drain = true;
      } else if (option == "--seed") {
        const std::string value = need_value(index, option);
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size() || parsed > 0xffffffffULL) {
          throw std::invalid_argument("seed must be an unsigned 32-bit integer");
        }
        result.seed = static_cast<std::uint32_t>(parsed);
        result.seed_set = true;
      } else if (option == "--a") {
        result.a = parse_int_list(need_value(index, option));
        seen_a = true;
      } else if (option == "--w") {
        result.weights = parse_int_list(need_value(index, option));
        seen_w = true;
      } else if (option == "--trace") {
        result.trace_path = need_value(index, option);
      } else {
        throw std::invalid_argument("unknown option '" + option + "'");
      }
    }
    if (!seen_rows || !seen_k || !seen_lanes || !seen_depth || !seen_drain || !result.seed_set) {
      throw std::invalid_argument("rows, k, lanes, depth, row-drain and seed are required");
    }
    if (seen_a != seen_w) {
      throw std::invalid_argument("--a and --w must be supplied together");
    }
    if (seen_a) {
      result.operands_set = true;
      const ValidationResult operands = validate_operands(result.config, result.a, result.weights);
      if (!operands.ok) {
        throw std::invalid_argument(operands.error);
      }
    } else {
      const ValidationResult config = validate_config(result.config);
      if (!config.ok) {
        throw std::invalid_argument(config.error);
      }
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

DseReport run_experiment(const Config& config,
                         std::uint32_t seed,
                         const std::vector<int>* a_override,
                         const std::vector<int>* weights_override,
                         TraceSink trace_sink) {
  const ValidationResult config_result = validate_config(config);
  if (!config_result.ok) {
    throw std::invalid_argument(config_result.error);
  }
  std::vector<int> a;
  std::vector<int> weights;
  if ((a_override == nullptr) != (weights_override == nullptr)) {
    throw std::invalid_argument("A and W overrides must be supplied together");
  }
  if (a_override != nullptr) {
    a = *a_override;
    weights = *weights_override;
  } else {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> value(-8, 7);
    a.resize(static_cast<std::size_t>(config.k));
    weights.resize(static_cast<std::size_t>(config.rows * config.k));
    for (int& element : a) {
      element = value(generator);
    }
    for (int& element : weights) {
      element = value(generator);
    }
  }
  const ValidationResult operands = validate_operands(config, a, weights);
  if (!operands.ok) {
    throw std::invalid_argument(operands.error);
  }

  std::vector<std::int8_t> a8(a.begin(), a.end());
  std::vector<std::int8_t> weights8(weights.begin(), weights.end());
  const std::vector<std::int16_t> functional_result = golden(config, a8, weights8);

  DseCore core(config);
  core.set_operands(a8, weights8);
  core.start();
  int cycles = 0;
  int issue_events = 0;
  int commit_events = 0;
  const int groups = (config.rows + config.lanes - 1) / config.lanes;
  const int expected_cycles = config.row_drain
                                  ? groups * (config.k + config.depth - 1)
                                  : groups * config.k + config.depth - 1;
  while (core.running()) {
    const Tick tick = core.tick();
    ++cycles;
    for (const LaneEvent& event : tick.events) {
      issue_events += event.issue ? 1 : 0;
      commit_events += event.commit ? 1 : 0;
      if (trace_sink) {
        trace_sink(event);
      }
    }
    if (cycles > expected_cycles) {
      throw std::runtime_error("DSE core did not reach DONE at expected latency");
    }
  }
  if (cycles != expected_cycles) {
    throw std::runtime_error("DSE core cycle count mismatch");
  }
  const std::vector<std::int16_t>& cycle_result = core.results();
  if (cycle_result != functional_result) {
    throw std::runtime_error("DSE cycle result differs from independent golden result");
  }
  if (issue_events != config.rows * config.k || commit_events != config.rows * config.k) {
    throw std::runtime_error("DSE issue/commit event count mismatch");
  }

  DseReport report;
  report.config = config;
  report.macs = config.rows * config.k;
  report.cycles = cycles;
  report.issue_events = issue_events;
  report.commit_events = commit_events;
  report.a = a;
  report.weights = weights;
  report.functional_result = functional_result;
  report.cycle_result = cycle_result;
  return report;
}

}  // namespace tinylpu_dse
