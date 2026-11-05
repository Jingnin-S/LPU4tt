#include "tinylpu_operators.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinylpu_operators {
namespace {

enum class Op {
  Dot,
  Gemv,
  Gemm,
  Conv1d,
  Conv2d,
  Dwconv2d,
  Fir,
  Add,
  Mul,
  Relu,
  Requant,
  ReduceSum,
  ReduceMax,
  Maxpool2d,
  Transpose,
};

Op decode_op(const std::string& name) {
  if (name == "dot") return Op::Dot;
  if (name == "gemv") return Op::Gemv;
  if (name == "gemm") return Op::Gemm;
  if (name == "conv1d") return Op::Conv1d;
  if (name == "conv2d") return Op::Conv2d;
  if (name == "dwconv2d") return Op::Dwconv2d;
  if (name == "fir") return Op::Fir;
  if (name == "add") return Op::Add;
  if (name == "mul") return Op::Mul;
  if (name == "relu") return Op::Relu;
  if (name == "requant") return Op::Requant;
  if (name == "reduce_sum") return Op::ReduceSum;
  if (name == "reduce_max") return Op::ReduceMax;
  if (name == "maxpool2d") return Op::Maxpool2d;
  if (name == "transpose") return Op::Transpose;
  throw std::invalid_argument("unknown operator '" + name + "'");
}

struct ShapeInfo {
  std::vector<int> output_shape;
  int a_count = 0;
  int b_count = 0;
  int output_count = 0;
  int terms_per_output = 0;
};

int checked_product(std::initializer_list<int> values) {
  long long product = 1;
  for (const int value : values) {
    product *= value;
    if (product > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("tensor size is too large");
    }
  }
  return static_cast<int>(product);
}

int shape_arity(Op op) {
  switch (op) {
    case Op::Dot:
      return 1;
    case Op::Gemv:
      return 2;
    case Op::Gemm:
      return 3;
    case Op::Conv1d:
      return 4;
    case Op::Conv2d:
      return 6;
    case Op::Dwconv2d:
      return 5;
    case Op::Fir:
      return 2;
    case Op::Add:
    case Op::Mul:
    case Op::Relu:
    case Op::Requant:
    case Op::ReduceSum:
    case Op::ReduceMax:
      return 1;
    case Op::Maxpool2d:
      return 7;
    case Op::Transpose:
      return 2;
  }
  return 0;
}

ShapeInfo shape_info(const Config& config, Op op) {
  const std::vector<int>& s = config.shape;
  ShapeInfo info;
  switch (op) {
    case Op::Dot:
      info.output_shape = {1};
      info.a_count = s[0];
      info.b_count = s[0];
      info.output_count = 1;
      info.terms_per_output = s[0];
      break;
    case Op::Gemv:
      info.output_shape = {s[0]};
      info.a_count = s[1];
      info.b_count = checked_product({s[0], s[1]});
      info.output_count = s[0];
      info.terms_per_output = s[1];
      break;
    case Op::Gemm:
      info.output_shape = {s[0], s[2]};
      info.a_count = checked_product({s[0], s[1]});
      info.b_count = checked_product({s[1], s[2]});
      info.output_count = checked_product({s[0], s[2]});
      info.terms_per_output = s[1];
      break;
    case Op::Conv1d: {
      const int out_l = s[0] - s[3] + 1;
      info.output_shape = {out_l, s[2]};
      info.a_count = checked_product({s[0], s[1]});
      info.b_count = checked_product({s[3], s[1], s[2]});
      info.output_count = checked_product({out_l, s[2]});
      info.terms_per_output = checked_product({s[1], s[3]});
      break;
    }
    case Op::Conv2d: {
      const int out_h = s[0] - s[4] + 1;
      const int out_w = s[1] - s[5] + 1;
      info.output_shape = {out_h, out_w, s[3]};
      info.a_count = checked_product({s[0], s[1], s[2]});
      info.b_count = checked_product({s[4], s[5], s[2], s[3]});
      info.output_count = checked_product({out_h, out_w, s[3]});
      info.terms_per_output = checked_product({s[2], s[4], s[5]});
      break;
    }
    case Op::Dwconv2d: {
      const int out_h = s[0] - s[3] + 1;
      const int out_w = s[1] - s[4] + 1;
      info.output_shape = {out_h, out_w, s[2]};
      info.a_count = checked_product({s[0], s[1], s[2]});
      info.b_count = checked_product({s[3], s[4], s[2]});
      info.output_count = checked_product({out_h, out_w, s[2]});
      // Depthwise output channels are independent; each output has only its
      // Kh*Kw spatial terms, not C copies of the same channel's window.
      info.terms_per_output = checked_product({s[3], s[4]});
      break;
    }
    case Op::Fir:
      info.output_shape = {s[0] - s[1] + 1};
      info.a_count = s[0];
      info.b_count = s[1];
      info.output_count = s[0] - s[1] + 1;
      info.terms_per_output = s[1];
      break;
    case Op::Add:
    case Op::Mul:
      info.output_shape = {s[0]};
      info.a_count = s[0];
      info.b_count = s[0];
      info.output_count = s[0];
      info.terms_per_output = 1;
      break;
    case Op::Relu:
    case Op::Requant:
    case Op::ReduceSum:
    case Op::ReduceMax:
      info.output_shape = (op == Op::ReduceSum || op == Op::ReduceMax)
                              ? std::vector<int>{1}
                              : std::vector<int>{s[0]};
      info.a_count = s[0];
      info.b_count = 0;
      info.output_count = (op == Op::ReduceSum || op == Op::ReduceMax) ? 1 : s[0];
      info.terms_per_output = (op == Op::ReduceSum || op == Op::ReduceMax) ? s[0] : 1;
      break;
    case Op::Maxpool2d: {
      const int out_h = (s[0] - s[3]) / s[5] + 1;
      const int out_w = (s[1] - s[4]) / s[6] + 1;
      info.output_shape = {out_h, out_w, s[2]};
      info.a_count = checked_product({s[0], s[1], s[2]});
      info.b_count = 0;
      info.output_count = checked_product({out_h, out_w, s[2]});
      info.terms_per_output = checked_product({s[3], s[4]});
      break;
    }
    case Op::Transpose:
      info.output_shape = {s[1], s[0]};
      info.a_count = checked_product({s[0], s[1]});
      info.b_count = 0;
      info.output_count = info.a_count;
      info.terms_per_output = 1;
      break;
  }
  return info;
}

bool is_requant(Op op) { return op == Op::Requant; }

bool output_is_int16(Op op) {
  return op == Op::Dot || op == Op::Gemv || op == Op::Gemm || op == Op::Conv1d ||
         op == Op::Conv2d || op == Op::Dwconv2d || op == Op::Fir || op == Op::Add ||
         op == Op::Mul || op == Op::ReduceSum;
}

int requantize(int value) {
  const int numerator = value * 3;
  const int magnitude = numerator >= 0 ? numerator : -numerator;
  const int rounded_magnitude = (magnitude + 2) / 4;
  const int rounded = numerator >= 0 ? rounded_magnitude : -rounded_magnitude;
  return std::max(-8, std::min(7, rounded));
}

std::string trim(std::string value) {
  const auto first = std::find_if(value.begin(), value.end(), [](unsigned char c) {
    return !std::isspace(c);
  });
  const auto last = std::find_if(value.rbegin(), value.rend(), [](unsigned char c) {
    return !std::isspace(c);
  }).base();
  if (first >= last) return {};
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
    if (token.empty()) throw std::invalid_argument("empty integer in comma-separated list");
    std::size_t consumed = 0;
    int value = 0;
    try {
      value = std::stoi(token, &consumed, 10);
    } catch (const std::exception&) {
      throw std::invalid_argument("invalid integer '" + token + "'");
    }
    if (consumed != token.size()) throw std::invalid_argument("invalid integer '" + token + "'");
    values.push_back(value);
    if (end == std::string::npos) break;
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
  if (consumed != text.size()) throw std::invalid_argument(std::string(name) + " must be an integer");
  return value;
}

void append_int_array(std::ostringstream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
}

void append_string_array(std::ostringstream& out, const std::vector<std::string>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << '"' << values[i] << '"';
  }
  out << ']';
}

struct CycleTick {
  int relative = 0;
  std::vector<TraceEvent> events;
  bool done = false;
};

class CycleCore {
 public:
  CycleCore(const Config& config, const std::vector<std::vector<WorkTerm>>& work)
      : config_(config), work_(work), pipeline_(static_cast<std::size_t>(config.depth),
                                                 std::vector<Pending>(static_cast<std::size_t>(config.lanes))),
        lane_output_(static_cast<std::size_t>(config.lanes)),
        lane_term_(static_cast<std::size_t>(config.lanes), 0),
        lane_acc_(static_cast<std::size_t>(config.lanes), 0),
        results_(work.size(), 0) {
    for (int lane = 0; lane < config_.lanes; ++lane) {
      lane_output_[static_cast<std::size_t>(lane)] = lane < static_cast<int>(work_.size()) ? lane : -1;
    }
  }

  bool running() const { return running_; }
  const std::vector<int>& results() const { return results_; }

  CycleTick tick() {
    CycleTick result;
    if (!running_) {
      result.relative = relative_;
      result.done = done_;
      return result;
    }

    result.relative = ++relative_;
    std::vector<TraceEvent> events(static_cast<std::size_t>(config_.lanes));
    std::vector<bool> active(static_cast<std::size_t>(config_.lanes), false);
    for (int lane = 0; lane < config_.lanes; ++lane) {
      events[static_cast<std::size_t>(lane)].relative = relative_;
      events[static_cast<std::size_t>(lane)].lane = lane;
    }

    const int due_slot = relative_ % config_.depth;
    for (int lane = 0; lane < config_.lanes; ++lane) {
      Pending& pending = pipeline_[static_cast<std::size_t>(due_slot)][static_cast<std::size_t>(lane)];
      if (!pending.valid) continue;
      TraceEvent& event = events[static_cast<std::size_t>(lane)];
      active[static_cast<std::size_t>(lane)] = true;
      event.commit = true;
      event.commit_output = pending.term.output;
      event.commit_term = pending.term.term;
      event.commit_value = pending.term.value;
      const int previous = lane_acc_[static_cast<std::size_t>(lane)];
      const int next = pending.term.term == 0
                           ? pending.term.value
                           : (pending.term.max_combine ? std::max(previous, pending.term.value)
                                                        : previous + pending.term.value);
      lane_acc_[static_cast<std::size_t>(lane)] = static_cast<std::int16_t>(next);
      event.acc = lane_acc_[static_cast<std::size_t>(lane)];
      if (pending.term.term + 1 == pending.term.total_terms) {
        results_[static_cast<std::size_t>(pending.term.output)] = event.acc;
        event.store = true;
        ++completed_outputs_;
        if (completed_outputs_ == static_cast<int>(work_.size())) {
          event.done = true;
          result.done = true;
        }
      }
      pending = {};
    }

    std::vector<int> a_addresses;
    std::vector<int> b_addresses;
    if (!result.done) {
      for (int lane = 0; lane < config_.lanes; ++lane) {
        const int output = lane_output_[static_cast<std::size_t>(lane)];
        if (output < 0) continue;
        const int term_index = lane_term_[static_cast<std::size_t>(lane)];
        const WorkTerm& term = work_[static_cast<std::size_t>(output)][static_cast<std::size_t>(term_index)];
        const bool new_a = term.has_a &&
                           std::find(a_addresses.begin(), a_addresses.end(), term.a_addr) == a_addresses.end();
        const bool new_b = term.has_b &&
                           std::find(b_addresses.begin(), b_addresses.end(), term.b_addr) == b_addresses.end();
        if ((new_a && static_cast<int>(a_addresses.size()) >= config_.a_ports) ||
            (new_b && static_cast<int>(b_addresses.size()) >= config_.b_ports)) {
          continue;
        }
        if (new_a) a_addresses.push_back(term.a_addr);
        if (new_b) b_addresses.push_back(term.b_addr);
        TraceEvent& event = events[static_cast<std::size_t>(lane)];
        active[static_cast<std::size_t>(lane)] = true;
        event.issue = true;
        event.issue_output = term.output;
        event.issue_term = term.term;
        event.issue_a_addr = term.has_a ? term.a_addr : -1;
        event.issue_b_addr = term.has_b ? term.b_addr : -1;
        event.issue_value = term.value;
        Pending& destination = pipeline_[static_cast<std::size_t>(
            (relative_ + config_.depth - 1) % config_.depth)][static_cast<std::size_t>(lane)];
        destination.valid = true;
        destination.term = term;
        ++lane_term_[static_cast<std::size_t>(lane)];
        if (lane_term_[static_cast<std::size_t>(lane)] == term.total_terms) {
          lane_term_[static_cast<std::size_t>(lane)] = 0;
          lane_output_[static_cast<std::size_t>(lane)] += config_.lanes;
          if (lane_output_[static_cast<std::size_t>(lane)] >= static_cast<int>(work_.size())) {
            lane_output_[static_cast<std::size_t>(lane)] = -1;
          }
        }
      }
    }

    for (int lane = 0; lane < config_.lanes; ++lane) {
      if (!active[static_cast<std::size_t>(lane)]) continue;
      TraceEvent& event = events[static_cast<std::size_t>(lane)];
      if (!event.commit) event.acc = lane_acc_[static_cast<std::size_t>(lane)];
      event.a_reads = static_cast<int>(a_addresses.size());
      event.b_reads = static_cast<int>(b_addresses.size());
      result.events.push_back(event);
    }
    if (result.done) {
      running_ = false;
      done_ = true;
    }
    return result;
  }

 private:
  struct Pending {
    bool valid = false;
    WorkTerm term;
  };

  Config config_;
  const std::vector<std::vector<WorkTerm>>& work_;
  std::vector<std::vector<Pending>> pipeline_;
  std::vector<int> lane_output_;
  std::vector<int> lane_term_;
  std::vector<std::int16_t> lane_acc_;
  std::vector<int> results_;
  int completed_outputs_ = 0;
  int relative_ = 0;
  bool running_ = true;
  bool done_ = false;
};

std::vector<std::string> required_units(Op op, const ShapeInfo& info) {
  std::vector<std::string> units;
  if (info.a_count > 0) units.emplace_back("A_read_ports");
  if (info.b_count > 0) units.emplace_back("B_read_ports");
  switch (op) {
    case Op::Dot:
    case Op::Gemv:
    case Op::Gemm:
    case Op::Conv1d:
    case Op::Conv2d:
    case Op::Dwconv2d:
    case Op::Fir:
      units.emplace_back("INT4_multiplier");
      units.emplace_back("ACC16_adder");
      break;
    case Op::Add:
      units.emplace_back("INT16_adder");
      break;
    case Op::Mul:
      units.emplace_back("INT4_multiplier");
      break;
    case Op::Relu:
      units.emplace_back("INT4_relu");
      break;
    case Op::Requant:
      units.emplace_back("fused_mul_round_clip");
      break;
    case Op::ReduceSum:
      units.emplace_back("ACC16_adder");
      break;
    case Op::ReduceMax:
    case Op::Maxpool2d:
      units.emplace_back("INT4_comparator");
      break;
    case Op::Transpose:
      units.emplace_back("address_permutation");
      break;
  }
  if (info.a_count > 0) units.emplace_back("explicit_A_read_bandwidth");
  if (info.b_count > 0) units.emplace_back("explicit_B_read_bandwidth");
  if (op == Op::Add || op == Op::Mul) units.emplace_back("B_second_operand_not_stationary_weight");
  return units;
}

}  // namespace

ValidationResult validate_config(const Config& config) {
  Op op;
  try {
    op = decode_op(config.op);
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
  if (static_cast<int>(config.shape.size()) != shape_arity(op)) {
    return {false, "shape has the wrong number of dimensions for " + config.op};
  }
  for (int dimension : config.shape) {
    if (dimension < 1 || dimension > 64) return {false, "shape dimensions must be in [1,64]"};
  }
  if (config.lanes != 1 && config.lanes != 2 && config.lanes != 4) {
    return {false, "lanes must be one of 1,2,4"};
  }
  if (config.depth < 2 || config.depth > 4) return {false, "depth must be in [2,4]"};
  if (config.a_ports < 1 || config.a_ports > 4) return {false, "a-ports must be in [1,4]"};
  if (config.b_ports < 1 || config.b_ports > 4) return {false, "b-ports must be in [1,4]"};

  const std::vector<int>& s = config.shape;
  if ((op == Op::Conv1d && s[0] < s[3]) ||
      (op == Op::Conv2d && (s[0] < s[4] || s[1] < s[5])) ||
      (op == Op::Dwconv2d && (s[0] < s[3] || s[1] < s[4])) ||
      (op == Op::Maxpool2d && (s[0] < s[3] || s[1] < s[4]))) {
    return {false, "VALID window must fit the input dimensions"};
  }
  if (op == Op::Fir && s[1] > s[0]) return {false, "FIR K must be <= input length L"};
  if (op == Op::Maxpool2d &&
      ((s[0] - s[3]) / s[5] + 1 <= 0 || (s[1] - s[4]) / s[6] + 1 <= 0)) {
    return {false, "maxpool stride/window produces an empty output"};
  }
  try {
    const ShapeInfo info = shape_info(config, op);
    if (info.terms_per_output > 64) return {false, "terms per output must be <=64"};
    if (info.output_count > 4096 || info.a_count > 8192 || info.b_count > 8192) {
      return {false, "tensor is too large for the bounded integer model"};
    }
    if (info.output_count * info.terms_per_output > 16384) {
      return {false, "total lowered terms must be <=16384"};
    }
  } catch (const std::exception& error) {
    return {false, error.what()};
  }
  return {true, {}};
}

ValidationResult validate_operands(const Config& config,
                                   const std::vector<int>& a,
                                   const std::vector<int>& b) {
  const ValidationResult config_result = validate_config(config);
  if (!config_result.ok) return config_result;
  const Op op = decode_op(config.op);
  const ShapeInfo info = shape_info(config, op);
  if (static_cast<int>(a.size()) != info.a_count) return {false, "A has the wrong element count"};
  if (static_cast<int>(b.size()) != info.b_count) return {false, "B has the wrong element count"};
  const int minimum = is_requant(op) ? -256 : -8;
  const int maximum = is_requant(op) ? 256 : 7;
  for (int value : a) {
    if (value < minimum || value > maximum) return {false, "A value is outside the operand range"};
  }
  for (int value : b) {
    if (value < -8 || value > 7) return {false, "B value is outside signed INT4 [-8,7]"};
  }
  return {true, {}};
}

std::vector<int> output_shape(const Config& config) {
  const ValidationResult result = validate_config(config);
  if (!result.ok) throw std::invalid_argument(result.error);
  return shape_info(config, decode_op(config.op)).output_shape;
}

int input_element_count(const Config& config) {
  const ValidationResult result = validate_config(config);
  if (!result.ok) throw std::invalid_argument(result.error);
  return shape_info(config, decode_op(config.op)).a_count;
}

int second_operand_element_count(const Config& config) {
  const ValidationResult result = validate_config(config);
  if (!result.ok) throw std::invalid_argument(result.error);
  return shape_info(config, decode_op(config.op)).b_count;
}

int output_element_count(const Config& config) {
  const ValidationResult result = validate_config(config);
  if (!result.ok) throw std::invalid_argument(result.error);
  return shape_info(config, decode_op(config.op)).output_count;
}

std::vector<int> golden(const Config& config,
                        const std::vector<int>& a,
                        const std::vector<int>& b) {
  const ValidationResult operands = validate_operands(config, a, b);
  if (!operands.ok) throw std::invalid_argument(operands.error);
  const Op op = decode_op(config.op);
  const std::vector<int>& s = config.shape;
  const ShapeInfo info = shape_info(config, op);
  std::vector<int> result(static_cast<std::size_t>(info.output_count), 0);
  switch (op) {
    case Op::Dot:
      for (int k = 0; k < s[0]; ++k) result[0] += a[k] * b[k];
      break;
    case Op::Gemv:
      for (int m = 0; m < s[0]; ++m)
        for (int k = 0; k < s[1]; ++k) result[m] += a[k] * b[m * s[1] + k];
      break;
    case Op::Gemm:
      for (int m = 0; m < s[0]; ++m)
        for (int n = 0; n < s[2]; ++n)
          for (int k = 0; k < s[1]; ++k) result[m * s[2] + n] +=
              a[m * s[1] + k] * b[k * s[2] + n];
      break;
    case Op::Conv1d: {
      const int out_l = s[0] - s[3] + 1;
      for (int t = 0; t < out_l; ++t)
        for (int co = 0; co < s[2]; ++co)
          for (int kw = 0; kw < s[3]; ++kw)
            for (int ci = 0; ci < s[1]; ++ci)
              result[t * s[2] + co] += a[(t + kw) * s[1] + ci] *
                                        b[(kw * s[1] + ci) * s[2] + co];
      break;
    }
    case Op::Conv2d: {
      const int out_h = s[0] - s[4] + 1;
      const int out_w = s[1] - s[5] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int co = 0; co < s[3]; ++co)
            for (int kh = 0; kh < s[4]; ++kh)
              for (int kw = 0; kw < s[5]; ++kw)
                for (int ci = 0; ci < s[2]; ++ci)
                  result[(oh * out_w + ow) * s[3] + co] +=
                      a[((oh + kh) * s[1] + ow + kw) * s[2] + ci] *
                      b[(((kh * s[5] + kw) * s[2] + ci) * s[3]) + co];
      break;
    }
    case Op::Dwconv2d: {
      const int out_h = s[0] - s[3] + 1;
      const int out_w = s[1] - s[4] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int c = 0; c < s[2]; ++c)
            for (int kh = 0; kh < s[3]; ++kh)
              for (int kw = 0; kw < s[4]; ++kw)
                result[(oh * out_w + ow) * s[2] + c] +=
                    a[((oh + kh) * s[1] + ow + kw) * s[2] + c] *
                    b[(kh * s[4] + kw) * s[2] + c];
      break;
    }
    case Op::Fir:
      for (int t = 0; t < s[0] - s[1] + 1; ++t)
        for (int k = 0; k < s[1]; ++k) result[t] += a[t + s[1] - 1 - k] * b[k];
      break;
    case Op::Add:
      for (int i = 0; i < s[0]; ++i) result[i] = static_cast<std::int16_t>(a[i] + b[i]);
      break;
    case Op::Mul:
      for (int i = 0; i < s[0]; ++i) result[i] = static_cast<std::int16_t>(a[i] * b[i]);
      break;
    case Op::Relu:
      for (int i = 0; i < s[0]; ++i) result[i] = std::max(0, a[i]);
      break;
    case Op::Requant:
      for (int i = 0; i < s[0]; ++i) result[i] = requantize(a[i]);
      break;
    case Op::ReduceSum:
      for (int i = 0; i < s[0]; ++i) result[0] += a[i];
      break;
    case Op::ReduceMax:
      result[0] = a[0];
      for (int i = 1; i < s[0]; ++i) result[0] = std::max(result[0], a[i]);
      break;
    case Op::Maxpool2d: {
      const int out_h = (s[0] - s[3]) / s[5] + 1;
      const int out_w = (s[1] - s[4]) / s[6] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int c = 0; c < s[2]; ++c) {
            int value = -8;
            for (int kh = 0; kh < s[3]; ++kh)
              for (int kw = 0; kw < s[4]; ++kw)
                value = std::max(value, a[((oh * s[5] + kh) * s[1] + ow * s[6] + kw) * s[2] + c]);
            result[(oh * out_w + ow) * s[2] + c] = value;
          }
      break;
    }
    case Op::Transpose:
      for (int row = 0; row < s[0]; ++row)
        for (int col = 0; col < s[1]; ++col) result[col * s[0] + row] = a[row * s[1] + col];
      break;
  }
  return result;
}

std::vector<std::vector<WorkTerm>> lower_terms(const Config& config,
                                               const std::vector<int>& a,
                                               const std::vector<int>& b) {
  const ValidationResult operands = validate_operands(config, a, b);
  if (!operands.ok) throw std::invalid_argument(operands.error);
  const Op op = decode_op(config.op);
  const std::vector<int>& s = config.shape;
  const ShapeInfo info = shape_info(config, op);
  std::vector<std::vector<WorkTerm>> work(static_cast<std::size_t>(info.output_count));
  auto add = [&](int out, int term, int aa, int bb, bool has_a, bool has_b, int value,
                 const std::string& kind, bool mac, bool scalar, bool max_combine) {
    WorkTerm item;
    item.output = out;
    item.term = term;
    item.total_terms = info.terms_per_output;
    item.has_a = has_a;
    item.has_b = has_b;
    item.a_addr = has_a ? aa : -1;
    item.b_addr = has_b ? bb : -1;
    item.value = value;
    item.kind = kind;
    item.mac = mac;
    item.scalar_op = scalar;
    item.max_combine = max_combine;
    work[static_cast<std::size_t>(out)].push_back(item);
  };
  switch (op) {
    case Op::Dot:
      for (int k = 0; k < s[0]; ++k) add(0, k, k, k, true, true, a[k] * b[k], "mac", true, false, false);
      break;
    case Op::Gemv:
      for (int m = 0; m < s[0]; ++m)
        for (int k = 0; k < s[1]; ++k)
          add(m, k, k, m * s[1] + k, true, true, a[k] * b[m * s[1] + k], "mac", true, false, false);
      break;
    case Op::Gemm:
      for (int m = 0; m < s[0]; ++m)
        for (int n = 0; n < s[2]; ++n)
          for (int k = 0; k < s[1]; ++k) {
            const int out = m * s[2] + n;
            add(out, k, m * s[1] + k, k * s[2] + n, true, true,
                a[m * s[1] + k] * b[k * s[2] + n], "mac", true, false, false);
          }
      break;
    case Op::Conv1d: {
      const int out_l = s[0] - s[3] + 1;
      int term = 0;
      for (int t = 0; t < out_l; ++t)
        for (int co = 0; co < s[2]; ++co) {
          term = 0;
          for (int kw = 0; kw < s[3]; ++kw)
            for (int ci = 0; ci < s[1]; ++ci, ++term) {
              const int aa = (t + kw) * s[1] + ci;
              const int bb = (kw * s[1] + ci) * s[2] + co;
              add(t * s[2] + co, term, aa, bb, true, true, a[aa] * b[bb], "mac", true, false, false);
            }
        }
      break;
    }
    case Op::Conv2d: {
      const int out_h = s[0] - s[4] + 1;
      const int out_w = s[1] - s[5] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int co = 0; co < s[3]; ++co) {
            int term = 0;
            for (int kh = 0; kh < s[4]; ++kh)
              for (int kw = 0; kw < s[5]; ++kw)
                for (int ci = 0; ci < s[2]; ++ci, ++term) {
                  const int aa = ((oh + kh) * s[1] + ow + kw) * s[2] + ci;
                  const int bb = (((kh * s[5] + kw) * s[2] + ci) * s[3]) + co;
                  add((oh * out_w + ow) * s[3] + co, term, aa, bb, true, true,
                      a[aa] * b[bb], "mac", true, false, false);
                }
          }
      break;
    }
    case Op::Dwconv2d: {
      const int out_h = s[0] - s[3] + 1;
      const int out_w = s[1] - s[4] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int c = 0; c < s[2]; ++c) {
            int term = 0;
            for (int kh = 0; kh < s[3]; ++kh)
              for (int kw = 0; kw < s[4]; ++kw, ++term) {
                const int aa = ((oh + kh) * s[1] + ow + kw) * s[2] + c;
                const int bb = (kh * s[4] + kw) * s[2] + c;
                add((oh * out_w + ow) * s[2] + c, term, aa, bb, true, true,
                    a[aa] * b[bb], "mac", true, false, false);
              }
          }
      break;
    }
    case Op::Fir:
      for (int t = 0; t < s[0] - s[1] + 1; ++t)
        for (int k = 0; k < s[1]; ++k) {
          const int aa = t + s[1] - 1 - k;
          add(t, k, aa, k, true, true, a[aa] * b[k], "mac", true, false, false);
        }
      break;
    case Op::Add:
      for (int i = 0; i < s[0]; ++i) add(i, 0, i, i, true, true, a[i] + b[i], "add", false, true, false);
      break;
    case Op::Mul:
      for (int i = 0; i < s[0]; ++i) add(i, 0, i, i, true, true, a[i] * b[i], "mul", false, true, false);
      break;
    case Op::Relu:
      for (int i = 0; i < s[0]; ++i) add(i, 0, i, -1, true, false, std::max(0, a[i]), "relu", false, true, false);
      break;
    case Op::Requant:
      for (int i = 0; i < s[0]; ++i) add(i, 0, i, -1, true, false, requantize(a[i]), "requant", false, true, false);
      break;
    case Op::ReduceSum:
      for (int i = 0; i < s[0]; ++i) add(0, i, i, -1, true, false, a[i], "reduce_sum", false, i > 0, false);
      break;
    case Op::ReduceMax:
      for (int i = 0; i < s[0]; ++i) add(0, i, i, -1, true, false, a[i], "reduce_max", false, i > 0, i > 0);
      break;
    case Op::Maxpool2d: {
      const int out_h = (s[0] - s[3]) / s[5] + 1;
      const int out_w = (s[1] - s[4]) / s[6] + 1;
      for (int oh = 0; oh < out_h; ++oh)
        for (int ow = 0; ow < out_w; ++ow)
          for (int c = 0; c < s[2]; ++c) {
            int term = 0;
            for (int kh = 0; kh < s[3]; ++kh)
              for (int kw = 0; kw < s[4]; ++kw, ++term) {
                const int aa = ((oh * s[5] + kh) * s[1] + ow * s[6] + kw) * s[2] + c;
                add((oh * out_w + ow) * s[2] + c, term, aa, -1, true, false, a[aa],
                    "maxpool", false, term > 0, term > 0);
              }
          }
      break;
    }
    case Op::Transpose:
      for (int row = 0; row < s[1]; ++row)
        for (int col = 0; col < s[0]; ++col) {
          const int aa = col * s[1] + row;
          add(row * s[0] + col, 0, aa, -1, true, false, a[aa], "transpose", false, false, false);
        }
      break;
  }
  return work;
}

std::string report_json(const OperatorReport& report) {
  std::ostringstream out;
  out << "{\"op\":\"" << report.config.op << "\",\"shape\":";
  append_int_array(out, report.config.shape);
  out << ",\"output_shape\":";
  append_int_array(out, report.output_shape);
  out << ",\"a\":";
  append_int_array(out, report.a);
  out << ",\"b\":";
  append_int_array(out, report.b);
  out << ",\"golden\":";
  append_int_array(out, report.golden);
  out << ",\"cycle_result\":";
  append_int_array(out, report.cycle_result);
  out << ",\"lanes\":" << report.config.lanes
      << ",\"depth\":" << report.config.depth
      << ",\"a_ports\":" << report.config.a_ports
      << ",\"b_ports\":" << report.config.b_ports
      << ",\"cycles\":" << report.cycles
      << ",\"macs\":" << report.macs
      << ",\"scalar_ops\":" << report.scalar_ops
      << ",\"input_bytes\":" << report.input_bytes
      << ",\"weight_bytes\":" << report.weight_bytes
      << ",\"output_bytes\":" << report.output_bytes
      << ",\"a_read_transactions\":" << report.a_read_transactions
      << ",\"b_read_transactions\":" << report.b_read_transactions
      << ",\"peak_a_reads\":" << report.peak_a_reads
      << ",\"peak_b_reads\":" << report.peak_b_reads
      << ",\"model_kind\":\"" << report.model_kind << "\",\"required_units\":";
  append_string_array(out, report.required_units);
  out << '}';
  return out.str();
}

std::string usage() {
  return "tinylpu_operators --op NAME --shape commaDims --lanes P --depth D "
         "--a-ports A --b-ports B --seed N [--a comma-list --b comma-list] [--trace FILE]";
}

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result;
  bool seen_op = false;
  bool seen_shape = false;
  bool seen_lanes = false;
  bool seen_depth = false;
  bool seen_a_ports = false;
  bool seen_b_ports = false;
  try {
    auto need_value = [&](int& index, const std::string& option) -> std::string {
      if (index + 1 >= argc) throw std::invalid_argument("missing value for " + option);
      return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--help" || option == "-h") {
        result.help = true;
        result.ok = true;
        return result;
      } else if (option == "--op") {
        result.config.op = need_value(index, option);
        seen_op = true;
      } else if (option == "--shape") {
        result.config.shape = parse_int_list(need_value(index, option));
        seen_shape = true;
      } else if (option == "--lanes") {
        result.config.lanes = parse_scalar(need_value(index, option), "lanes");
        seen_lanes = true;
      } else if (option == "--depth") {
        result.config.depth = parse_scalar(need_value(index, option), "depth");
        seen_depth = true;
      } else if (option == "--a-ports") {
        result.config.a_ports = parse_scalar(need_value(index, option), "a-ports");
        seen_a_ports = true;
      } else if (option == "--b-ports") {
        result.config.b_ports = parse_scalar(need_value(index, option), "b-ports");
        seen_b_ports = true;
      } else if (option == "--seed") {
        const std::string text = need_value(index, option);
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || parsed > 0xffffffffULL) {
          throw std::invalid_argument("seed must be an unsigned 32-bit integer");
        }
        result.seed = static_cast<std::uint32_t>(parsed);
        result.seed_set = true;
      } else if (option == "--a") {
        result.a = parse_int_list(need_value(index, option));
        result.a_set = true;
      } else if (option == "--b") {
        result.b = parse_int_list(need_value(index, option));
        result.b_set = true;
      } else if (option == "--trace") {
        result.trace_path = need_value(index, option);
      } else {
        throw std::invalid_argument("unknown option '" + option + "'");
      }
    }
    if (!seen_op || !seen_shape || !seen_lanes || !seen_depth || !seen_a_ports || !seen_b_ports ||
        !result.seed_set) {
      throw std::invalid_argument("op, shape, lanes, depth, a-ports, b-ports and seed are required");
    }
    const ValidationResult config_result = validate_config(result.config);
    if (!config_result.ok) throw std::invalid_argument(config_result.error);
    const int b_count = second_operand_element_count(result.config);
    if (b_count > 0 && result.a_set != result.b_set) {
      throw std::invalid_argument("--a and --b must be supplied together for this operator");
    }
    if (b_count == 0 && result.b_set) throw std::invalid_argument("this operator has no B operand");
    if (result.a_set) {
      const ValidationResult operands = validate_operands(result.config, result.a, result.b);
      if (!operands.ok) throw std::invalid_argument(operands.error);
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  return result;
}

OperatorReport run_experiment(const Config& config,
                              std::uint32_t seed,
                              const std::vector<int>* a_override,
                              const std::vector<int>* b_override,
                              TraceSink trace_sink) {
  const ValidationResult config_result = validate_config(config);
  if (!config_result.ok) throw std::invalid_argument(config_result.error);
  const Op op = decode_op(config.op);
  const ShapeInfo info = shape_info(config, op);
  std::vector<int> a;
  std::vector<int> b;
  if (a_override != nullptr) {
    a = *a_override;
    if (info.b_count > 0) {
      if (b_override == nullptr) throw std::invalid_argument("B override is required");
      b = *b_override;
    } else if (b_override != nullptr && !b_override->empty()) {
      throw std::invalid_argument("this operator has no B operand");
    }
  } else if (b_override != nullptr) {
    throw std::invalid_argument("A override is required when B is overridden");
  } else {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> int4(-8, 7);
    std::uniform_int_distribution<int> int16(-256, 256);
    a.resize(static_cast<std::size_t>(info.a_count));
    b.resize(static_cast<std::size_t>(info.b_count));
    for (int& value : a) value = is_requant(op) ? int16(generator) : int4(generator);
    for (int& value : b) value = int4(generator);
  }
  const ValidationResult operands = validate_operands(config, a, b);
  if (!operands.ok) throw std::invalid_argument(operands.error);

  const std::vector<int> functional = golden(config, a, b);
  const std::vector<std::vector<WorkTerm>> work = lower_terms(config, a, b);
  CycleCore core(config, work);
  int cycles = 0;
  int macs = 0;
  int scalar_ops = 0;
  int a_read_transactions = 0;
  int b_read_transactions = 0;
  int peak_a_reads = 0;
  int peak_b_reads = 0;
  while (core.running()) {
    const CycleTick tick = core.tick();
    ++cycles;
    int tick_a_reads = 0;
    int tick_b_reads = 0;
    for (const TraceEvent& event : tick.events) {
      tick_a_reads = std::max(tick_a_reads, event.a_reads);
      tick_b_reads = std::max(tick_b_reads, event.b_reads);
      if (event.issue) {
        const WorkTerm& term = work[static_cast<std::size_t>(event.issue_output)]
                                     [static_cast<std::size_t>(event.issue_term)];
        if (term.mac) ++macs;
        if (term.scalar_op) ++scalar_ops;
        if (trace_sink) trace_sink(event);
      } else if (trace_sink) {
        trace_sink(event);
      }
    }
    a_read_transactions += tick_a_reads;
    b_read_transactions += tick_b_reads;
    peak_a_reads = std::max(peak_a_reads, tick_a_reads);
    peak_b_reads = std::max(peak_b_reads, tick_b_reads);
    if (cycles > 1000000) throw std::runtime_error("operator cycle model exceeded safety limit");
  }
  if (core.results() != functional) throw std::runtime_error("cycle result differs from golden result");

  OperatorReport report;
  report.config = config;
  report.output_shape = info.output_shape;
  report.a = std::move(a);
  report.b = std::move(b);
  report.golden = functional;
  report.cycle_result = core.results();
  report.cycles = cycles;
  report.macs = macs;
  report.scalar_ops = scalar_ops;
  report.input_bytes = is_requant(op) ? info.a_count * 2 : (info.a_count + 1) / 2;
  report.weight_bytes = (info.b_count + 1) / 2;
  report.output_bytes = output_is_int16(op) ? info.output_count * 2 : (info.output_count + 1) / 2;
  report.a_read_transactions = a_read_transactions;
  report.b_read_transactions = b_read_transactions;
  report.peak_a_reads = peak_a_reads;
  report.peak_b_reads = peak_b_reads;
  report.required_units = required_units(op, info);
  return report;
}

}  // namespace tinylpu_operators
