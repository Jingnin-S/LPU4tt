#include "tinylpu_operators.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using tinylpu_operators::Config;
using tinylpu_operators::OperatorReport;
using tinylpu_operators::TraceEvent;

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

int oracle_requant(int value) {
  const int numerator = value * 3;
  const int magnitude = numerator >= 0 ? numerator : -numerator;
  const int rounded_magnitude = (magnitude + 2) / 4;
  const int rounded = numerator >= 0 ? rounded_magnitude : -rounded_magnitude;
  return std::max(-8, std::min(7, rounded));
}

// This oracle deliberately does not call the operator library's golden or
// lowering functions. It is kept as straightforward nested loops so tests
// catch address-generation and scheduling errors independently.
std::vector<int> oracle(const Config& config,
                        const std::vector<int>& a,
                        const std::vector<int>& b) {
  const std::string& op = config.op;
  const std::vector<int>& s = config.shape;
  if (op == "dot") {
    int sum = 0;
    for (int k = 0; k < s[0]; ++k) sum += a[k] * b[k];
    return {sum};
  }
  if (op == "gemv") {
    std::vector<int> out(s[0], 0);
    for (int m = 0; m < s[0]; ++m)
      for (int k = 0; k < s[1]; ++k) out[m] += a[k] * b[m * s[1] + k];
    return out;
  }
  if (op == "gemm") {
    std::vector<int> out(s[0] * s[2], 0);
    for (int m = 0; m < s[0]; ++m)
      for (int n = 0; n < s[2]; ++n)
        for (int k = 0; k < s[1]; ++k) out[m * s[2] + n] += a[m * s[1] + k] * b[k * s[2] + n];
    return out;
  }
  if (op == "conv1d") {
    const int out_l = s[0] - s[3] + 1;
    std::vector<int> out(out_l * s[2], 0);
    for (int t = 0; t < out_l; ++t)
      for (int co = 0; co < s[2]; ++co)
        for (int kw = 0; kw < s[3]; ++kw)
          for (int ci = 0; ci < s[1]; ++ci)
            out[t * s[2] + co] += a[(t + kw) * s[1] + ci] * b[(kw * s[1] + ci) * s[2] + co];
    return out;
  }
  if (op == "conv2d") {
    const int oh_count = s[0] - s[4] + 1;
    const int ow_count = s[1] - s[5] + 1;
    std::vector<int> out(oh_count * ow_count * s[3], 0);
    for (int oh = 0; oh < oh_count; ++oh)
      for (int ow = 0; ow < ow_count; ++ow)
        for (int co = 0; co < s[3]; ++co)
          for (int kh = 0; kh < s[4]; ++kh)
            for (int kw = 0; kw < s[5]; ++kw)
              for (int ci = 0; ci < s[2]; ++ci)
                out[(oh * ow_count + ow) * s[3] + co] +=
                    a[((oh + kh) * s[1] + ow + kw) * s[2] + ci] *
                    b[((kh * s[5] + kw) * s[2] + ci) * s[3] + co];
    return out;
  }
  if (op == "dwconv2d") {
    const int oh_count = s[0] - s[3] + 1;
    const int ow_count = s[1] - s[4] + 1;
    std::vector<int> out(oh_count * ow_count * s[2], 0);
    for (int oh = 0; oh < oh_count; ++oh)
      for (int ow = 0; ow < ow_count; ++ow)
        for (int c = 0; c < s[2]; ++c)
          for (int kh = 0; kh < s[3]; ++kh)
            for (int kw = 0; kw < s[4]; ++kw)
              out[(oh * ow_count + ow) * s[2] + c] +=
                  a[((oh + kh) * s[1] + ow + kw) * s[2] + c] * b[(kh * s[4] + kw) * s[2] + c];
    return out;
  }
  if (op == "fir") {
    std::vector<int> out(s[0] - s[1] + 1, 0);
    for (int t = 0; t < s[0] - s[1] + 1; ++t)
      for (int k = 0; k < s[1]; ++k) out[t] += b[k] * a[t + s[1] - 1 - k];
    return out;
  }
  if (op == "add" || op == "mul") {
    std::vector<int> out(s[0], 0);
    for (int i = 0; i < s[0]; ++i) out[i] = op == "add" ? a[i] + b[i] : a[i] * b[i];
    return out;
  }
  if (op == "relu") {
    std::vector<int> out(s[0], 0);
    for (int i = 0; i < s[0]; ++i) out[i] = std::max(0, a[i]);
    return out;
  }
  if (op == "requant") {
    std::vector<int> out(s[0], 0);
    for (int i = 0; i < s[0]; ++i) out[i] = oracle_requant(a[i]);
    return out;
  }
  if (op == "reduce_sum") {
    int sum = 0;
    for (int value : a) sum += value;
    return {sum};
  }
  if (op == "reduce_max") {
    int value = a[0];
    for (std::size_t i = 1; i < a.size(); ++i) value = std::max(value, a[i]);
    return {value};
  }
  if (op == "maxpool2d") {
    const int oh_count = (s[0] - s[3]) / s[5] + 1;
    const int ow_count = (s[1] - s[4]) / s[6] + 1;
    std::vector<int> out(oh_count * ow_count * s[2], -8);
    for (int oh = 0; oh < oh_count; ++oh)
      for (int ow = 0; ow < ow_count; ++ow)
        for (int c = 0; c < s[2]; ++c)
          for (int kh = 0; kh < s[3]; ++kh)
            for (int kw = 0; kw < s[4]; ++kw)
              out[(oh * ow_count + ow) * s[2] + c] = std::max(
                  out[(oh * ow_count + ow) * s[2] + c],
                  a[((oh * s[5] + kh) * s[1] + ow * s[6] + kw) * s[2] + c]);
    return out;
  }
  if (op == "transpose") {
    std::vector<int> out(s[0] * s[1], 0);
    for (int row = 0; row < s[0]; ++row)
      for (int col = 0; col < s[1]; ++col) out[col * s[0] + row] = a[row * s[1] + col];
    return out;
  }
  throw std::runtime_error("oracle missing operator " + op);
}

std::vector<int> make_values(const Config& config, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> int4(-8, 7);
  std::uniform_int_distribution<int> int16(-256, 256);
  std::vector<int> values(static_cast<std::size_t>(tinylpu_operators::input_element_count(config)));
  for (int& value : values) value = config.op == "requant" ? int16(generator) : int4(generator);
  return values;
}

std::vector<int> make_b_values(const Config& config, std::uint32_t seed) {
  std::mt19937 generator(seed + 17);
  std::uniform_int_distribution<int> int4(-8, 7);
  std::vector<int> values(static_cast<std::size_t>(tinylpu_operators::second_operand_element_count(config)));
  for (int& value : values) value = int4(generator);
  return values;
}

void check_trace(const Config& config, const std::vector<TraceEvent>& events) {
  std::map<std::tuple<int, int, int>, int> issues;
  int done_count = 0;
  for (const TraceEvent& event : events) {
    expect(event.a_reads <= config.a_ports, "A read port budget exceeded");
    expect(event.b_reads <= config.b_ports, "B read port budget exceeded");
    if (event.issue) issues[{event.lane, event.issue_output, event.issue_term}] = event.relative;
    if (event.commit) {
      const auto key = std::make_tuple(event.lane, event.commit_output, event.commit_term);
      const auto found = issues.find(key);
      expect(found != issues.end(), "commit has no matching issue");
      expect(event.relative - found->second == config.depth - 1, "commit latency mismatch");
    }
    if (event.done) {
      ++done_count;
      expect(event.commit && event.store, "DONE must be a stored commit");
    }
  }
  expect(done_count == 1, "expected one DONE event");
}

OperatorReport check_case(const Config& config,
                          const std::vector<int>& a,
                          const std::vector<int>& b) {
  std::vector<TraceEvent> events;
  const OperatorReport report = tinylpu_operators::run_experiment(
      config, 1, &a, &b, [&events](const TraceEvent& event) { events.push_back(event); });
  expect(report.golden == oracle(config, a, b), config.op + " independent oracle mismatch");
  expect(report.cycle_result == report.golden, config.op + " cycle result mismatch");
  expect(report.cycles > 0, config.op + " has no cycles");
  expect(report.model_kind == "integer_bit_exact", config.op + " model kind mismatch");
  check_trace(config, events);
  return report;
}

void test_all_operators() {
  check_case({"dot", {4}, 2, 3, 1, 1}, {1, -2, 3, -4}, {2, 2, 2, 2});
  check_case({"gemv", {2, 3}, 2, 2, 2, 1}, {1, -2, 3}, {1, 2, 3, -1, 0, 2});
  check_case({"gemm", {2, 2, 2}, 2, 3, 2, 2}, {1, 2, -3, 4}, {1, -1, 2, 3});
  check_case({"conv1d", {4, 1, 2, 3}, 2, 2, 1, 1}, {1, 2, -1, 3}, {2, -1, 1, 2, -2, 1});
  check_case({"conv2d", {3, 3, 1, 2, 2, 2}, 2, 2, 2, 1},
             {1, 2, 3, 4, 5, 6, 7, 1, -2}, {1, -1, 2, 0, -2, 1, 1, 2});
  check_case({"dwconv2d", {3, 3, 2, 2, 2}, 4, 2, 1, 1},
             {1, -2, 3, 4, -5, 6, 7, 1, -2, 3, 4, -5, 6, 7, 1, -2, 3, 4},
             {1, -1, 2, 0, -2, 1, 1, 2});
  check_case({"fir", {5, 3}, 2, 3, 1, 1}, {1, 2, 3, 4, 5}, {1, -1, 2});
  check_case({"add", {5}, 4, 2, 2, 1}, {1, -2, 3, -4, 7}, {-1, 2, -3, 4, -7});
  check_case({"mul", {5}, 4, 2, 2, 2}, {1, -2, 3, -4, 7}, {-1, 2, -3, 4, -7});
  check_case({"relu", {5}, 4, 2, 1, 1}, {-8, -1, 0, 3, 7}, {});
  check_case({"requant", {5}, 2, 3, 1, 1}, {2, -2, 256, -256, 1}, {});
  check_case({"reduce_sum", {6}, 2, 3, 1, 1}, {-8, 7, -8, 7, 1, -1}, {});
  check_case({"reduce_max", {6}, 2, 3, 1, 1}, {-8, 7, -8, 3, 1, -1}, {});
  check_case({"maxpool2d", {4, 4, 1, 2, 2, 2, 2}, 4, 2, 1, 1},
             {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, {});
  const OperatorReport transpose = check_case({"transpose", {2, 3}, 4, 2, 1, 1},
                                               {1, 2, 3, 4, 5, 6}, {});
  expect(transpose.macs == 0 && transpose.scalar_ops == 0, "transpose operation counts incorrect");
}

void test_random_and_boundaries() {
  const std::vector<Config> configs = {
      {"dot", {7}, 1, 2, 1, 1},
      {"gemv", {3, 4}, 4, 4, 2, 2},
      {"gemm", {2, 3, 3}, 2, 4, 2, 2},
      {"conv1d", {5, 2, 2, 2}, 4, 3, 2, 2},
      {"conv2d", {4, 4, 2, 2, 2, 2}, 4, 3, 2, 2},
      {"dwconv2d", {4, 4, 2, 2, 2}, 4, 3, 2, 2},
      {"fir", {5, 4}, 2, 4, 2, 2},
      {"add", {9}, 4, 2, 1, 2},
      {"mul", {9}, 2, 3, 2, 1},
      {"relu", {9}, 4, 2, 1, 1},
      {"requant", {9}, 4, 3, 1, 1},
      {"reduce_sum", {16}, 4, 4, 1, 1},
      {"reduce_max", {16}, 4, 4, 1, 1},
      {"maxpool2d", {5, 5, 2, 3, 2, 1, 2}, 4, 2, 2, 1},
      {"transpose", {3, 4}, 2, 4, 1, 1},
  };
  for (std::uint32_t seed = 0; seed < 5; ++seed) {
    for (const Config& config : configs) {
      check_case(config, make_values(config, seed), make_b_values(config, seed));
    }
  }

  Config extreme{"dot", {64}, 4, 4, 4, 4};
  std::vector<int> a(64, -8);
  std::vector<int> b(64, 7);
  const OperatorReport extreme_report = check_case(extreme, a, b);
  expect(extreme_report.golden[0] == -3584, "INT4 extreme dot mismatch");

  Config fir{"fir", {5, 4}, 1, 2, 1, 1};
  const OperatorReport fir_report = check_case(fir, {1, 2, 3, 4, 5}, {1, 2, 3, 4});
  expect(fir_report.golden == std::vector<int>({20, 30}), "FIR reversal mismatch");

  expect(!tinylpu_operators::validate_config({"fir", {2, 4}, 1, 2, 1, 1}).ok,
         "FIR K>L was accepted");

  Config quant{"requant", {9}, 4, 3, 1, 1};
  const OperatorReport quant_report = check_case(
      quant, {-256, -3, -2, -1, 0, 1, 2, 3, 256}, {});
  expect(quant_report.golden == std::vector<int>({-8, -2, -2, -1, 0, 1, 2, 2, 7}),
         "requant ties/clamps mismatch");
}

void test_read_ports_and_validation() {
  const Config p1{"dwconv2d", {4, 4, 1, 2, 2}, 4, 2, 1, 1};
  const Config p4{"dwconv2d", {4, 4, 1, 2, 2}, 4, 2, 4, 4};
  const std::vector<int> a(16, 1);
  const std::vector<int> b(4, 1);
  const OperatorReport one = check_case(p1, a, b);
  const OperatorReport four = check_case(p4, a, b);
  expect(one.cycles > four.cycles, "DWConv A-port bottleneck did not reduce throughput");
  expect(one.peak_a_reads == 1 && four.peak_a_reads <= 4, "DWConv A read peak mismatch");

  expect(!tinylpu_operators::validate_config({"dot", {4, 2}, 1, 2, 1, 1}).ok,
         "bad dot shape accepted");
  expect(!tinylpu_operators::validate_config({"conv2d", {2, 2, 1, 1, 3, 3}, 1, 2, 1, 1}).ok,
         "oversized VALID window accepted");
  expect(!tinylpu_operators::validate_config({"add", {4}, 1, 2, 0, 1}).ok,
         "zero A ports accepted");
  expect(!tinylpu_operators::validate_operands({"requant", {2}, 1, 2, 1, 1}, {257, 0}, {}).ok,
         "out-of-range requant input accepted");
}

void test_cli_parser() {
  const char* args[] = {"tinylpu_operators", "--op", "gemv", "--shape", "2,3", "--lanes", "2",
                        "--depth", "3", "--a-ports", "1", "--b-ports", "2", "--seed", "1"};
  auto parsed = tinylpu_operators::parse_cli(static_cast<int>(std::size(args)), const_cast<char**>(args));
  expect(parsed.ok && parsed.config.op == "gemv" && parsed.seed == 1, "valid operator CLI parse failed");
  const char* bad[] = {"tinylpu_operators", "--op", "dot", "--shape", "4", "--lanes", "3",
                       "--depth", "2", "--a-ports", "1", "--b-ports", "1", "--seed", "1"};
  parsed = tinylpu_operators::parse_cli(static_cast<int>(std::size(bad)), const_cast<char**>(bad));
  expect(!parsed.ok, "invalid lane count accepted");
}

}  // namespace

int main() {
  try {
    test_all_operators();
    test_random_and_boundaries();
    test_read_ports_and_validation();
    test_cli_parser();
    std::cout << "test_operators: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_operators: FAIL: " << error.what() << '\n';
    return 1;
  }
}
