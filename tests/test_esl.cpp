#include "tinylpu_esl.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tinylpu::CommandKind;
using tinylpu::HostCommand;
using tinylpu::HostModel;
using tinylpu::HostResponse;
using tinylpu::Kernel;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_accept(const HostResponse& response, const std::string& message) {
  expect(response.accepted && !response.error, message + " should be accepted");
}

void expect_reject(const HostResponse& response, const std::string& message) {
  expect(response.rejected && response.error, message + " should be rejected with error");
}

void load_host(HostModel& host,
               const std::vector<int>& a,
               const std::vector<int>& weights) {
  expect_accept(host.tick(true, true,
                          {CommandKind::WriteA, 0,
                           tinylpu::pack_int4(a[0], a[1])}),
                "WriteA[0]");
  expect_accept(host.tick(true, true,
                          {CommandKind::WriteA, 1,
                           tinylpu::pack_int4(a[2], a[3])}),
                "WriteA[1]");
  for (std::size_t address = 0; address < weights.size() / 2; ++address) {
    expect_accept(host.tick(true, true,
                            {CommandKind::WriteW,
                             static_cast<std::uint8_t>(address),
                             tinylpu::pack_int4(weights[address * 2], weights[address * 2 + 1])}),
                  "WriteW");
  }
}

std::vector<std::int16_t> read_host(HostModel& host, int rows) {
  std::vector<std::int16_t> result(static_cast<std::size_t>(rows), 0);
  for (int row = 0; row < rows; ++row) {
    std::uint16_t bits = 0;
    for (int byte = 0; byte < 2; ++byte) {
      const HostResponse response = host.tick(
          true, true,
          {CommandKind::ReadResult, static_cast<std::uint8_t>(row * 2 + byte), 0});
      expect_accept(response, "ReadResult");
      bits |= static_cast<std::uint16_t>(response.data) << (8 * byte);
    }
    result[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(bits);
  }
  return result;
}

void test_all_int4_products_and_packing() {
  for (int a = -8; a <= 7; ++a) {
    for (int w = -8; w <= 7; ++w) {
      expect(tinylpu::FunctionalModel::product(a, w) == a * w,
             "INT4 product mismatch");
      const auto values = tinylpu::unpack_int4(tinylpu::pack_int4(a, w));
      expect(values[0] == a && values[1] == w, "INT4 nibble packing mismatch");
    }
  }
  expect(tinylpu::decode_int4(0x0f) == -1, "negative nibble decode");
  expect(tinylpu::decode_int4(0x08) == -8, "minimum nibble decode");
}

void test_functional_random_and_boundaries() {
  const std::vector<int> boundary_a = {-8, -8, 7, 7};
  const std::vector<int> boundary_w = {-8, 7, -8, 7};
  tinylpu::Vec4 a{};
  for (int index = 0; index < 4; ++index) {
    a[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(boundary_a[index]);
  }
  std::vector<std::int8_t> weights(boundary_w.begin(), boundary_w.end());
  expect(tinylpu::FunctionalModel::evaluate(Kernel::Dot4, a, weights)[0] == 1,
         "boundary arithmetic result");
  expect(tinylpu::FunctionalModel::product(-8, -8) == 64,
         "INT4 minimum product must remain positive 64");

  std::mt19937 generator(0x5eed);
  std::uniform_int_distribution<int> value(-8, 7);
  for (Kernel kernel : {Kernel::Dot4, Kernel::Gemv2x4}) {
    const int rows = tinylpu::kernel_rows(kernel);
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<int> a_values(4);
      std::vector<int> w_values(static_cast<std::size_t>(rows * 4));
      for (int& element : a_values) {
        element = value(generator);
      }
      for (int& element : w_values) {
        element = value(generator);
      }
      tinylpu::Vec4 a_value{};
      for (int index = 0; index < 4; ++index) {
        a_value[static_cast<std::size_t>(index)] = static_cast<std::int8_t>(a_values[index]);
      }
      std::vector<std::int8_t> w_value(w_values.begin(), w_values.end());
      const auto actual = tinylpu::FunctionalModel::evaluate(kernel, a_value, w_value);
      std::vector<std::int16_t> oracle(static_cast<std::size_t>(rows), 0);
      for (int row = 0; row < rows; ++row) {
        int sum = 0;
        for (int index = 0; index < 4; ++index) {
          sum += a_values[index] * w_values[static_cast<std::size_t>(row * 4 + index)];
        }
        oracle[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(sum);
      }
      expect(actual == oracle, "random functional result mismatch");
    }
  }
}

void test_random_cycle_against_independent_oracle() {
  std::mt19937 generator(0xC0DE);
  std::uniform_int_distribution<int> value(-8, 7);
  for (Kernel kernel : {Kernel::Dot4, Kernel::Gemv2x4}) {
    const int rows = tinylpu::kernel_rows(kernel);
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<int> a(4);
      std::vector<int> w(static_cast<std::size_t>(rows * 4));
      for (int& element : a) {
        element = value(generator);
      }
      for (int& element : w) {
        element = value(generator);
      }

      // Independent arithmetic oracle: no FunctionalModel or CycleModel call.
      std::vector<std::int16_t> oracle(static_cast<std::size_t>(rows), 0);
      for (int row = 0; row < rows; ++row) {
        int sum = 0;
        for (int index = 0; index < 4; ++index) {
          sum += a[index] * w[static_cast<std::size_t>(row * 4 + index)];
        }
        oracle[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(sum);
      }

      HostModel host(kernel);
      std::vector<tinylpu::TraceRow> trace;
      host.set_trace_sink([&trace](const tinylpu::TraceRow& row) { trace.push_back(row); });
      load_host(host, a, w);
      expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "random START");
      int relative = 0;
      while (!host.done()) {
        ++relative;
        host.tick(true, true, {});
      }
      expect(relative == 5 * rows, "random RUN latency");

      // Check issue and commit slots independently, including overlap cycles.
      int compute_relative = 0;
      for (const auto& event : trace) {
        if (event.phase != "compute") {
          continue;
        }
        ++compute_relative;
        const int expected_row = (compute_relative - 1) / 5;
        const int slot = (compute_relative - 1) % 5;
        const bool expected_issue = slot < 4;
        const bool expected_commit = slot >= 1;
        expect(event.issue == expected_issue, "issue valid timing mismatch");
        expect(event.commit == expected_commit, "commit valid timing mismatch");
        if (expected_issue) {
          expect(event.row == expected_row && event.index == slot,
                 "issue row/index timing mismatch");
        }
        if (expected_commit) {
          expect(event.commit_row == expected_row && event.commit_index == slot - 1,
                 "commit row/index timing mismatch");
        }
        expect(event.done == (compute_relative == 5 * rows), "DONE timing mismatch");
        if (event.done) {
          expect(event.busy_before && !event.busy, "final commit busy transition");
        }
      }
      expect(read_host(host, rows) == oracle, "random cycle result mismatch");
    }
  }

  // The two dot4 mathematical extrema exercise signed accumulation bounds.
  tinylpu::Vec4 min_a = {7, 7, 7, 7};
  std::vector<std::int8_t> min_w = {-8, -8, -8, -8};
  expect(tinylpu::FunctionalModel::evaluate(Kernel::Dot4, min_a, min_w)[0] == -224,
         "dot4 minimum result");
  tinylpu::Vec4 max_a = {-8, -8, -8, -8};
  std::vector<std::int8_t> max_w = {-8, -8, -8, -8};
  expect(tinylpu::FunctionalModel::evaluate(Kernel::Dot4, max_a, max_w)[0] == 256,
         "dot4 maximum result");
}

void test_fixed_cycle_timing() {
  for (Kernel kernel : {Kernel::Dot4, Kernel::Gemv2x4}) {
    const int rows = tinylpu::kernel_rows(kernel);
    const std::vector<int> a = {1, -2, 3, -4};
    const std::vector<int> w = rows == 1
                                   ? std::vector<int>{1, 1, 1, 1}
                                   : std::vector<int>{1, 1, 1, 1, -1, -1, -1, -1};
    HostModel host(kernel);
    std::vector<tinylpu::TraceRow> trace;
    host.set_trace_sink([&trace](const tinylpu::TraceRow& row) { trace.push_back(row); });
    load_host(host, a, w);
    expect_reject(host.tick(true, true, {CommandKind::ReadResult, 0, 0}),
                  "read before START/DONE");
    expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "START");
    int compute_ticks = 0;
    while (!host.done()) {
      host.tick(true, true, {});
      ++compute_ticks;
      expect(compute_ticks <= 5 * rows, "DONE exceeded fixed RUN length");
    }
    expect(compute_ticks == 5 * rows, "RUN must be exactly 5*M cycles");
    expect(static_cast<int>(std::count_if(trace.begin(), trace.end(),
                                          [](const tinylpu::TraceRow& row) {
                                            return row.issue;
                                          })) == 4 * rows,
           "issue count mismatch");
    expect(static_cast<int>(std::count_if(trace.begin(), trace.end(),
                                          [](const tinylpu::TraceRow& row) {
                                            return row.commit;
                                          })) == 4 * rows,
           "commit count mismatch");
    expect(static_cast<int>(std::count_if(trace.begin(), trace.end(),
                                          [](const tinylpu::TraceRow& row) {
                                            return row.done;
                                          })) == 1,
           "DONE must be a single final edge");
    const auto result = read_host(host, rows);
    tinylpu::Vec4 a_value = {1, -2, 3, -4};
    std::vector<std::int8_t> w_value(w.begin(), w.end());
    expect(result == tinylpu::FunctionalModel::evaluate(kernel, a_value, w_value),
           "cycle result mismatch");
  }
}

void test_host_protocol_and_recovery() {
  HostModel host(Kernel::Dot4);
  std::vector<tinylpu::TraceRow> control_trace;
  host.set_trace_sink([&control_trace](const tinylpu::TraceRow& row) {
    control_trace.push_back(row);
  });
  expect_reject(host.tick(true, true, {CommandKind::Start, 0, 0}),
                "START with no loaded data");
  expect_reject(host.tick(true, true, {CommandKind::WriteA, 2, 0}),
                "invalid A address");
  expect_reject(host.tick(true, true, {CommandKind::WriteW, 2, 0}),
                "invalid W address");

  const std::vector<int> a = {1, -2, 3, -4};
  const std::vector<int> w = {1, 1, 1, 1};
  load_host(host, a, w);
  expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "START after load");
  expect_reject(host.tick(true, true, {CommandKind::WriteA, 0, 0}),
                "write while busy");
  expect(host.busy(), "busy must continue after rejected command");
  expect_reject(host.tick(true, true, {CommandKind::ReadResult, 0, 0}),
                "read while busy");
  // The two rejected commands above still advance the running pipeline. The
  // final commit edge is therefore absolute RUN tick five, not loop iteration
  // five.
  int final_run_tick = 2;
  while (!host.done()) {
    ++final_run_tick;
    const HostResponse response = host.tick(
        true, true,
        final_run_tick == 5
            ? HostCommand{CommandKind::WriteA, 0, tinylpu::pack_int4(0, 0)}
            : HostCommand{});
    if (final_run_tick == 5) {
      expect_reject(response, "command on final commit edge");
    }
  }
  const auto result = read_host(host, 1);
  expect(result.size() == 1 && result[0] == -2, "little-endian signed result");
  expect(host.done() && !host.busy(), "DONE must persist after RUN");

  // Reset wins over a command and clears the load-valid state.
  load_host(host, a, w);
  expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "second START");
  host.tick(true, true, {});
  control_trace.clear();
  host.tick(false, true, {CommandKind::ReadResult, 0, 0});
  expect(control_trace.size() == 1 && control_trace.back().phase == "reset" &&
             control_trace.back().busy_before && !control_trace.back().busy &&
             !control_trace.back().done,
         "reset trace must show busy-to-idle transition");
  expect(!host.busy() && !host.done() && !host.a_valid() && !host.weights_valid(),
         "reset must cancel and clear all valid state");
  expect_reject(host.tick(true, true, {CommandKind::Start, 0, 0}),
                "START after reset without reload");

  load_host(host, a, w);
  expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "third START");
  control_trace.clear();
  host.tick(true, false, {});
  expect(control_trace.size() == 1 && control_trace.back().phase == "disabled" &&
             control_trace.back().busy_before && !control_trace.back().busy &&
             !control_trace.back().done,
         "ena trace must show busy-to-idle transition");
  expect(!host.busy() && !host.done() && !host.a_valid() && !host.weights_valid(),
         "ena low must clear state");
  expect_reject(host.tick(true, true, {CommandKind::Start, 0, 0}),
                "START after ena re-enable without reload");

  // Weights persist across successful transactions while A may be replaced.
  load_host(host, a, w);
  expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}), "reuse START");
  while (!host.done()) {
    host.tick(true, true, {});
  }
  (void)read_host(host, 1);
  const std::vector<int> a2 = {1, 1, 1, 1};
  expect_accept(host.tick(true, true,
                          {CommandKind::WriteA, 0, tinylpu::pack_int4(a2[0], a2[1])}),
                "updated A low byte");
  expect_accept(host.tick(true, true,
                          {CommandKind::WriteA, 1, tinylpu::pack_int4(a2[2], a2[3])}),
                "updated A high byte");
  expect_accept(host.tick(true, true, {CommandKind::Start, 0, 0}),
                "START with persistent W");
  while (!host.done()) {
    host.tick(true, true, {});
  }
  expect(read_host(host, 1)[0] == 4, "persistent W with updated A");
}

void test_experiment_report_and_trace() {
  const std::string trace_path = "tinylpu_esl_test_trace.csv";
  tinylpu::ExperimentConfig config;
  config.kernel = Kernel::Gemv2x4;
  config.a = {1, -2, 3, -4};
  config.weights = {1, 1, 1, 1, -1, -1, -1, -1};
  config.reuse = 3;
  config.host_wait = 2;
  config.trace_path = trace_path;
  const auto report = tinylpu::run_experiment(config);
  expect(report.functional_result == std::vector<std::int16_t>{-2, 2},
         "GEMV report result");
  expect(report.cycle_result == report.functional_result, "report cycle result");
  expect(report.macs == 24, "MAC count");
  expect(report.weight_load_cycles == 4 && report.input_load_cycles == 6,
         "load cycle accounting");
  expect(report.setup_cycles == 3 && report.compute_cycles == 30,
         "setup/compute accounting");
  expect(report.readout_cycles == 12 && report.host_wait_cycles == 6,
         "readout/wait accounting");
  expect(report.total_cycles == 4 + 3 * (2 + 2 + 1 + 10 + 4),
         "total cycle accounting");
  const std::string json = tinylpu::report_json(report);
  expect(json.find("\"core_macs_per_cycle\":0.8") != std::string::npos,
         "core throughput must be numeric JSON");

  std::ifstream trace(trace_path);
  expect(static_cast<bool>(trace), "trace file missing");
  std::string line;
  expect(std::getline(trace, line) && line.find("cycle,phase,action") == 0,
         "trace header missing");
  int lines = 0;
  bool saw_issue = false;
  bool saw_commit = false;
  bool saw_done = false;
  while (std::getline(trace, line)) {
    ++lines;
    saw_issue = saw_issue || line.find(",1,0,") != std::string::npos;
    saw_commit = saw_commit || line.find(",0,1,") != std::string::npos;
    saw_done = saw_done || (line.size() >= 2 && line.compare(line.size() - 2, 2, ",1") == 0);
  }
  expect(lines > 0 && saw_issue && saw_commit && saw_done,
         "trace must show issue, commit and DONE");
  std::remove(trace_path.c_str());
}

void test_validation_and_cli_parser() {
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0}, {0, 0, 0, 0}, 1, 0).ok,
         "short A must fail validation");
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0, 0}, {8, 0, 0, 0}, 1, 0).ok,
         "out-of-range W must fail validation");
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, 0).ok,
         "zero reuse must fail validation");
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0, 0}, {0, 0, 0, 0}, 1, -1).ok,
         "negative host wait must fail validation");
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   std::numeric_limits<int>::max(), 0).ok,
         "huge reuse must fail bounded ESL validation");
  expect(!tinylpu::validate_inputs(Kernel::Dot4, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   1, std::numeric_limits<int>::max()).ok,
         "huge host wait must fail bounded ESL validation");

  const char* good_args[] = {"tinylpu_esl", "--kernel", "dot4", "--a", "1,2,3,4",
                             "--w", "1,2,3,4", "--reuse", "2", "--host-wait", "3"};
  auto good = tinylpu::parse_cli(static_cast<int>(std::size(good_args)),
                                 const_cast<char**>(good_args));
  expect(good.ok && !good.help && good.config.reuse == 2 && good.config.host_wait == 3,
         "valid CLI parse");
  const char* bad_args[] = {"tinylpu_esl", "--kernel", "dot4", "--a", "0,0,0,0",
                            "--w", "0,0,0,0", "--reuse", "0", "--host-wait", "0"};
  auto bad = tinylpu::parse_cli(static_cast<int>(std::size(bad_args)),
                                const_cast<char**>(bad_args));
  expect(!bad.ok && !bad.error.empty(), "invalid CLI range parse");
}

}  // namespace

int main() {
  try {
    test_all_int4_products_and_packing();
    test_functional_random_and_boundaries();
    test_random_cycle_against_independent_oracle();
    test_fixed_cycle_timing();
    test_host_protocol_and_recovery();
    test_experiment_report_and_trace();
    test_validation_and_cli_parser();
    std::cout << "test_esl: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_esl: FAIL: " << error.what() << '\n';
    return 1;
  }
}
