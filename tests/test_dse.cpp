#include "tinylpu_dse.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tinylpu_dse::Config;
using tinylpu_dse::DseCore;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::int16_t> oracle(const Config& config,
                                 const std::vector<int>& a,
                                 const std::vector<int>& weights) {
  std::vector<std::int16_t> result(static_cast<std::size_t>(config.rows), 0);
  for (int row = 0; row < config.rows; ++row) {
    int sum = 0;
    for (int index = 0; index < config.k; ++index) {
      sum += a[static_cast<std::size_t>(index)] *
             weights[static_cast<std::size_t>(row * config.k + index)];
    }
    result[static_cast<std::size_t>(row)] = static_cast<std::int16_t>(sum);
  }
  return result;
}

std::vector<tinylpu_dse::Tick> run_core(const std::vector<int>& a,
                                        const std::vector<int>& weights,
                                        DseCore& core) {
  std::vector<std::int8_t> a8(a.begin(), a.end());
  std::vector<std::int8_t> w8(weights.begin(), weights.end());
  core.set_operands(a8, w8);
  core.start();
  std::vector<tinylpu_dse::Tick> ticks;
  while (core.running()) {
    ticks.push_back(core.tick());
    expect(ticks.size() < 10000, "DSE core did not terminate");
  }
  return ticks;
}

void check_schedule(const Config& config,
                    const std::vector<tinylpu_dse::Tick>& ticks) {
  const int groups = (config.rows + config.lanes - 1) / config.lanes;
  const int expected_cycles = config.row_drain
                                  ? groups * (config.k + config.depth - 1)
                                  : groups * config.k + config.depth - 1;
  expect(static_cast<int>(ticks.size()) == expected_cycles, "cycle formula mismatch");
  int issue_count = 0;
  int commit_count = 0;
  int done_count = 0;
  for (const auto& tick : ticks) {
    for (const auto& event : tick.events) {
      if (event.issue) {
        ++issue_count;
        const int group = event.issue_row / config.lanes;
        const int expected = config.row_drain
                                 ? group * (config.k + config.depth - 1) + event.issue_index + 1
                                 : group * config.k + event.issue_index + 1;
        expect(tick.relative == expected, "issue relative cycle mismatch");
      }
      if (event.commit) {
        ++commit_count;
        const int group = event.commit_row / config.lanes;
        const int expected_issue = config.row_drain
                                       ? group * (config.k + config.depth - 1) +
                                             event.commit_index + 1
                                       : group * config.k + event.commit_index + 1;
        expect(tick.relative == expected_issue + config.depth - 1,
               "commit relative cycle mismatch");
      }
      if (event.done) {
        ++done_count;
        expect(event.commit && event.store, "DONE must be the last store commit");
        expect(tick.relative == expected_cycles, "DONE relative cycle mismatch");
      }
    }
    if (tick.done) {
      expect(tick.relative == expected_cycles, "tick DONE relative cycle mismatch");
    }
  }
  expect(issue_count == config.rows * config.k, "scalar issue event count mismatch");
  expect(commit_count == config.rows * config.k, "scalar commit event count mismatch");
  expect(done_count == 1, "DONE event count mismatch");
}

void test_parameter_validation() {
  for (Config config : {Config{0, 1, 1, 2, false}, Config{1, 0, 1, 2, false},
                        Config{1, 1, 3, 2, false}, Config{1, 1, 1, 1, false},
                        Config{17, 1, 1, 2, false}, Config{1, 65, 1, 2, false}}) {
    expect(!tinylpu_dse::validate_config(config).ok, "invalid config accepted");
  }
  expect(tinylpu_dse::validate_config(Config{16, 64, 4, 4, true}).ok,
         "maximum legal config rejected");
  expect(!tinylpu_dse::validate_operands(Config{1, 4, 1, 2, false}, {0, 0, 0},
                                         {0, 0, 0, 0})
              .ok,
         "short A accepted");
  expect(!tinylpu_dse::validate_operands(Config{1, 4, 1, 2, false}, {0, 0, 0, 0},
                                         {8, 0, 0, 0})
              .ok,
         "out-of-range W accepted");
}

void test_parameter_sweep_and_tail_lanes() {
  std::mt19937 generator(0xD5E);
  std::uniform_int_distribution<int> value(-8, 7);
  for (int rows = 1; rows <= 5; ++rows) {
    for (int k = 1; k <= 8; ++k) {
      for (int lanes : {1, 2, 4}) {
        for (int depth : {2, 3, 4}) {
          for (bool drain : {false, true}) {
            Config config{rows, k, lanes, depth, drain};
            std::vector<int> a(static_cast<std::size_t>(k));
            std::vector<int> weights(static_cast<std::size_t>(rows * k));
            for (int& element : a) {
              element = value(generator);
            }
            for (int& element : weights) {
              element = value(generator);
            }
            DseCore core(config);
            const auto ticks = run_core(a, weights, core);
            check_schedule(config, ticks);
            expect(core.results() == oracle(config, a, weights),
                   "parameter sweep result mismatch");
          }
        }
      }
    }
  }
  Config tail{3, 3, 4, 3, false};
  std::vector<int> a = {1, -2, 3};
  std::vector<int> weights = {1, 1, 1, -1, 2, -2, 3, 1, -1};
  DseCore core(tail);
  const auto ticks = run_core(a, weights, core);
  check_schedule(tail, ticks);
  expect(core.results() == oracle(tail, a, weights), "masked tail lane result mismatch");
  for (const auto& tick : ticks) {
    for (const auto& event : tick.events) {
      expect(event.issue_row < 3 && event.commit_row < 3,
             "masked lane emitted a phantom row");
    }
  }
}

void test_depth_and_continuous_rows() {
  for (int depth : {2, 3, 4}) {
    Config config{1, 5, 1, depth, false};
    std::vector<int> a = {1, -2, 3, -4, 5};
    std::vector<int> weights = {1, 1, 1, 1, 1};
    DseCore core(config);
    const auto ticks = run_core(a, weights, core);
    check_schedule(config, ticks);
    expect(core.results()[0] == 3, "depth result mismatch");
    int first_commit = -1;
    for (const auto& tick : ticks) {
      for (const auto& event : tick.events) {
        if (event.commit && event.commit_index == 0) {
          first_commit = tick.relative;
        }
      }
    }
    expect(first_commit == depth, "D-1 commit latency mismatch");
  }

  // With K=1, D=4 and no drain, row one issues before row zero commits. Their
  // independent accumulators must not be cleared or mixed.
  Config overlap{2, 1, 1, 4, false};
  DseCore overlap_core(overlap);
  const auto overlap_ticks = run_core({2}, {3, 4}, overlap_core);
  check_schedule(overlap, overlap_ticks);
  expect(overlap_ticks.size() == 5, "continuous overlap cycle count");
  expect(overlap_core.results() == std::vector<std::int16_t>({6, 8}),
         "continuous row accumulator hazard");

  // K=1 makes first/last tags adjacent across groups. With two lanes, the
  // third row reuses lane zero's feedback accumulator after row zero is done.
  Config lane_reuse{3, 1, 2, 4, false};
  DseCore lane_reuse_core(lane_reuse);
  const auto lane_reuse_ticks = run_core({2}, {3, 4, -5}, lane_reuse_core);
  check_schedule(lane_reuse, lane_reuse_ticks);
  expect(lane_reuse_core.results() == std::vector<std::int16_t>({6, 8, -10}),
         "lane accumulator reuse across groups");

  Config no_drain{5, 3, 2, 4, false};
  Config drain{5, 3, 2, 4, true};
  DseCore no_drain_core(no_drain);
  DseCore drain_core(drain);
  const auto no_drain_ticks = run_core({1, 2, 3}, std::vector<int>(15, 1), no_drain_core);
  const auto drain_ticks = run_core({1, 2, 3}, std::vector<int>(15, 1), drain_core);
  check_schedule(no_drain, no_drain_ticks);
  check_schedule(drain, drain_ticks);
  expect(no_drain_ticks.size() == 12 && drain_ticks.size() == 18,
         "row drain cycle formula");
}

void test_k_extremes_and_seeded_report() {
  Config maximum{16, 64, 4, 4, false};
  std::vector<int> a(64, -8);
  std::vector<int> weights(16 * 64, -8);
  const auto report = tinylpu_dse::run_experiment(maximum, 1, &a, &weights);
  expect(report.cycles == 16 / 4 * 64 + 3, "maximum gapless cycle count");
  expect(report.issue_events == 16 * 64 && report.commit_events == 16 * 64,
         "maximum event count");
  expect(std::all_of(report.cycle_result.begin(), report.cycle_result.end(),
                     [](std::int16_t value) { return value == 4096; }),
         "maximum K signed accumulation");

  Config seeded{2, 4, 2, 3, false};
  const auto first = tinylpu_dse::run_experiment(seeded, 1);
  const auto second = tinylpu_dse::run_experiment(seeded, 1);
  expect(first.a == second.a && first.weights == second.weights &&
             first.cycle_result == second.cycle_result,
         "seeded operands are not deterministic");
  expect(first.cycle_result == first.functional_result, "seeded report result mismatch");
}

void test_cli_parser() {
  const char* good_args[] = {"tinylpu_dse_core", "--rows", "2", "--k", "4", "--lanes", "2",
                             "--depth", "3", "--row-drain", "0", "--seed", "1",
                             "--a", "1,2,3,4", "--w", "1,1,1,1,-1,-1,-1,-1"};
  auto good = tinylpu_dse::parse_cli(static_cast<int>(std::size(good_args)),
                                     const_cast<char**>(good_args));
  expect(good.ok && good.operands_set && good.seed == 1, "valid DSE CLI parse");
  const char* bad_args[] = {"tinylpu_dse_core", "--rows", "2", "--k", "4", "--lanes", "2",
                            "--depth", "3", "--row-drain", "2", "--seed", "1"};
  auto bad = tinylpu_dse::parse_cli(static_cast<int>(std::size(bad_args)),
                                    const_cast<char**>(bad_args));
  expect(!bad.ok && !bad.error.empty(), "invalid row-drain accepted");
}

}  // namespace

int main() {
  try {
    test_parameter_validation();
    test_parameter_sweep_and_tail_lanes();
    test_depth_and_continuous_rows();
    test_k_extremes_and_seeded_report();
    test_cli_parser();
    std::cout << "test_dse: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_dse: FAIL: " << error.what() << '\n';
    return 1;
  }
}
