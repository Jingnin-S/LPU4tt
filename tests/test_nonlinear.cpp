#include "tinylpu_nonlinear.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>

void check(bool condition, const char* why) { if (!condition) throw std::runtime_error(why); }

void check_events(const tinylpu_nonlinear::Report& r) {
  std::map<int, int> committed;
  std::vector<tinylpu_nonlinear::Event> by_issue = r.events;
  std::sort(by_issue.begin(), by_issue.end(), [](const auto& a, const auto& b) { return a.issue < b.issue; });
  int sfu_free = 0, alu_last = -1;
  for (const auto& e : r.events) committed[e.node] = e.commit;
  for (const auto& e : by_issue) {
    check(e.commit > e.issue, "positive latency");
    for (int dep : {e.a, e.b}) if (committed.count(dep)) check(e.issue >= committed[dep], "dependency used too early");
    if (e.resource == "sfu") { check(e.issue >= sfu_free, "SFU overlapped"); sfu_free = e.commit; }
    else { check(e.issue != alu_last, "multiple ALU issues in one cycle"); alu_last = e.issue; }
  }
}

int main() {
  try {
    using namespace tinylpu_nonlinear;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-4, 4);
    for (const auto& op : {"softmax", "layernorm", "rmsnorm", "gelu", "silu", "sigmoid", "tanh"}) {
      for (int cols : {1, 3, 8, 16}) {
        Config c; c.op = op; c.rows = 2; c.cols = cols;
        std::vector<double> x(2 * cols), gamma(cols, 1), beta(cols, 0);
        for (double& v : x) v = dist(rng);
        const auto r = run(c, x, gamma, beta);
        check(r.golden.size() == x.size() && r.cycle_result.size() == x.size(), "output shape");
        check_events(r);
        if (c.op == "softmax") for (int row = 0; row < 2; ++row) {
          double sum = 0;
          for (int j = 0; j < cols; ++j) { check(r.cycle_result[row * cols + j] > 0, "softmax positive"); sum += r.cycle_result[row * cols + j]; }
          check(std::abs(sum - 1) < 1e-12, "softmax sum");
        }
        if (c.op == "layernorm") for (int row = 0; row < 2; ++row) {
          double sum = 0; for (int j = 0; j < cols; ++j) sum += r.cycle_result[row * cols + j];
          check(std::abs(sum) < 1e-10, "centered norm");
        }
        Config slow = c;
        slow.exp_latency *= 2; slow.div_latency *= 2; slow.sqrt_latency *= 2; slow.erf_latency *= 2; slow.tanh_latency *= 2;
        const auto slower = run(slow, x, gamma, beta);
        check(slower.cycles >= r.cycles, "SFU sensitivity");
        check(slower.cycle_result == r.cycle_result, "latency must not alter arithmetic");
      }
    }
    Config c; c.op = "softmax"; c.rows = 1; c.cols = 3;
    const auto stable = run(c, {10000, 10001, 10002}, {1,1,1}, {0,0,0});
    check(std::abs(stable.golden[2] - 0.6652409557748218) < 1e-12, "stable large-input softmax");
    c.op = "layernorm";
    auto flat = run(c, {3,3,3}, {2,3,4}, {1,2,3});
    check(flat.cycle_result == std::vector<double>({1,2,3}), "constant norm returns beta");
    const auto offset = run(c, {999999.9,999999.9,999999.9}, {1,1,1}, {0,0,0});
    for (double v : offset.cycle_result) check(std::abs(v) < 1e-6, "large-offset near-constant norm");
    c.op = "sigmoid";
    auto extreme = run(c, {-10000,0,10000}, {1,1,1}, {0,0,0});
    check(extreme.cycle_result[0] == 0 && extreme.cycle_result[1] == 0.5 && extreme.cycle_result[2] == 1, "stable sigmoid extremes");
    bool rejected = false;
    try { c.epsilon = 0; (void)run(c, {0,0,0}, {1,1,1}, {0,0,0}); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid epsilon");
    rejected = false;
    try { c.epsilon = 1e-5; (void)run(c, {0,std::numeric_limits<double>::infinity(),0}, {1,1,1}, {0,0,0}); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "nonfinite input");
    std::cout << "test_nonlinear: PASS\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
