#pragma once

#include <map>
#include <string>
#include <vector>

namespace tinylpu_nonlinear {

struct Config {
  std::string op = "softmax";
  int rows = 2;
  int cols = 8;
  double epsilon = 1e-5;
  int alu_latency = 1;
  int exp_latency = 8;
  int div_latency = 16;
  int sqrt_latency = 16;
  int erf_latency = 16;
  int tanh_latency = 8;
};

struct Event {
  int node = 0;
  std::string resource;
  std::string primitive;
  int issue = 0;
  int commit = 0;
  int a = -1;
  int b = -1;
  double value = 0;
};

struct Report {
  Config config;
  std::vector<double> x, gamma, beta, golden, cycle_result;
  int cycles = 0;
  int alu_issues = 0;
  int sfu_busy_cycles = 0;
  std::map<std::string, int> primitive_counts;
  std::vector<Event> events;
};

// Binary64 numerical reference; no hardware precision or SFU approximation is
// implied. Rows normalize over the last (cols) dimension.
std::vector<double> reference(const Config& config, const std::vector<double>& x,
                              const std::vector<double>& gamma,
                              const std::vector<double>& beta);
Report run(const Config& config, const std::vector<double>& x,
           const std::vector<double>& gamma, const std::vector<double>& beta);
std::string report_json(const Report& report);

}  // namespace tinylpu_nonlinear
