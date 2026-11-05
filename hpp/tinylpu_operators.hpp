#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tinylpu_operators {

// The operator model uses the explicit tensor layouts documented by the
// operator-model change plan. A/B are flat row-major tensors; FIR uses
// shape[L,K] with A length L, B length K, and VALID output length L-K+1.
struct Config {
  std::string op;
  std::vector<int> shape;
  int lanes = 1;
  int depth = 2;
  int a_ports = 1;
  int b_ports = 1;
};

struct ValidationResult {
  bool ok = false;
  std::string error;
};

struct WorkTerm {
  int output = -1;
  int term = -1;
  int total_terms = 0;
  bool has_a = false;
  bool has_b = false;
  int a_addr = -1;
  int b_addr = -1;
  int value = 0;
  bool mac = false;
  bool scalar_op = false;
  bool max_combine = false;
  std::string kind;
};

struct TraceEvent {
  int relative = 0;
  int lane = -1;
  bool issue = false;
  int issue_output = -1;
  int issue_term = -1;
  int issue_a_addr = -1;
  int issue_b_addr = -1;
  int issue_value = 0;
  bool commit = false;
  int commit_output = -1;
  int commit_term = -1;
  int commit_value = 0;
  int acc = 0;
  bool store = false;
  bool done = false;
  int a_reads = 0;
  int b_reads = 0;
};

struct OperatorReport {
  Config config;
  std::vector<int> output_shape;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> golden;
  std::vector<int> cycle_result;
  int cycles = 0;
  int macs = 0;
  int scalar_ops = 0;
  int input_bytes = 0;
  int weight_bytes = 0;
  int output_bytes = 0;
  int a_read_transactions = 0;
  int b_read_transactions = 0;
  int peak_a_reads = 0;
  int peak_b_reads = 0;
  std::string model_kind = "integer_bit_exact";
  std::vector<std::string> required_units;
};

ValidationResult validate_config(const Config& config);
ValidationResult validate_operands(const Config& config,
                                   const std::vector<int>& a,
                                   const std::vector<int>& b);

std::vector<int> output_shape(const Config& config);
int input_element_count(const Config& config);
int second_operand_element_count(const Config& config);
int output_element_count(const Config& config);

// Independent functional oracle. The cycle model lowers source tensors into
// WorkTerm objects and never calls this function to obtain its result.
std::vector<int> golden(const Config& config,
                        const std::vector<int>& a,
                        const std::vector<int>& b);

std::vector<std::vector<WorkTerm>> lower_terms(const Config& config,
                                               const std::vector<int>& a,
                                               const std::vector<int>& b);

std::string report_json(const OperatorReport& report);

struct CliParseResult {
  bool ok = false;
  bool help = false;
  Config config;
  std::uint32_t seed = 0;
  bool seed_set = false;
  std::vector<int> a;
  std::vector<int> b;
  bool a_set = false;
  bool b_set = false;
  std::string trace_path;
  std::string error;
};

CliParseResult parse_cli(int argc, char** argv);
std::string usage();

using TraceSink = std::function<void(const TraceEvent&)>;

OperatorReport run_experiment(const Config& config,
                              std::uint32_t seed,
                              const std::vector<int>* a_override = nullptr,
                              const std::vector<int>* b_override = nullptr,
                              TraceSink trace_sink = {});

}  // namespace tinylpu_operators
