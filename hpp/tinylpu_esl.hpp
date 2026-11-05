#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tinylpu {

constexpr int kK = 4;
using Vec4 = std::array<std::int8_t, kK>;

enum class Kernel {
  Dot4,
  Gemv2x4,
};

std::string kernel_name(Kernel kernel);
int kernel_rows(Kernel kernel);

// Values are held in int8_t after checking the signed INT4 contract.
struct ValidationResult {
  bool ok = false;
  std::string error;
};

ValidationResult validate_inputs(Kernel kernel,
                                 const std::vector<int>& a,
                                 const std::vector<int>& weights,
                                 int reuse,
                                 int host_wait);

std::int8_t decode_int4(std::uint8_t nibble);
std::uint8_t encode_int4(int value);
std::uint8_t pack_int4(int low_index, int high_index);
std::array<std::int8_t, 2> unpack_int4(std::uint8_t payload);

class FunctionalModel {
 public:
  // This is the independent reference path. The CycleModel does not call it.
  static std::int8_t product(int a, int b);
  static std::vector<std::int16_t> evaluate(Kernel kernel,
                                             const Vec4& a,
                                             const std::vector<std::int8_t>& weights);
};

struct CycleStep {
  int relative = 0;
  bool issue = false;
  int issue_row = -1;
  int issue_index = -1;
  int issue_product = 0;
  bool commit = false;
  int commit_row = -1;
  int commit_index = -1;
  int commit_product = 0;
  int acc = 0;
  bool store = false;
  bool done = false;
};

// A one-PE, two-stage multiplier/accumulator timing model. It deliberately
// computes the pipeline state from ticks and never invokes FunctionalModel.
class CycleModel {
 public:
  explicit CycleModel(int rows);

  void set_operands(const Vec4& a, const std::vector<std::int8_t>& weights);
  void reset();
  void start();
  CycleStep tick();

  bool running() const { return running_; }
  bool done() const { return done_; }
  int relative() const { return relative_; }
  const std::vector<std::int16_t>& results() const { return results_; }

 private:
  int rows_;
  Vec4 a_{};
  std::vector<std::int8_t> weights_;
  std::vector<std::int16_t> results_;
  std::int16_t acc_ = 0;
  int relative_ = 0;
  bool running_ = false;
  bool done_ = false;
  bool pending_valid_ = false;
  int pending_row_ = -1;
  int pending_index_ = -1;
  std::int8_t pending_product_ = 0;
};

enum class CommandKind {
  Noop,
  WriteA,
  WriteW,
  Start,
  ReadResult,
};

struct HostCommand {
  CommandKind kind = CommandKind::Noop;
  std::uint8_t address = 0;
  std::uint8_t payload = 0;
};

struct HostResponse {
  bool accepted = false;
  bool rejected = false;
  bool error = false;
  std::uint8_t data = 0;
  std::string message;
};

struct TraceRow {
  std::uint64_t cycle = 0;
  std::string phase;
  std::string action;
  int row = -1;
  int index = -1;
  int commit_row = -1;
  int commit_index = -1;
  bool issue = false;
  bool commit = false;
  int product = 0;
  int issue_product = 0;
  int commit_product = 0;
  int acc = 0;
  // busy is the post-tick state; busy_before preserves the protocol's
  // pre-tick command-admission state (including the final commit edge).
  bool busy_before = false;
  bool busy = false;
  bool done = false;
};

using TraceSink = std::function<void(const TraceRow&)>;

// Abstract host transaction model. It is intentionally independent of any
// final Tiny Tapeout pin mapping. A WriteA byte packs two values and addresses
// 0..1; A is the K=4 vector shared by all output rows. W is row-major and uses
// addresses 0..2*M-1. This matches the endpoint cycle contract.
class HostModel {
 public:
  explicit HostModel(Kernel kernel);

  HostResponse tick(bool rst_n = true,
                    bool ena = true,
                    HostCommand command = {});

  void set_trace_sink(TraceSink sink) { sink_ = std::move(sink); }
  void clear_trace_sink() { sink_ = {}; }

  Kernel kernel() const { return kernel_; }
  int rows() const { return rows_; }
  std::uint64_t cycle() const { return cycle_; }
  bool busy() const { return busy_; }
  bool done() const { return done_; }
  bool a_valid() const;
  bool weights_valid() const;
  const std::vector<std::int16_t>& results() const { return results_; }

 private:
  HostResponse reject(const std::string& message);
  void clear_state();
  void emit(TraceRow row);

  Kernel kernel_;
  int rows_;
  CycleModel core_;
  Vec4 a_{};
  std::vector<std::int8_t> weights_;
  std::array<bool, 2> a_valid_{};
  std::vector<bool> weight_valid_;
  std::vector<std::int16_t> results_;
  bool busy_ = false;
  bool done_ = false;
  std::uint64_t cycle_ = 0;
  TraceSink sink_;
};

struct ExperimentConfig {
  Kernel kernel = Kernel::Dot4;
  std::vector<int> a;
  std::vector<int> weights;
  int reuse = 1;
  int host_wait = 0;
  std::string trace_path;
};

struct ExperimentReport {
  Kernel kernel = Kernel::Dot4;
  int rows = 0;
  int k = kK;
  int reuse = 0;
  int host_wait = 0;
  std::vector<std::int16_t> functional_result;
  std::vector<std::int16_t> cycle_result;
  int transactions = 0;
  int macs = 0;
  int weight_load_cycles = 0;
  int input_load_cycles = 0;
  int setup_cycles = 0;
  int compute_cycles = 0;
  int readout_cycles = 0;
  int host_wait_cycles = 0;
  int total_cycles = 0;
  int peak_macs_per_cycle = 1;
  double core_macs_per_cycle = 0.8;
};

ExperimentReport run_experiment(const ExperimentConfig& config);
std::string report_json(const ExperimentReport& report);

struct CliParseResult {
  bool ok = false;
  bool help = false;
  ExperimentConfig config;
  std::string error;
};

CliParseResult parse_cli(int argc, char** argv);
std::string usage();

}  // namespace tinylpu
