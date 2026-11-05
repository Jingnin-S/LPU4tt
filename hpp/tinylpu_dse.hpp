#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tinylpu_dse {

struct Config {
  int rows = 1;
  int k = 1;
  int lanes = 1;
  int depth = 2;
  bool row_drain = false;
};

struct ValidationResult {
  bool ok = false;
  std::string error;
};

ValidationResult validate_config(const Config& config);
ValidationResult validate_operands(const Config& config,
                                   const std::vector<int>& a,
                                   const std::vector<int>& weights);

struct DseReport {
  Config config;
  int macs = 0;
  int cycles = 0;
  int issue_events = 0;
  int commit_events = 0;
  std::vector<int> a;
  std::vector<int> weights;
  std::vector<std::int16_t> functional_result;
  std::vector<std::int16_t> cycle_result;
  int accumulator_bits = 16;
};

// Independent mathematical oracle. DseCore never calls this function.
std::vector<std::int16_t> golden(const Config& config,
                                 const std::vector<std::int8_t>& a,
                                 const std::vector<std::int8_t>& weights);

struct LaneEvent {
  int relative = 0;
  int lane = -1;
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

struct Tick {
  int relative = 0;
  std::vector<LaneEvent> events;
  bool done = false;
};

// Parameterized one-PE-per-lane timing model. Products are computed from the
// current operands on issue and committed after depth-1 ticks. The pipeline
// carries row/index/first/last metadata so rows may overlap without clearing a
// shared accumulator at the wrong time.
class DseCore {
 public:
  explicit DseCore(Config config);

  void set_operands(const std::vector<std::int8_t>& a,
                    const std::vector<std::int8_t>& weights);
  void reset();
  void start();
  Tick tick();

  const Config& config() const { return config_; }
  bool running() const { return running_; }
  bool done() const { return done_; }
  int relative() const { return relative_; }
  const std::vector<std::int16_t>& results() const { return results_; }

 private:
  struct PipeEntry {
    bool valid = false;
    int lane = -1;
    int row = -1;
    int index = -1;
    int product = 0;
    bool first = false;
    bool last = false;
  };

  bool issue_for(int relative, int lane, int& row, int& index) const;
  Config config_;
  std::vector<std::int8_t> a_;
  std::vector<std::int8_t> weights_;
  // One feedback accumulator per physical lane. Rows are serialized through
  // their lane, while completed values are retained separately by row.
  std::vector<std::int16_t> accumulators_;
  std::vector<std::int16_t> results_;
  std::vector<std::vector<PipeEntry>> pipeline_;
  int groups_ = 0;
  int group_stride_ = 0;
  int relative_ = 0;
  bool running_ = false;
  bool done_ = false;
};

std::string report_json(const DseReport& report);

struct CliParseResult {
  bool ok = false;
  bool help = false;
  Config config;
  std::uint32_t seed = 0;
  bool seed_set = false;
  std::vector<int> a;
  std::vector<int> weights;
  bool operands_set = false;
  std::string trace_path;
  std::string error;
};

CliParseResult parse_cli(int argc, char** argv);
std::string usage();

using TraceSink = std::function<void(const LaneEvent&)>;
DseReport run_experiment(const Config& config,
                         std::uint32_t seed,
                         const std::vector<int>* a_override = nullptr,
                         const std::vector<int>* weights_override = nullptr,
                         TraceSink trace_sink = {});

}  // namespace tinylpu_dse
