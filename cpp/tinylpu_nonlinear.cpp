#include "tinylpu_nonlinear.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tinylpu_nonlinear {
namespace {

const std::vector<std::string> kOps = {"softmax", "layernorm", "rmsnorm", "gelu", "silu", "sigmoid", "tanh"};

void validate(const Config& c, const std::vector<double>& x,
              const std::vector<double>& gamma, const std::vector<double>& beta) {
  if (std::find(kOps.begin(), kOps.end(), c.op) == kOps.end())
    throw std::invalid_argument("unsupported nonlinear operator");
  if (c.rows < 1 || c.rows > 16 || c.cols < 1 || c.cols > 64 ||
      x.size() != static_cast<std::size_t>(c.rows * c.cols) ||
      gamma.size() != static_cast<std::size_t>(c.cols) || beta.size() != gamma.size())
    throw std::invalid_argument("invalid nonlinear shape or affine parameter sizes");
  if (!std::isfinite(c.epsilon) || c.epsilon <= 0)
    throw std::invalid_argument("epsilon must be finite and positive");
  for (int n : {c.alu_latency, c.exp_latency, c.div_latency, c.sqrt_latency, c.erf_latency, c.tanh_latency})
    if (n < 1 || n > 1024) throw std::invalid_argument("primitive latency must be in [1,1024]");
  for (const auto* values : {&x, &gamma, &beta})
    for (double value : *values)
      if (!std::isfinite(value) || std::abs(value) > 1e6)
        throw std::invalid_argument("reference values must be finite and within [-1e6,1e6]");
}

double sigmoid(double value) {
  if (value >= 0) return 1 / (1 + std::exp(-value));
  const double e = std::exp(value);
  return e / (1 + e);
}

struct Node {
  std::string op;
  int a = -1, b = -1;
  double value = 0;
  bool done = false;
  int issue = -1, finish = -1;
};

struct Graph {
  std::vector<Node> nodes;
  std::vector<int> outputs;
  int value(double v) {
    nodes.push_back({"value", -1, -1, v, true, 0, 0});
    return static_cast<int>(nodes.size()) - 1;
  }
  int op(const std::string& name, int a, int b = -1) {
    nodes.push_back({name, a, b, 0, false, -1, -1});
    return static_cast<int>(nodes.size()) - 1;
  }
  int sum(const std::vector<int>& values) {
    int result = values.front();
    for (std::size_t i = 1; i < values.size(); ++i) result = op("add", result, values[i]);
    return result;
  }
};

bool sfu(const std::string& op) {
  return op == "exp" || op == "div" || op == "sqrt" || op == "erf" || op == "tanh";
}

int latency(const Config& c, const std::string& op) {
  if (op == "exp") return c.exp_latency;
  if (op == "div") return c.div_latency;
  if (op == "sqrt") return c.sqrt_latency;
  if (op == "erf") return c.erf_latency;
  if (op == "tanh") return c.tanh_latency;
  return c.alu_latency;
}

double evaluate(const Node& n, const Graph& g) {
  const double a = g.nodes[n.a].value;
  const double b = n.b >= 0 ? g.nodes[n.b].value : 0;
  if (n.op == "add") return a + b;
  if (n.op == "sub") return a - b;
  if (n.op == "mul") return a * b;
  if (n.op == "max") return std::max(a, b);
  if (n.op == "exp") return std::exp(a);
  if (n.op == "div") return a / b;
  if (n.op == "sqrt") return std::sqrt(a);
  if (n.op == "erf") return std::erf(a);
  if (n.op == "tanh") return std::tanh(a);
  throw std::runtime_error("unknown primitive");
}

Graph build(const Config& c, const std::vector<double>& x,
             const std::vector<double>& gamma, const std::vector<double>& beta) {
  Graph g;
  const int one = g.value(1), half = g.value(0.5), inv_sqrt2 = g.value(1 / std::sqrt(2.0));
  const int inv_n = g.value(1.0 / c.cols), eps = g.value(c.epsilon);
  for (int row = 0; row < c.rows; ++row) {
    std::vector<int> xs;
    for (int j = 0; j < c.cols; ++j) xs.push_back(g.value(x[row * c.cols + j]));
    if (c.op == "softmax") {
      int maximum = xs.front();
      for (int j = 1; j < c.cols; ++j) maximum = g.op("max", maximum, xs[j]);
      std::vector<int> es;
      for (int v : xs) es.push_back(g.op("exp", g.op("sub", v, maximum)));
      const int reciprocal = g.op("div", one, g.sum(es));
      for (int e : es) g.outputs.push_back(g.op("mul", e, reciprocal));
    } else if (c.op == "layernorm" || c.op == "rmsnorm") {
      std::vector<int> centered = xs;
      if (c.op == "layernorm") {
        const int mean = g.op("mul", g.sum(xs), inv_n);
        for (int j = 0; j < c.cols; ++j) centered[j] = g.op("sub", xs[j], mean);
      }
      std::vector<int> squares;
      for (int v : centered) squares.push_back(g.op("mul", v, v));
      const int mean_square = g.op("mul", g.sum(squares), inv_n);
      const int inv = g.op("div", one, g.op("sqrt", g.op("add", mean_square, eps)));
      for (int j = 0; j < c.cols; ++j) {
        const int normalized = g.op("mul", centered[j], inv);
        const int scale = g.value(gamma[j]);
        int y = g.op("mul", normalized, scale);
        if (c.op == "layernorm") y = g.op("add", y, g.value(beta[j]));
        g.outputs.push_back(y);
      }
    } else {
      for (int v : xs) {
        int y;
        if (c.op == "gelu") {
          const int e = g.op("erf", g.op("mul", v, inv_sqrt2));
          const int h = g.op("mul", v, half);
          const int gate = g.op("add", one, e);
          y = g.op("mul", h, gate);
        } else if (c.op == "tanh") {
          y = g.op("tanh", v);
        } else {
          // Stable identity avoids exponential overflow; the reference uses
          // a separate exp-based sigmoid calculation.
          y = g.op("mul", half, g.op("add", one, g.op("tanh", g.op("mul", half, v))));
          if (c.op == "silu") y = g.op("mul", v, y);
        }
        g.outputs.push_back(y);
      }
    }
  }
  return g;
}

void array(std::ostringstream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}

}  // namespace

std::vector<double> reference(const Config& c, const std::vector<double>& x,
                              const std::vector<double>& gamma, const std::vector<double>& beta) {
  validate(c, x, gamma, beta);
  std::vector<double> y(x.size());
  for (int row = 0; row < c.rows; ++row) {
    const int base = row * c.cols;
    if (c.op == "softmax") {
      double m = x[base];
      for (int j = 1; j < c.cols; ++j) m = std::max(m, x[base + j]);
      double denominator = 0;
      for (int j = 0; j < c.cols; ++j) denominator += std::exp(x[base + j] - m);
      for (int j = 0; j < c.cols; ++j) y[base + j] = std::exp(x[base + j] - m) / denominator;
    } else if (c.op == "layernorm" || c.op == "rmsnorm") {
      double mean = 0, sum = 0;
      if (c.op == "layernorm") {
        for (int j = 0; j < c.cols; ++j) mean += x[base + j];
        mean /= c.cols;
      }
      for (int j = 0; j < c.cols; ++j) { const double d = x[base + j] - mean; sum += d * d; }
      const double denominator = std::sqrt(sum / c.cols + c.epsilon);
      for (int j = 0; j < c.cols; ++j)
        y[base + j] = (x[base + j] - mean) / denominator * gamma[j] + (c.op == "layernorm" ? beta[j] : 0);
    } else {
      for (int j = 0; j < c.cols; ++j) {
        const double v = x[base + j];
        if (c.op == "gelu") y[base + j] = 0.5 * v * (1 + std::erf(v / std::sqrt(2.0)));
        else if (c.op == "silu") y[base + j] = v * sigmoid(v);
        else if (c.op == "sigmoid") y[base + j] = sigmoid(v);
        else y[base + j] = std::tanh(v);
      }
    }
  }
  return y;
}

Report run(const Config& c, const std::vector<double>& x,
           const std::vector<double>& gamma, const std::vector<double>& beta) {
  validate(c, x, gamma, beta);
  Report report;
  report.config = c;
  report.x = x; report.gamma = gamma; report.beta = beta;
  report.golden = reference(c, x, gamma, beta);
  Graph graph = build(c, x, gamma, beta);
  int remaining = 0, sfu_free = 1;
  for (const auto& n : graph.nodes) if (!n.done) ++remaining;
  int cycle = 1;
  while (remaining > 0) {
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
      auto& n = graph.nodes[i];
      if (!n.done && n.finish == cycle) {
        n.value = evaluate(n, graph);
        if (!std::isfinite(n.value)) throw std::runtime_error("nonfinite primitive result");
        n.done = true;
        --remaining;
        report.events.push_back({static_cast<int>(i), sfu(n.op) ? "sfu" : "alu", n.op,
                                 n.issue, n.finish, n.a, n.b, n.value});
      }
    }
    if (!remaining) break;
    bool alu_issued = false, sfu_issued = false;
    for (auto& n : graph.nodes) {
      if (n.done || n.issue >= 0 || !graph.nodes[n.a].done || (n.b >= 0 && !graph.nodes[n.b].done)) continue;
      const bool special = sfu(n.op);
      if ((special && (sfu_issued || cycle < sfu_free)) || (!special && alu_issued)) continue;
      n.issue = cycle;
      n.finish = cycle + latency(c, n.op);
      ++report.primitive_counts[n.op];
      if (special) {
        sfu_issued = true;
        sfu_free = n.finish;
        report.sfu_busy_cycles += latency(c, n.op);
      } else {
        alu_issued = true;
        ++report.alu_issues;
      }
    }
    ++cycle;
    if (cycle > 2000000) throw std::runtime_error("nonlinear scheduler exceeded cycle budget");
  }
  report.cycles = cycle;
  for (int id : graph.outputs) report.cycle_result.push_back(graph.nodes[id].value);
  double conditioning_tolerance = 0;
  if (c.op == "layernorm") {
    double x_scale = 0, gamma_scale = 0;
    for (double v : x) x_scale = std::max(x_scale, std::abs(v));
    for (double v : gamma) gamma_scale = std::max(gamma_scale, std::abs(v));
    // Constant reciprocal multiplication versus division can differ by a
    // rounding unit in the mean, amplified by 1/sqrt(epsilon) near zero var.
    conditioning_tolerance = 8 * std::numeric_limits<double>::epsilon() *
                             x_scale * gamma_scale / std::sqrt(c.epsilon);
  }
  for (std::size_t i = 0; i < report.golden.size(); ++i)
    if (std::abs(report.cycle_result[i] - report.golden[i]) >
        1e-9 * (1 + std::abs(report.golden[i])) + conditioning_tolerance)
      throw std::runtime_error("nonlinear DAG differs from independent reference");
  return report;
}

std::string report_json(const Report& r) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\"op\":\"" << r.config.op << "\",\"shape\":[" << r.config.rows << ',' << r.config.cols
      << "],\"model_kind\":\"binary64_reference_and_assumed_primitive_DAG\",\"hardware_precision\":null,\"cycles\":" << r.cycles
      << ",\"alu_issues\":" << r.alu_issues << ",\"sfu_busy_cycles\":" << r.sfu_busy_cycles
      << ",\"epsilon\":" << r.config.epsilon << ",\"x\":";
  array(out, r.x); out << ",\"gamma\":"; array(out, r.gamma); out << ",\"beta\":"; array(out, r.beta);
  out << ",\"golden\":"; array(out, r.golden); out << ",\"cycle_result\":"; array(out, r.cycle_result);
  out << ",\"primitive_counts\":{";
  bool first = true;
  for (const auto& item : r.primitive_counts) { if (!first) out << ','; first = false; out << '"' << item.first << "\":" << item.second; }
  out << "},\"latencies\":{\"alu\":" << r.config.alu_latency << ",\"exp\":" << r.config.exp_latency
      << ",\"div\":" << r.config.div_latency << ",\"sqrt\":" << r.config.sqrt_latency
      << ",\"erf\":" << r.config.erf_latency << ",\"tanh\":" << r.config.tanh_latency
      << "},\"required_units\":[\"scalar arithmetic pipeline (II=1)\",\"serialized special-function unit\",\"operand register file\"]}";
  return out.str();
}

}  // namespace tinylpu_nonlinear
