#include "tinylpu_nonlinear.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    tinylpu_nonlinear::Config c;
    std::string trace;
    int seed = 1;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout << "tinylpu_nonlinear --op softmax|layernorm|rmsnorm|gelu|silu|sigmoid|tanh --rows R --cols C [--seed 1] [--sfu-scale 1] [--trace FILE]\n";
        return 0;
      }
      if (i + 1 == argc) throw std::invalid_argument("missing option value");
      const std::string value = argv[++i];
      if (option == "--op") c.op = value;
      else if (option == "--trace") trace = value;
      else {
        std::size_t used = 0;
        int n = std::stoi(value, &used);
        if (used != value.size()) throw std::invalid_argument("integer option required");
        if (option == "--rows") c.rows = n;
        else if (option == "--cols") c.cols = n;
        else if (option == "--seed") { if (n < 0) throw std::invalid_argument("seed must be nonnegative"); seed = n; }
        else if (option == "--sfu-scale") {
          if (n < 1 || n > 16) throw std::invalid_argument("SFU scale must be in [1,16]");
          c.exp_latency = c.tanh_latency = 8 * n;
          c.div_latency = c.sqrt_latency = c.erf_latency = 16 * n;
        } else throw std::invalid_argument("unknown option " + option);
      }
    }
    if (c.rows < 1 || c.rows > 16 || c.cols < 1 || c.cols > 64)
      throw std::invalid_argument("shape out of range");
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> dist(-16, 16);
    std::vector<double> x(static_cast<std::size_t>(c.rows * c.cols));
    for (double& v : x) v = dist(rng) / 4.0;
    std::vector<double> gamma(static_cast<std::size_t>(c.cols)), beta(gamma.size());
    for (int j = 0; j < c.cols; ++j) {
      gamma[j] = 0.75 + 0.125 * (j % 5);
      beta[j] = -0.25 + 0.125 * (j % 3);
    }
    const auto result = tinylpu_nonlinear::run(c, x, gamma, beta);
    if (!trace.empty()) {
      std::ofstream stream(trace);
      if (!stream) throw std::runtime_error("cannot open trace file");
      stream << "node,resource,primitive,issue,commit,a,b,value\n";
      for (const auto& e : result.events)
        stream << e.node << ',' << e.resource << ',' << e.primitive << ',' << e.issue << ',' << e.commit << ',' << e.a << ',' << e.b << ',' << e.value << '\n';
      if (!stream) throw std::runtime_error("cannot write trace file");
    }
    std::cout << tinylpu_nonlinear::report_json(result) << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "nonlinear model: " << e.what() << '\n';
    return 1;
  }
}
