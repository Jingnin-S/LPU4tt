#include "tinylpu_esl.hpp"

#include <iostream>

int main(int argc, char** argv) {
  const tinylpu::CliParseResult parsed = tinylpu::parse_cli(argc, argv);
  if (parsed.help) {
    std::cout << tinylpu::usage() << '\n';
    return 0;
  }
  if (!parsed.ok) {
    std::cerr << "error: " << parsed.error << '\n' << "usage: " << tinylpu::usage() << '\n';
    return 2;
  }

  try {
    const tinylpu::ExperimentReport report = tinylpu::run_experiment(parsed.config);
    std::cout << tinylpu::report_json(report) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
