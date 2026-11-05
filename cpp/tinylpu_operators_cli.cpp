#include "tinylpu_operators.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  const tinylpu_operators::CliParseResult parsed = tinylpu_operators::parse_cli(argc, argv);
  if (parsed.help) {
    std::cout << tinylpu_operators::usage() << '\n';
    return 0;
  }
  if (!parsed.ok) {
    std::cerr << "error: " << parsed.error << '\n'
              << "usage: " << tinylpu_operators::usage() << '\n';
    return 2;
  }

  try {
    std::ofstream trace_file;
    tinylpu_operators::TraceSink sink;
    if (!parsed.trace_path.empty()) {
      trace_file.open(parsed.trace_path, std::ios::out | std::ios::trunc);
      if (!trace_file) throw std::runtime_error("cannot open trace file '" + parsed.trace_path + "'");
      trace_file << "relative,lane,issue_output,issue_term,issue_a_addr,issue_b_addr,issue,"
                    "commit_output,commit_term,commit_value,commit,issue_value,acc,store,done,"
                    "a_reads,b_reads\n";
      sink = [&trace_file](const tinylpu_operators::TraceEvent& event) {
        trace_file << event.relative << ',' << event.lane << ',' << event.issue_output << ','
                   << event.issue_term << ',' << event.issue_a_addr << ',' << event.issue_b_addr << ','
                   << (event.issue ? 1 : 0) << ',' << event.commit_output << ',' << event.commit_term << ','
                   << event.commit_value << ',' << (event.commit ? 1 : 0) << ',' << event.issue_value << ','
                   << event.acc << ',' << (event.store ? 1 : 0) << ',' << (event.done ? 1 : 0) << ','
                   << event.a_reads << ',' << event.b_reads << '\n';
      };
    }
    const tinylpu_operators::OperatorReport report = tinylpu_operators::run_experiment(
        parsed.config,
        parsed.seed,
        parsed.a_set ? &parsed.a : nullptr,
        parsed.b_set ? &parsed.b : nullptr,
        sink);
    std::cout << tinylpu_operators::report_json(report) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
