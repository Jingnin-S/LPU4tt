#include "tinylpu_dse.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  const tinylpu_dse::CliParseResult parsed = tinylpu_dse::parse_cli(argc, argv);
  if (parsed.help) {
    std::cout << tinylpu_dse::usage() << '\n';
    return 0;
  }
  if (!parsed.ok) {
    std::cerr << "error: " << parsed.error << '\n'
              << "usage: " << tinylpu_dse::usage() << '\n';
    return 2;
  }

  try {
    std::ofstream trace_file;
    tinylpu_dse::TraceSink sink;
    if (!parsed.trace_path.empty()) {
      trace_file.open(parsed.trace_path, std::ios::out | std::ios::trunc);
      if (!trace_file) {
        throw std::runtime_error("cannot open trace file '" + parsed.trace_path + "'");
      }
      trace_file << "relative,lane,row,index,commit_row,commit_index,issue,commit,product,issue_product,commit_product,acc,store,done\n";
      sink = [&trace_file](const tinylpu_dse::LaneEvent& event) {
        const int row = event.issue ? event.issue_row : event.commit_row;
        const int index = event.issue ? event.issue_index : event.commit_index;
        const int product = event.issue ? event.issue_product : event.commit_product;
        trace_file << event.relative << ',' << event.lane << ',' << row << ',' << index << ','
                   << event.commit_row << ',' << event.commit_index << ','
                   << (event.issue ? 1 : 0) << ',' << (event.commit ? 1 : 0) << ',' << product << ','
                   << event.issue_product << ',' << event.commit_product << ',' << event.acc << ','
                   << (event.store ? 1 : 0) << ',' << (event.done ? 1 : 0) << '\n';
      };
    }
    const tinylpu_dse::DseReport report = tinylpu_dse::run_experiment(
        parsed.config,
        parsed.seed,
        parsed.operands_set ? &parsed.a : nullptr,
        parsed.operands_set ? &parsed.weights : nullptr,
        sink);
    std::cout << tinylpu_dse::report_json(report) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
