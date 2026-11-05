#include "tinylpu_plan.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

void write_trace_header(std::ofstream& stream) {
  stream << "sequence,kind,instruction,opcode,relative,lane,issue_row,issue_index,"
            "commit_row,commit_index,issue,commit,done,issue_product,commit_product,acc\n";
}

void write_trace_event(std::ofstream& stream,
                       const tinylpu_plan::TraceEvent& event) {
  stream << event.sequence << ',' << event.kind << ',' << event.instruction << ','
         << tinylpu_plan::opcode_name(event.opcode) << ',' << event.relative << ','
         << event.lane << ',' << event.issue_row << ',' << event.issue_index << ','
         << event.commit_row << ',' << event.commit_index << ','
         << (event.issue ? 1 : 0) << ',' << (event.commit ? 1 : 0) << ','
         << (event.done ? 1 : 0) << ',' << event.issue_product << ','
         << event.commit_product << ',' << event.acc << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const tinylpu_plan::CliParseResult parsed = tinylpu_plan::parse_cli(argc, argv);
  if (parsed.help) {
    std::cout << tinylpu_plan::usage() << '\n';
    return 0;
  }
  if (!parsed.ok) {
    std::cerr << "error: " << parsed.error << '\n'
              << "usage: " << tinylpu_plan::usage() << '\n';
    return 2;
  }

  try {
    const tinylpu_plan::Program program =
        tinylpu_plan::load_program(parsed.program_path, parsed.weights_path);
    const std::vector<int> input =
        tinylpu_plan::parse_input_values(parsed.input_text);
    std::ofstream trace_file;
    tinylpu_plan::TraceSink trace_sink;
    if (!parsed.trace_path.empty()) {
      trace_file.open(parsed.trace_path, std::ios::out | std::ios::trunc);
      if (!trace_file) {
        throw std::runtime_error("cannot open trace file '" + parsed.trace_path + "'");
      }
      write_trace_header(trace_file);
      trace_sink = [&trace_file](const tinylpu_plan::TraceEvent& event) {
        write_trace_event(trace_file, event);
      };
    }
    const tinylpu_plan::SimulationReport report =
        tinylpu_plan::simulate(program, input, trace_sink);
    if (!parsed.trace_path.empty() && trace_file.fail()) {
      throw std::runtime_error("cannot write trace file");
    }
    std::cout << tinylpu_plan::report_json(report) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
