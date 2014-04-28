//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

//
// collect grammar
//

#include <iostream>

#include <rnnp/tree.hpp>
#include <rnnp/graphviz.hpp>

#include "utils/compress_stream.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

typedef boost::filesystem::path path_type;

typedef rnnp::Tree    tree_type;

path_type input_file = "-";
path_type output_file = "-";

int debug = 0;

void options(int argc, char** argv);

int main(int argc, char** argv)
{
  try {
    options(argc, argv);

    tree_type tree;
    
    utils::compress_istream is(input_file, 1024 * 1024);
    utils::compress_ostream os(output_file, 1024 * 1024);

    const bool flush_output = (input_file == "-"
			       || (boost::filesystem::exists(input_file)
				   && ! boost::filesystem::is_regular_file(input_file)));

    if (flush_output) {
      while (is >> tree)
	rnnp::graphviz(os, tree) << std::endl;
    } else {
      while (is >> tree)
	rnnp::graphviz(os, tree) << '\n';
    }
  }
  catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}


void options(int argc, char** argv)
{
  namespace po = boost::program_options;

  po::options_description desc("options");
  desc.add_options()
    ("input",     po::value<path_type>(&input_file)->default_value(input_file),   "input file")
    ("output",    po::value<path_type>(&output_file)->default_value(output_file), "output")
    
    ("debug", po::value<int>(&debug)->implicit_value(1), "debug level")
    
    ("help", "help message");
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::unix_style & (~po::command_line_style::allow_guessing)), vm);
  po::notify(vm);
  
  if (vm.count("help")) {
    std::cout << argv[0] << " [options]" << '\n' << desc << '\n';
    exit(0);
  }
}


