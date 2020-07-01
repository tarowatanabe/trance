//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <cstdio>
#include <unistd.h>

#define BOOST_DISABLE_ASSERTS
#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <stdexcept>
#include <iostream>
#include <map>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <trance/tree.hpp>
#include <trance/sentence.hpp>
#include <trance/oracle.hpp>
#include <trance/binarize.hpp>

#include "utils/bithack.hpp"
#include "utils/compress_stream.hpp"
#include "utils/getline.hpp"

typedef boost::filesystem::path path_type;

path_type input_file = "-";
path_type output_file = "-";

bool binarize_left = false;
bool binarize_right = false;

int debug = 0;

void options(int argc, char** argv);

int main(int argc, char** argv)
{
  try {
    options(argc, argv);

    if (int(binarize_left) + binarize_right > 1)
      throw std::runtime_error("either one of --binarize-{left,right}");

    if (int(binarize_left) + binarize_right == 0)
      binarize_left = true;

    const bool flush_output = (output_file == "-"
			       || (boost::filesystem::exists(output_file)
				   && ! boost::filesystem::is_regular_file(output_file)));

    utils::compress_istream is(input_file, 1024 * 1024);
    utils::compress_ostream os(output_file, 1024 * 1024);

    trance::Oracle oracle;
    trance::Tree tree;

    while (is >> tree) {
      oracle.assign(tree, binarize_left);

      bool init = true;
      trance::Oracle::action_set_type::const_iterator oiter_end = oracle.actions_.end();
      for (trance::Oracle::action_set_type::const_iterator oiter = oracle.actions_.begin(); oiter != oiter_end; ++ oiter) {
	if (! init)
	  os << ' ';
	init = false;

	os << oiter->operation_ << "-" << oiter->label_;

	if (oiter->operation_.shift())
	  os << ":" << oiter->head_;
      }

      if (flush_output)
	os << std::endl;
      else
	os << '\n';
    }

  } catch (const std::exception& err) {
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

    ("binarize-left",  po::bool_switch(&binarize_left),  "left recursive (or left heavy) binarization (default)")
    ("binarize-right", po::bool_switch(&binarize_right), "right recursive (or right heavy) binarization")

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
