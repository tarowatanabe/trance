//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

// transform and normalize treebank...

#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iterator>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>

#include <boost/fusion/tuple.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <boost/xpressive/xpressive.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/shared_ptr.hpp>

#include "utils/program_options.hpp"
#include "utils/compress_stream.hpp"
#include "utils/space_separator.hpp"
#include "utils/piece.hpp"
#include "utils/getline.hpp"

typedef boost::filesystem::path path_type;

// tree-bank parser...

struct treebank_type
{
  typedef std::vector<treebank_type> antecedents_type;

  std::string cat_;
  antecedents_type antecedents_;
  bool removed_;
  
  treebank_type() : removed_(false) {}
  treebank_type(const std::string& cat) : cat_(cat), removed_(false) {}

  void clear()
  {
    cat_.clear();
    antecedents_.clear();
    removed_ = false;
  }
};

BOOST_FUSION_ADAPT_STRUCT(
			  treebank_type,
			  (std::string, cat_)
			  (std::vector<treebank_type>, antecedents_)
			  )

template <typename Iterator>
struct penntreebank_grammar : boost::spirit::qi::grammar<Iterator, treebank_type(), boost::spirit::standard::space_type>
{
  penntreebank_grammar() : penntreebank_grammar::base_type(root)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;

    comment_last %= qi::no_skip[!(*standard::blank >> qi::lit('(')) >> *(standard::char_ - qi::eol) >> qi::eoi];
    comment      %= qi::no_skip[*(!(*standard::blank >> qi::lit('(')) >> *(standard::char_ - qi::eol) >> qi::eol) >> -comment_last];
    
    cat %= qi::lexeme[+(standard::char_ - standard::space - '(' - ')')];
    treebank %= qi::hold['(' >> cat >> +treebank >> ')'] | cat;
    root %= (qi::omit[comment]
	     >> (qi::hold['(' >> cat >> +treebank >> ')']
		 | qi::hold['(' >> qi::attr("ROOT") >> +treebank >> ')']
		 | qi::hold[qi::lit('(') >> qi::attr("") >> qi::lit('(') >> qi::lit(')') >> qi::lit(')')])
	     >> qi::omit[comment]);
  }
  
  boost::spirit::qi::rule<Iterator, std::string()> comment;
  boost::spirit::qi::rule<Iterator, std::string()> comment_last;

  boost::spirit::qi::rule<Iterator, std::string(),   boost::spirit::standard::space_type> cat;
  boost::spirit::qi::rule<Iterator, treebank_type(), boost::spirit::standard::space_type> treebank;
  boost::spirit::qi::rule<Iterator, treebank_type(), boost::spirit::standard::space_type> root;
};

typedef std::vector<std::string, std::allocator<std::string> > sentence_type;

void transform_leaf(const treebank_type& treebank, sentence_type& sent) 
{
  if (treebank.antecedents_.empty())
    sent.push_back(treebank.cat_);
  else
    for (treebank_type::antecedents_type::const_iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
      transform_leaf(*aiter, sent);
}

void transform_normalize(treebank_type& treebank)
{
  // no terminal...
  if (treebank.antecedents_.empty()) return;
  
  // normalize treebank-category...
  if (treebank.cat_.size() == 1) {
#if 0
    switch (treebank.cat_[0]) {
    case '.' : treebank.cat_ = "PERIOD"; break;
    case ',' : treebank.cat_ = "COMMA"; break;
    case ':' : treebank.cat_ = "COLON"; break;
    case ';' : treebank.cat_ = "SEMICOLON"; break;
    }
#endif
  } else {
    namespace xpressive = boost::xpressive;
    
    typedef xpressive::basic_regex<utils::piece::const_iterator> pregex;
    typedef xpressive::match_results<utils::piece::const_iterator> pmatch;

    static pregex re = (xpressive::s1= -+(~xpressive::_s)) >> (xpressive::as_xpr('-') | xpressive::as_xpr('=')) >> +(~xpressive::_s);
    
    pmatch what;
    if (xpressive::regex_match(utils::piece(treebank.cat_), what, re))
      treebank.cat_ = what[1];
  }
  
  for (treebank_type::antecedents_type::iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
    transform_normalize(*aiter);
}


template <typename Iterator>
struct terminal_parser : boost::spirit::qi::grammar<Iterator, std::string()>
{
  terminal_parser() : terminal_parser::base_type(terminal)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    escape_char.add
      ("-LRB-", '(')
      ("-RRB-", ')')
      ("-LSB-", '[')
      ("-RSB-", ']')
      ("-LCB-", '{')
      ("-RCB-", '}')
      ("-PLUS-", '+') // added for ATB
      ("\\/", '/')
      ("\\*", '*');
    
    terminal %= +(escape_char | standard::char_);
  }
  
  boost::spirit::qi::symbols<char, char> escape_char;
  boost::spirit::qi::rule<Iterator, std::string()> terminal;
};

void transform_unescape(treebank_type& treebank)
{
  if (treebank.antecedents_.empty()) {
    // terminal...
    
    namespace qi = boost::spirit::qi;
    
    static terminal_parser<std::string::const_iterator> parser;

    std::string::const_iterator iter = treebank.cat_.begin();
    std::string::const_iterator iter_end = treebank.cat_.end();

    std::string terminal;
    
    if (! qi::parse(iter, iter_end, parser, terminal) || iter != iter_end)
      throw std::runtime_error("terminal parsing failed?");
    
    treebank.cat_.swap(terminal);
    
  } else
    for (treebank_type::antecedents_type::iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
      transform_unescape(*aiter);
}

void transform_remove_none(treebank_type& treebank)
{
  if (treebank.cat_ == "-NONE-") {
    treebank.removed_ = true;
    return;
  }
  
  if (treebank.antecedents_.empty()) return;
  
  treebank_type::antecedents_type antecedents;
  
  for (treebank_type::antecedents_type::iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
    if (aiter->cat_ != "-NONE-") {
      transform_remove_none(*aiter);
      if (! aiter->removed_)
	antecedents.push_back(*aiter);
    }
  
  treebank.removed_ = antecedents.empty();
  treebank.antecedents_.swap(antecedents);
}

void transform_cycle(treebank_type& treebank)
{
  // no terminal...
  if (treebank.antecedents_.empty()) return;
  
  for (treebank_type::antecedents_type::iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
    transform_cycle(*aiter);

  // unary rule + the same category...
  // TODO: buggy.. why?
  if (treebank.antecedents_.size() == 1
      && treebank.antecedents_.front().antecedents_.size() == 1
      && treebank.cat_ == treebank.antecedents_.front().cat_) {
    treebank_type::antecedents_type antecedents;
    
    antecedents.swap(treebank.antecedents_.front().antecedents_);
    
    treebank.antecedents_.swap(antecedents);
  }
}

bool treebank_validate(const treebank_type& treebank)
{
  if (treebank.cat_.empty() && treebank.antecedents_.empty())
    return true;
  
  if (treebank.antecedents_.empty())
    return false;
  
  treebank_type::antecedents_type::const_iterator aiter_end = treebank.antecedents_.end();
  for (treebank_type::antecedents_type::const_iterator aiter = treebank.antecedents_.begin(); aiter != aiter_end; ++ aiter)
    if (aiter->antecedents_.empty())
      return false;
  
  return true;
}

std::ostream& treebank_output(const treebank_type& treebank, std::ostream& os)
{
  if (treebank.antecedents_.empty())
    os << treebank.cat_;
  else {
    os << '(';
    os << treebank.cat_;
    os << ' ';
    
    for (treebank_type::antecedents_type::const_iterator aiter = treebank.antecedents_.begin(); aiter != treebank.antecedents_.end(); ++ aiter)
      treebank_output(*aiter, os);
    os << ')';
  }
  
  return os;
}


path_type input_file = "-";
path_type output_file = "-";

std::string root_symbol;
bool normalize = false;
bool remove_none = false;
bool remove_cycle = false;
bool unescape_terminal = false;

bool leaf_mode = false;
bool treebank_mode = false;

bool validate = false;

int debug = 0;

void options(int argc, char** argv);

int main(int argc, char** argv)
{
  try {
    options(argc, argv);

    if (int(leaf_mode) + treebank_mode > 1)
      throw std::runtime_error("multiple output options specified: leaf/treebank(default: treebank)");
    if (int(leaf_mode) + treebank_mode == 0)
      treebank_mode = true;
    
    typedef boost::spirit::istream_iterator iter_type;

    const bool flush_output = (output_file == "-"
			       || (boost::filesystem::exists(output_file)
				   && ! boost::filesystem::is_regular_file(output_file)));
    
    penntreebank_grammar<iter_type> grammar;

    treebank_type   parsed;
    sentence_type   sent;
    
    std::string line;
    
    utils::compress_istream is(input_file, 1024 * 1024);
    utils::compress_ostream os(output_file, 1024 * 1024);
    
    is.unsetf(std::ios::skipws);
    iter_type iter(is);
    iter_type iter_end;

    while (iter != iter_end) {
      parsed.clear();
      
      if (! boost::spirit::qi::phrase_parse(iter, iter_end, grammar, boost::spirit::standard::space, parsed)) {
	std::string buffer;
	for (int i = 0; i != 64 && iter != iter_end; ++ i, ++iter)
	  buffer += *iter;
	
	throw std::runtime_error("parsing failed: " + buffer);
      }

      if (! root_symbol.empty())
	parsed.cat_ = root_symbol;
      else if (parsed.cat_.empty())
	parsed.cat_ = "ROOT";

      if (validate)
	if (! treebank_validate(parsed))
	  throw std::runtime_error("invalid tree");
      
      if (remove_none)
	transform_remove_none(parsed);
      
      if (normalize)
	transform_normalize(parsed);
      
      if (remove_cycle)
	transform_cycle(parsed);

      if (unescape_terminal)
	transform_unescape(parsed);
      
      if (leaf_mode) {
	sent.clear();
	
	transform_leaf(parsed, sent);
	
	if (! sent.empty()) {
	  std::copy(sent.begin(), sent.end() - 1, std::ostream_iterator<std::string>(os, " "));
	  os << sent.back();
	}
      } else if (treebank_mode) {
	if (parsed.antecedents_.empty())
	  os << "(())";
	else
	  treebank_output(parsed, os);
      } 
      
      os << '\n';
      if (flush_output)
	os << std::flush;
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

    ("replace-root",   po::value<std::string>(&root_symbol), "replace root symbol")
    ("unescape",       po::bool_switch(&unescape_terminal),  "unescape terminal symbols, such as -LRB-, \\* etc.")
    ("normalize",      po::bool_switch(&normalize),          "normalize category, such as [,] [.] etc.")
    ("remove-none",    po::bool_switch(&remove_none),        "remove -NONE-")
    ("remove-cycle",   po::bool_switch(&remove_cycle),       "remove cycle unary rules")
    
    ("leaf",      po::bool_switch(&leaf_mode),     "output leaf nodes")
    ("treebank",  po::bool_switch(&treebank_mode), "output treebank")

    ("validate", po::bool_switch(&validate), "validate treebank")
    
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

