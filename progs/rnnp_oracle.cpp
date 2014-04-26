//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <cstdio>
#include <unistd.h>

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <stdexcept>
#include <iostream>
#include <map>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <rnnp/grammar.hpp>
#include <rnnp/model.hpp>
#include <rnnp/parser_oracle.hpp>
#include <rnnp/derivation.hpp>
#include <rnnp/tree.hpp>

#include "utils/lockfree_list_queue.hpp"
#include "utils/bithack.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"
#include "utils/getline.hpp"
#include "utils/random_seed.hpp"

#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

typedef rnnp::Tree          tree_type;
typedef rnnp::Grammar       grammar_type;
typedef rnnp::Model         model_type;

typedef boost::filesystem::path path_type;

path_type input_file = "-";
path_type output_file = "-";

path_type grammar_file;
path_type model_file;
int hidden_size = 128;
int embedding_size = 32;

int beam_size = 100;
int kbest_size = 1;
int unary_size = 3;

bool binarize_left = false;
bool binarize_right = false;

// this is for debugging purpose...
bool randomize = false;
path_type embedding_file;

int threads = 1;

int debug = 0;

void parse(const grammar_type& grammar,
	   const model_type& theta,
	   const path_type& input_path,
	   const path_type& output_path);
void options(int argc, char** argv);

int main(int argc, char** argv)
{
  try {
    options(argc, argv);
    
    threads = utils::bithack::max(1, threads);
  
    if (beam_size <= 0)
      throw std::runtime_error("invalid beam size: " + utils::lexical_cast<std::string>(beam_size));
    if (kbest_size <= 0)
      throw std::runtime_error("invalid kbest size: " + utils::lexical_cast<std::string>(kbest_size));
    if (unary_size < 0)
      throw std::runtime_error("invalid unary size: " + utils::lexical_cast<std::string>(unary_size));

    if (grammar_file  != "-" && ! boost::filesystem::exists(grammar_file))
      throw std::runtime_error("no grammar file? " + grammar_file.string());
    
    if (int(binarize_left) + binarize_right > 1)
      throw std::runtime_error("either one of --binarize-{left,right}");
    
    if (int(binarize_left) + binarize_right == 0)
      binarize_left = true;

    grammar_type grammar(grammar_file);
    
    if (debug)
      std::cerr << "binary: " << grammar.binary_size()
		<< " unary: " << grammar.unary_size()
		<< " preterminal: " << grammar.preterminal_size()
		<< std::endl;
    
    model_type theta(hidden_size, embedding_size, grammar);

    if (! model_file.empty()) {
      if (! boost::filesystem::exists(model_file))
	throw std::runtime_error("no model file?");
      
      theta.read(model_file);
    } else {
      if (randomize) {
	boost::mt19937 generator;
	generator.seed(utils::random_seed());
	
	theta.random(generator);
      }
      
      if (! embedding_file.empty())
	theta.read_embedding(embedding_file);
    }
    
    parse(grammar, theta, input_file, output_file);
    
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}

struct MapReduce
{
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;
  
  typedef uint64_t    id_type;
  typedef std::string buffer_type;
  
  struct id_buffer_type
  {
    id_type id_;
    buffer_type buffer_;
    
    id_buffer_type() : id_(id_type(-1)), buffer_() {}
    id_buffer_type(const id_type& id, const buffer_type& buffer) : id_(id), buffer_(buffer) {}

    void clear()
    {
      id_ = id_type(-1);
      buffer_.clear();
    }
    
    void swap(id_buffer_type& x)
    {
      std::swap(id_, x.id_);
      buffer_.swap(x.buffer_);
    }
  };
  
  typedef utils::lockfree_list_queue<id_buffer_type, std::allocator<id_buffer_type> > queue_type;
};

namespace std
{
  inline
  void swap(MapReduce::id_buffer_type& x, MapReduce::id_buffer_type& y)
  {
    x.swap(y);
  }
};


struct Mapper : public MapReduce
{
  Mapper(const grammar_type& grammar,
	 const model_type&   theta,
	 queue_type& mapper,
	 queue_type& reducer)
    : grammar_(grammar),
      theta_(theta),
      mapper_(mapper),
      reducer_(reducer) {}
  
  struct real_precision : boost::spirit::karma::real_policies<double>
  {
    static unsigned int precision(double) 
    { 
      return 10;
    }
  };
  
  void operator()()
  {
    typedef rnnp::ParserOracle parser_type;
    typedef rnnp::Derivation derivation_type;
    
    typedef parser_type::derivation_set_type derivation_set_type;
    typedef std::vector<char, std::allocator<char> > buf_type;
    
    parser_type parser(beam_size, unary_size, binarize_left);
    
    id_buffer_type mapped;
    id_buffer_type reduced;
    
    tree_type input;
    derivation_type derivation;
    derivation_set_type derivations;
    buf_type buf;
    
    for (;;) {
      mapper_.pop_swap(mapped);
      
      if (mapped.id_ == id_type(-1)) break;
      
      input.assign(mapped.buffer_);
      
      parser(input, grammar_, theta_, kbest_size, derivations);
      
      // output kbest derivations
      reduced.id_ = mapped.id_;
      reduced.buffer_.clear();
      
      buf.clear();
      
      boost::iostreams::filtering_ostream os;
      os.push(boost::iostreams::back_inserter(buf));
      
      if (! derivations.empty()) {
	namespace karma = boost::spirit::karma;
	namespace standard = boost::spirit::standard;
	
	karma::real_generator<double, real_precision> double10;
      
	derivation_set_type::const_iterator diter_end = derivations.end();
	for (derivation_set_type::const_iterator diter = derivations.begin(); diter != diter_end; ++ diter) {
	  derivation.assign(*diter);
	  
	  os << reduced.id_ << " ||| " << derivation.tree_;
	  
	  karma::generate(std::ostream_iterator<char>(os), " ||| " << double10 << '\n', diter->score());
	}
      } else
	os << reduced.id_ << " ||| ||| 0" << '\n';
      
      os.reset();
      
      reduced.buffer_ = buffer_type(buf.begin(), buf.end());
      
      reducer_.push_swap(reduced);
    }
  }

  const grammar_type& grammar_;
  const model_type&   theta_;
  
  queue_type& mapper_;
  queue_type& reducer_;
};

struct Reducer : public MapReduce
{
  Reducer(const path_type& path,
	  queue_type& reducer)
    : path_(path),
      reducer_(reducer) {}

  typedef std::map<id_type, std::string, std::less<id_type>,
		   std::allocator<std::pair<const id_type, std::string> > > buffer_map_type;
  
  void operator()()
  {
    const bool flush_output = (path_ == "-"
			       || (boost::filesystem::exists(path_)
				   && ! boost::filesystem::is_regular_file(path_)));
    
    utils::compress_ostream os(path_, 1024 * 1024);
    
    buffer_map_type maps;
    id_type id = 0;
    id_buffer_type  reduced;
    
    for (;;) {
      reducer_.pop_swap(reduced);
      
      if (reduced.id_ == id_type(-1) && reduced.buffer_.empty()) break;
      
      bool dump = false;
      
      if (reduced.id_ == id) {
	os << reduced.buffer_;
	dump = true;
	++ id;
      } else
	maps[reduced.id_].swap(reduced.buffer_);
      
      for (buffer_map_type::iterator iter = maps.find(id); iter != maps.end() && iter->first == id; /**/) {
	os << iter->second;
	dump = true;
	++ id;
	
	maps.erase(iter ++);
      }
      
      if (dump && flush_output)
	os << std::flush;
    }
    
    for (buffer_map_type::iterator iter = maps.find(id); iter != maps.end() && iter->first == id; /**/) {
      os << iter->second;
      ++ id;
      
      maps.erase(iter ++);
    }
    
    // we will do twice, in case we have wrap-around for id...!
    if (! maps.empty())
      for (buffer_map_type::iterator iter = maps.find(id); iter != maps.end() && iter->first == id; /**/) {
	os << iter->second;
	++ id;
	
	maps.erase(iter ++);
      }
    
    if (flush_output)
      os << std::flush;
    
    if (! maps.empty())
      throw std::runtime_error("id mismatch! expecting: " + utils::lexical_cast<std::string>(id)
			       + " next: " + utils::lexical_cast<std::string>(maps.begin()->first)
			       + " renamining: " + utils::lexical_cast<std::string>(maps.size()));
  }
  
  const path_type path_;
  queue_type& reducer_;
};

void parse(const grammar_type& grammar,
	   const model_type& theta,
	   const path_type& input_path,
	   const path_type& output_path)
{
  typedef MapReduce map_reduce_type;
  typedef Mapper    mapper_type;
  typedef Reducer   reducer_type;
  
  map_reduce_type::queue_type queue_mapper(threads);
  map_reduce_type::queue_type queue_reducer;
  
  boost::thread_group reducers;
  reducers.add_thread(new boost::thread(reducer_type(output_path, queue_reducer)));
  
  boost::thread_group mappers;
  for (int i = 0; i != threads; ++ i)
    mappers.add_thread(new boost::thread(mapper_type(grammar, theta, queue_mapper, queue_reducer)));
  
  map_reduce_type::id_buffer_type id_buffer;
  map_reduce_type::id_type id = 0;
  std::string line;
  
  utils::compress_istream is(input_path, 1024 * 1024);

  while (utils::getline(is, line)) {
    id_buffer.id_ = id;
    id_buffer.buffer_.swap(line);
    
    queue_mapper.push_swap(id_buffer);
    ++ id;
  }
  
  // terminate mappers
  for (int i = 0; i != threads; ++ i) {
    id_buffer.clear();
    queue_mapper.push_swap(id_buffer);
  }
  mappers.join_all();
  
  // terminate reducers
  id_buffer.clear();
  queue_reducer.push(id_buffer);
  reducers.join_all();
}

void options(int argc, char** argv)
{
  namespace po = boost::program_options;
  
  po::options_description opts_config("configuration options");
  opts_config.add_options()
    ("input",     po::value<path_type>(&input_file)->default_value(input_file),   "input file")
    ("output",    po::value<path_type>(&output_file)->default_value(output_file), "output file")
    
    ("grammar",    po::value<path_type>(&grammar_file),                      "grammar file")
    ("model",      po::value<path_type>(&model_file),                        "model file")
    
    ("hidden",    po::value<int>(&hidden_size)->default_value(hidden_size),       "hidden dimension")
    ("embedding", po::value<int>(&embedding_size)->default_value(embedding_size), "embedding dimension")
    
    ("beam",  po::value<int>(&beam_size)->default_value(beam_size),   "beam size")
    ("kbest", po::value<int>(&kbest_size)->default_value(kbest_size), "kbest size")
    ("unary", po::value<int>(&unary_size)->default_value(unary_size), "unary size")
    
    ("binarize-left",  po::bool_switch(&binarize_left),  "left recursive (or left heavy) binarization (default)")
    ("binarize-right", po::bool_switch(&binarize_right), "right recursive (or right heavy) binarization")
    
    ("randomize",      po::bool_switch(&randomize),           "randomize model parameters")
    ("word-embedding", po::value<path_type>(&embedding_file), "word embedding file");
    
  po::options_description opts_command("command line options");
  opts_command.add_options()
    ("config",  po::value<path_type>(),                    "configuration file")
    ("threads", po::value<int>(&threads)->default_value(threads), "# of threads")
    
    ("debug", po::value<int>(&debug)->implicit_value(1), "debug level")
    ("help", "help message");
  
  po::options_description desc_config;
  po::options_description desc_command;
  
  desc_config.add(opts_config);
  desc_command.add(opts_config).add(opts_command);
  
  po::variables_map variables;
  
  po::store(po::parse_command_line(argc, argv, desc_command, po::command_line_style::unix_style & (~po::command_line_style::allow_guessing)), variables);
  
  if (variables.count("config")) {
    const path_type path_config = variables["config"].as<path_type>();
    
    if (! boost::filesystem::exists(path_config))
      throw std::runtime_error("no config file: " + path_config.string());
    
    utils::compress_istream is(path_config);
    
    po::store(po::parse_config_file(is, desc_config), variables);
  }
  
  po::notify(variables);
  
  if (variables.count("help")) {
    std::cout << argv[0] << " [options]" << '\n' << desc_command << '\n';
    exit(0);
  }
}
