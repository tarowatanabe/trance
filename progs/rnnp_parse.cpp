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

#include <rnnp/sentence.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/signature.hpp>
#include <rnnp/model/model1.hpp>
#include <rnnp/model/model2.hpp>
#include <rnnp/model/model3.hpp>
#include <rnnp/model/model4.hpp>
#include <rnnp/parser.hpp>
#include <rnnp/derivation.hpp>

#include "utils/lockfree_list_queue.hpp"
#include "utils/bithack.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"
#include "utils/getline.hpp"
#include "utils/random_seed.hpp"
#include "utils/resource.hpp"

#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

typedef rnnp::Sentence      sentence_type;
typedef rnnp::Grammar       grammar_type;
typedef rnnp::Signature     signature_type;
typedef rnnp::Model         model_type;

typedef boost::filesystem::path path_type;

path_type input_file = "-";
path_type output_file = "-";

bool simple_mode = false;
bool forest_mode = false;

path_type grammar_file;
std::string signature_name = "none";

bool model_model1 = false;
bool model_model2 = false;
bool model_model3 = false;
bool model_model4 = false;

path_type model_file;
int hidden_size = 64;
int embedding_size = 32;

int beam_size = 50;
int kbest_size = 1;
int unary_size = 3;

// this is for debugging purpose...
bool randomize = false;
path_type embedding_file;

int threads = 1;

int debug = 0;

template <typename Theta>
void parse(const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
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

    if (simple_mode && forest_mode)
      throw std::runtime_error("either one of --simple or --forest");

    if (simple_mode && kbest_size > 1)
      throw std::runtime_error("--simple assumes --kbest 1");
    
    if (grammar_file  != "-" && ! boost::filesystem::exists(grammar_file))
      throw std::runtime_error("no grammar file? " + grammar_file.string());
    
    if (model_file.empty()) {
      if (int(model_model1) + model_model2 + model_model3 + model_model4 > 1)
	throw std::runtime_error("either one of --model{1,2,3,4}");
      
      if (int(model_model1) + model_model2 + model_model3 + model_model4 == 0)
	model_model2 = true;
    } else {
      if (int(model_model1) + model_model2 + model_model3 + model_model4)
	throw std::runtime_error("model file is specified via --model, but with --model{1,2,3,4}?");
      
      if (! boost::filesystem::exists(model_file))
	throw std::runtime_error("no model file? " + model_file.string());
      
      switch (model_type::model(model_file)) {
      case 1: model_model1 = true; break;
      case 2: model_model2 = true; break;
      case 3: model_model3 = true; break;
      case 4: model_model4 = true; break;
      default:
	throw std::runtime_error("invalid model file");
      }
    }
    
    grammar_type grammar(grammar_file);
    
    if (debug)
      std::cerr << "binary: " << grammar.binary_size()
		<< " unary: " << grammar.unary_size()
		<< " preterminal: " << grammar.preterminal_size()
		<< " terminals: " << grammar.terminal_.size()
		<< " non-terminals: " << grammar.non_terminal_.size()
		<< " POS: " << grammar.pos_.size()
		<< std::endl;

    signature_type::signature_ptr_type signature(signature_type::create(signature_name));

    if (model_model1) {
      rnnp::model::Model1 theta(hidden_size, embedding_size, grammar);
      
      parse(grammar, *signature, theta, input_file, output_file);
    } else if (model_model2) {
      rnnp::model::Model2 theta(hidden_size, embedding_size, grammar);
      
      parse(grammar, *signature, theta, input_file, output_file);
    } else if (model_model3) {
      rnnp::model::Model3 theta(hidden_size, embedding_size, grammar);
      
      parse(grammar, *signature, theta, input_file, output_file);
    } else if (model_model4) {
      rnnp::model::Model4 theta(hidden_size, embedding_size, grammar);
      
      parse(grammar, *signature, theta, input_file, output_file);
    } else
      throw std::runtime_error("no model?");
    
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


template <typename Theta>
struct Mapper : public MapReduce
{
  Mapper(const grammar_type& grammar,
	 const signature_type& signature,
	 const Theta&   theta,
	 queue_type& mapper,
	 queue_type& reducer)
    : grammar_(grammar),
      signature_(signature),
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
    typedef rnnp::Parser     parser_type;
    typedef rnnp::Derivation derivation_type;
    
    typedef parser_type::derivation_set_type derivation_set_type;
    typedef std::vector<char, std::allocator<char> > buf_type;
    
    parser_type parser(beam_size, unary_size);
    
    id_buffer_type mapped;
    id_buffer_type reduced;
    
    sentence_type input;
    derivation_type derivation;
    derivation_set_type derivations;
    buf_type buf;
    
    parsed_  = 0;
    skipped_ = 0;
    resource_.clear();

    signature_type::signature_ptr_type signature(signature_.clone());
    
    for (;;) {
      mapper_.pop_swap(mapped);
      
      if (mapped.id_ == id_type(-1)) break;
      
      input.assign(mapped.buffer_);
      
      utils::resource start;
      
      parser(input, grammar_, *signature, theta_, kbest_size, derivations);
      
      utils::resource end;
      
      parsed_  += 1;
      skipped_ += derivations.empty();
      resource_ += end - start;
      
      // output kbest derivations
      reduced.id_ = mapped.id_;
      reduced.buffer_.clear();
      
      buf.clear();
      
      boost::iostreams::filtering_ostream os;
      os.push(boost::iostreams::back_inserter(buf));
      
      if (simple_mode) {
	if (! derivations.empty()) {
	  derivation.assign(derivations.front());
	  os << derivation.tree_ << '\n';
	} else
	  os << "(())" << '\n';
      } else if (forest_mode) {
	derivation.assign(derivations);

	os << reduced.id_ << " ||| " << derivation.forest_ << '\n';
      } else {
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
      }
      
      os.reset();
      
      reduced.buffer_ = buffer_type(buf.begin(), buf.end());
      
      reducer_.push_swap(reduced);
    }
  }
  
  const grammar_type&   grammar_;
  const signature_type& signature_;
  const Theta&          theta_;
  
  queue_type& mapper_;
  queue_type& reducer_;
  
  size_type       parsed_;
  size_type       skipped_;
  utils::resource resource_;
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

template <typename Theta>
void parse(const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   const path_type& input_path,
	   const path_type& output_path)
{
  typedef MapReduce     map_reduce_type;
  typedef Mapper<Theta> mapper_type;
  typedef Reducer       reducer_type;
  
  if (! model_file.empty())
    theta.read(model_file);
  else {
    if (randomize) {
      boost::mt19937 generator;
      generator.seed(utils::random_seed());
      
      theta.random(generator);
    }
    
    if (! embedding_file.empty())
      theta.embedding(embedding_file);
  }
  
  if (debug) {
    const size_t terminals = std::count(theta.vocab_terminal_.begin(), theta.vocab_terminal_.end(), true);
    const size_t non_terminals = (theta.vocab_category_.size()
				  - std::count(theta.vocab_category_.begin(), theta.vocab_category_.end(),
					       model_type::symbol_type()));
    
    std::cerr << "terminals: " << terminals
	      << " non-terminals: " << non_terminals
	      << std::endl;
  }
  
  map_reduce_type::queue_type queue_mapper(threads);
  map_reduce_type::queue_type queue_reducer;
  
  boost::thread_group reducers;
  reducers.add_thread(new boost::thread(reducer_type(output_path, queue_reducer)));
  
  std::vector<mapper_type, std::allocator<mapper_type> > workers(threads,
								 mapper_type(grammar, signature, theta, queue_mapper, queue_reducer));
  
  boost::thread_group mappers;
  for (int i = 0; i != threads; ++ i)
    mappers.add_thread(new boost::thread(boost::ref(workers[i])));
  
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
  
  for (int i = 1; i != threads; ++ i) {
    workers.front().parsed_   += workers[i].parsed_;
    workers.front().skipped_  += workers[i].skipped_;
    workers.front().resource_ += workers[i].resource_;
  }
  
  if (debug)
    std::cerr << "parsed: " << workers.front().parsed_ << std::endl
	      << "skipped: " << workers.front().skipped_ << std::endl
	      << "cpu time:    " << workers.front().resource_.cpu_time() << std::endl
	      << "user time:   " << workers.front().resource_.user_time() << std::endl
	      << "thread time: " << workers.front().resource_.thread_time() << std::endl;
  
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
    
    ("simple",    po::bool_switch(&simple_mode), "output parse tree only")
    ("forest",    po::bool_switch(&forest_mode), "output forest")
    
    ("grammar",    po::value<path_type>(&grammar_file),                                    "grammar file")
    ("signature",  po::value<std::string>(&signature_name)->default_value(signature_name), "language specific signature")

    ("model1",    po::bool_switch(&model_model1), "parsing by model1")
    ("model2",    po::bool_switch(&model_model2), "parsing by model2 (default)")
    ("model3",    po::bool_switch(&model_model3), "parsing by model3")
    ("model4",    po::bool_switch(&model_model4), "parsing by model4")
    
    ("model",     po::value<path_type>(&model_file),                              "model file")
    ("hidden",    po::value<int>(&hidden_size)->default_value(hidden_size),       "hidden dimension")
    ("embedding", po::value<int>(&embedding_size)->default_value(embedding_size), "embedding dimension")
    
    ("beam",  po::value<int>(&beam_size)->default_value(beam_size),   "beam size")
    ("kbest", po::value<int>(&kbest_size)->default_value(kbest_size), "kbest size")
    ("unary", po::value<int>(&unary_size)->default_value(unary_size), "unary size")
    
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
