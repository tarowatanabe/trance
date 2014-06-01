//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <stdexcept>
#include <iostream>

#include <rnnp/evalb.hpp>
#include <rnnp/tree.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/signature.hpp>
#include <rnnp/model/model1.hpp>
#include <rnnp/model/model2.hpp>
#include <rnnp/model/model3.hpp>
#include <rnnp/model/model4.hpp>
#include <rnnp/model/model5.hpp>
#include <rnnp/parser.hpp>
#include <rnnp/parser_oracle.hpp>
#include <rnnp/loss.hpp>
#include <rnnp/learn_option.hpp>
#include <rnnp/derivation.hpp>

#include <rnnp/objective/margin_derivation.hpp>
#include <rnnp/objective/margin_evalb.hpp>
#include <rnnp/objective/margin_early.hpp>
#include <rnnp/objective/margin_late.hpp>
#include <rnnp/objective/margin_max.hpp>
#include <rnnp/objective/violation_early.hpp>
#include <rnnp/objective/violation_late.hpp>
#include <rnnp/objective/violation_max.hpp>

#include <rnnp/optimize/adagrad.hpp>
#include <rnnp/optimize/adadec.hpp>
#include <rnnp/optimize/adadelta.hpp>
#include <rnnp/optimize/sgd.hpp>

#include <rnnp/loss.hpp>

#include "codec/lz4.hpp"

#include "utils/mpi.hpp"
#include "utils/mpi_device.hpp"
#include "utils/mpi_device_bcast.hpp"
#include "utils/mpi_stream.hpp"
#include "utils/mpi_stream_simple.hpp"
#include "utils/mpi_traits.hpp"

#include "utils/compact_map.hpp"
#include "utils/lockfree_list_queue.hpp"
#include "utils/bithack.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"
#include "utils/getline.hpp"
#include "utils/random_seed.hpp"
#include "utils/resource.hpp"
#include "utils/chunk_vector.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

typedef rnnp::Sentence  sentence_type;
typedef rnnp::Tree      tree_type;
typedef rnnp::Grammar   grammar_type;
typedef rnnp::Signature signature_type;
typedef rnnp::Model     model_type;

typedef rnnp::LearnOption option_type;

typedef boost::filesystem::path path_type;

typedef std::vector<std::string, std::allocator<std::string> > opt_set_type;
typedef std::vector<option_type, std::allocator<option_type> > option_set_type;

typedef utils::chunk_vector<tree_type, 4096 / sizeof(tree_type), std::allocator<tree_type> > tree_set_type;

path_type input_file = "-";
path_type test_file;
path_type output_file;

path_type grammar_file;
std::string signature_name = "none";

bool model_model1 = false;
bool model_model2 = false;
bool model_model3 = false;
bool model_model4 = false;
bool model_model5 = false;

path_type model_file;
path_type embedding_file;
int hidden_size = 64;
int embedding_size = 32;

int beam_size = 50;
int kbest_size = 50;
int unary_size = 3;

bool binarize_left = false;
bool binarize_right = false;

bool randomize = false;

opt_set_type optimize_options;

bool mix_none_mode = false;
bool mix_average_mode = false;
bool mix_select_mode = false;

path_type output_prefix;
int dump = 0;

int debug = 0;

template <typename Theta, typename Gen>
void learn(const option_set_type& options,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen);

void read_data(const path_type& path_input,
	       tree_set_type& trees);

// mpi-related stuff
template <typename Theta>
void bcast_model(Theta& theta);

void options(int argc, char** argv);

int main(int argc, char** argv)
{
  utils::mpi_world mpi_world(argc, argv);
  
  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();

  try {
    options(argc, argv);
    
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

    if (int(mix_none_mode) + mix_average_mode + mix_select_mode > 1)
      throw std::runtime_error("you can specify only one of mix-{none,average,select}");
    if (int(mix_none_mode) + mix_average_mode + mix_select_mode == 0)
      mix_none_mode = true;

    if (model_file.empty()) {
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5 > 1)
	throw std::runtime_error("either one of --model{1,2,3,4,5}");
      
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5 == 0)
	model_model2 = true;
    } else {
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5)
	throw std::runtime_error("model file is specified via --model, but with --model{1,2,3,4,5}?");
      
      if (! boost::filesystem::exists(model_file))
	throw std::runtime_error("no model file? " + model_file.string());
      
      switch (model_type::model(model_file)) {
      case rnnp::model::MODEL1: model_model1 = true; break;
      case rnnp::model::MODEL2: model_model2 = true; break;
      case rnnp::model::MODEL3: model_model3 = true; break;
      case rnnp::model::MODEL4: model_model4 = true; break;
      case rnnp::model::MODEL5: model_model5 = true; break;
      default:
	throw std::runtime_error("invalid model file");
      }
    }
    
    if (output_file.empty())
      throw std::runtime_error("no output?");

    boost::mt19937 generator;
    generator.seed(utils::random_seed());
    
    option_set_type optimizations(optimize_options.begin(), optimize_options.end());

    if (optimizations.empty())
      optimizations.push_back(option_type());
        
    tree_set_type trees;
    read_data(input_file, trees);
    
    if (mpi_rank == 0 && debug)
      std::cerr << "# of training data: " << trees.size() << std::endl;

    tree_set_type tests;
    if (! test_file.empty())
      read_data(test_file, tests);
    
    if (mpi_rank == 0 && debug && ! tests.empty())
      std::cerr << "# of test data: " << tests.size() << std::endl;
    
    grammar_type grammar(grammar_file);
    
    if (mpi_rank == 0 && debug)
      std::cerr << "binary: " << grammar.binary_size()
		<< " unary: " << grammar.unary_size()
		<< " preterminal: " << grammar.preterminal_size()
		<< " terminals: " << grammar.terminal_.size()
		<< " non-terminals: " << grammar.non_terminal_.size()
		<< " POS: " << grammar.pos_.size()
		<< std::endl;

    signature_type::signature_ptr_type signature(signature_type::create(signature_name));
    
    if (model_model1) {
      if (mpi_rank == 0 && debug)
	std::cerr << "model1" << std::endl;
      
      rnnp::model::Model1 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, theta, generator);
    } else if (model_model2) {
      if (mpi_rank == 0 && debug)
	std::cerr << "model2" << std::endl;
      
      rnnp::model::Model2 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, theta, generator);
    } else if (model_model3) {
      if (mpi_rank == 0 && debug)
	std::cerr << "model3" << std::endl;
      
      rnnp::model::Model3 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, theta, generator);
    } else if (model_model4) {
      if (mpi_rank == 0 && debug)
	std::cerr << "model4" << std::endl;
      
      rnnp::model::Model4 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, theta, generator);
    } else if (model_model5) {
      if (mpi_rank == 0 && debug)
	std::cerr << "model5" << std::endl;
      
      rnnp::model::Model5 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, theta, generator);
    } else
      throw std::runtime_error("no model?");
    
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}

enum {
  model_tag = 1000,
  gradient_tag,
  tree_tag,
  loss_tag,
  evalb_tag,
};

inline
int loop_sleep(bool found, int non_found_iter)
{
  if (! found) {
    boost::thread::yield();
    ++ non_found_iter;
  } else
    non_found_iter = 0;
    
  if (non_found_iter >= 50) {
    struct timespec tm;
    tm.tv_sec = 0;
    tm.tv_nsec = 2000001;
    nanosleep(&tm, NULL);
      
    non_found_iter = 0;
  }
  return non_found_iter;
}

template <typename Theta, typename Optimizer, typename Objective>
struct Task
{
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;

  typedef typename rnnp::model_traits<Theta>::model_type    model_type;
  typedef typename rnnp::model_traits<Theta>::gradient_type gradient_type;

  typedef rnnp::Loss loss_type;

  typedef std::string encoded_type;
  typedef std::vector<char, std::allocator<char> > buffer_type;

  typedef utils::lockfree_list_queue<tree_type, std::allocator<tree_type> >       queue_tree_type;
  typedef utils::lockfree_list_queue<encoded_type, std::allocator<encoded_type> > queue_gradient_type;
  
  Task(const Optimizer& optimizer,
       const Objective& objective,
       const option_type& option,
       const grammar_type& grammar,
       const signature_type& signature,
       model_type& theta,
       queue_tree_type& tree_mapper,
       queue_gradient_type& gradient_mapper,
       queue_gradient_type& gradient_reducer)
    : optimizer_(optimizer),
      objective_(objective),
      option_(option),
      grammar_(grammar),
      signature_(signature),
      theta_(theta),
      tree_mapper_(tree_mapper),
      gradient_mapper_(gradient_mapper),
      gradient_reducer_(gradient_reducer),
      parser_(beam_size, unary_size),
      parser_oracle_(beam_size, unary_size, binarize_left),
      gradient_(theta),
      gradient_batch_(theta)
  { }
  
  Optimizer   optimizer_;
  Objective   objective_;
  option_type option_;
  
  const grammar_type&   grammar_;
  const signature_type& signature_;
  model_type& theta_;
  
  queue_tree_type&     tree_mapper_;
  queue_gradient_type& gradient_mapper_;
  queue_gradient_type& gradient_reducer_;

  rnnp::Parser       parser_;
  rnnp::ParserOracle parser_oracle_;

  loss_type loss_;
  size_type instances_;
  size_type parsed_;
  size_type updated_;
  
  buffer_type   buffer_;
  gradient_type gradient_;
  gradient_type gradient_batch_;
  
  void operator()()
  {
    clear();
    
    const size_type batch_size = option_.batch_;

    signature_type::signature_ptr_type signature(signature_.clone());

    rnnp::Parser::derivation_set_type candidates;
    rnnp::Parser::derivation_set_type oracles;

    rnnp::Derivation derivation;

    tree_type    tree;
    encoded_type encoded;

    size_type batch_learn = 0;
    
    bool merge_finished = false;
    bool learn_finished = false;

    int non_found_iter = 0;
    
    while (! merge_finished || ! learn_finished) {
      bool found = false;
      
      if (! merge_finished && ! batch_learn)
	while (gradient_reducer_.pop_swap(encoded, true)) {
	  if (encoded.empty()) 
	    merge_finished = true;
	  else {
	    boost::iostreams::filtering_istream is;
	    is.push(codec::lz4_decompressor());
	    is.push(boost::iostreams::array_source(&(*encoded.begin()), encoded.size()));
	    
	    is >> gradient_;
	    
	    is.reset();
	    encoded.clear();
	    
	    optimizer_(theta_, gradient_, option_);
	  }

	  found = true;
	}
      
      if (! learn_finished && tree_mapper_.pop_swap(tree, true)) {
	found = true;

	if (tree.empty())
	  learn_finished = true;
	else  {
	  parser_oracle_(tree, grammar_, *signature, theta_, kbest_size, oracles);
	    
	  parser_(parser_oracle_.oracle_.sentence_, grammar_, *signature, theta_, kbest_size, candidates);
	  
	  parsed_ += (! candidates.empty());
	  ++ instances_;
	  ++ batch_learn;
	  
	  loss_ += objective_(theta_, parser_, parser_oracle_, option_, gradient_batch_);
	}
	
	if (batch_learn == batch_size || (learn_finished && batch_learn)) {
	  loss_ += objective_(gradient_batch_);
	  
	  updated_ += gradient_batch_.count_;

	  optimizer_(theta_, gradient_batch_, option_);
	  
	  // encoding..
	  buffer_.clear();
	  
	  boost::iostreams::filtering_ostream os;
	  os.push(codec::lz4_compressor());
	  os.push(boost::iostreams::back_inserter(buffer_));
	  
	  os << gradient_batch_;
	  
	  os.reset();
	  
	  gradient_batch_.clear();
	  
	  gradient_mapper_.push(encoded_type(buffer_.begin(), buffer_.end()));
	  
	  batch_learn = 0;
	}
	
	if (learn_finished)
	  gradient_mapper_.push(encoded_type());
      }
      
      non_found_iter = loop_sleep(found, non_found_iter);
    }
  }

  inline
  int loop_sleep(bool found, int non_found_iter)
  {
    if (! found) {
      boost::thread::yield();
      ++ non_found_iter;
    } else
      non_found_iter = 0;
    
    if (non_found_iter >= 50) {
      struct timespec tm;
      tm.tv_sec = 0;
      tm.tv_nsec = 2000001;
      nanosleep(&tm, NULL);
      
      non_found_iter = 0;
    }
    return non_found_iter;
  }

  void clear()
  {
    loss_      = loss_type();
    instances_ = 0;
    parsed_    = 0;
    updated_   = 0;
  }
};


template <typename Theta>
void bcast_model(int rank, Theta& theta)
{
  if (MPI::COMM_WORLD.Get_rank() == rank) {
    boost::iostreams::filtering_ostream os;
    os.push(codec::lz4_compressor());
    os.push(utils::mpi_device_bcast_sink(rank, 1024 * 1024));
    
    os << theta;
  } else {
    boost::iostreams::filtering_istream is;
    is.push(codec::lz4_decompressor());
    is.push(utils::mpi_device_bcast_source(rank, 1024 * 1024));
    
    is >> theta;
  }
}

template <typename Theta>
void bcast_model(Theta& theta)
{
  bcast_model(0, theta);
}

template <typename Theta>
void merge_model(int root, Theta& theta)
{
  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();
  
  if (mpi_rank == root) {
    Theta theta_reduced(theta);
    
    for (int rank = 0; rank != mpi_size; ++ rank)
      if (rank != root) {
	boost::iostreams::filtering_istream is;
	is.push(codec::lz4_decompressor());
	is.push(utils::mpi_device_source(rank, model_tag, 1024 * 1024));
	
	is >> theta_reduced;
	theta += theta_reduced;
      }
  } else {
    boost::iostreams::filtering_ostream os;
    os.push(codec::lz4_compressor());
    os.push(utils::mpi_device_sink(root, model_tag, 1024 * 1024));
    
    os << theta;
  }
}

template <typename Theta>
void merge_model(Theta& theta)
{
  merge_model(0, theta);
}

template <typename Theta>
void average_model(Theta& theta)
{
  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();
  
  merge_model(theta);
  
  if (mpi_rank == 0)
    theta *= 1.0 / mpi_size;
  
  bcast_model(theta);
}

template <typename Theta>
void select_model(Theta& theta)
{
  typedef std::vector<double, std::allocator<double> > buffer_type;  

  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();

  const double l1 = theta.l1();
  
  buffer_type buffer_send(mpi_size, 0.0);
  buffer_type buffer_recv(mpi_size, 0.0);
  
  buffer_send[mpi_rank] = l1;
  buffer_recv[mpi_rank] = l1;
  
  MPI::COMM_WORLD.Reduce(&(*buffer_send.begin()), &(*buffer_recv.begin()), mpi_size, utils::mpi_traits<double>::data_type(), MPI::MAX, 0);
  
  int rank_min = (std::min_element(buffer_recv.begin(), buffer_recv.end()) - buffer_recv.begin());
  
  MPI::COMM_WORLD.Bcast(&rank_min, 1, utils::mpi_traits<int>::data_type(), 0);
  
  bcast_model(rank_min, theta);
}

template <typename Theta>
struct OutputModel
{
  typedef std::pair<Theta, path_type> model_path_type;
  typedef utils::lockfree_list_queue<model_path_type, std::allocator<model_path_type> > queue_type;
  
  OutputModel(queue_type& queue) : queue_(queue) {}

  void operator()()
  {
    model_path_type theta;
    
    for (;;) {
      queue_.pop_swap(theta);
      
      if (theta.second.empty()) break;
      
      theta.first.write(theta.second);
    }
  }
  
  queue_type& queue_;
};

template <typename Theta>
inline
void swap(typename OutputModel<Theta>::model_path_type& x, typename OutputModel<Theta>::model_path_type& y)
{
  x.first.swap(y.first);
  x.second.swap(y.second);
}

template <typename Theta, typename Task>
struct Test
{
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;
  
  typedef rnnp::Evalb       evalb_type;
  typedef rnnp::EvalbScorer scorer_type;
  
  typedef utils::lockfree_list_queue<tree_type, std::allocator<tree_type> > queue_type;

  Test(Task& task,
       evalb_type& evalb,
       queue_type& queue)
    : task_(task),
      evalb_(evalb),
      queue_(queue) {}
  
  void operator()()
  {
    signature_type::signature_ptr_type signature(task_.signature_.clone());
    
    rnnp::Parser::derivation_set_type candidates;
    
    evalb_.clear();
    
    tree_type tree;
    
    for (;;) {
      queue_.pop_swap(tree);
      
      if (tree.empty()) break;
      
      task_.parser_(tree.leaf(), task_.grammar_, *signature, task_.theta_, kbest_size, candidates);

      if (candidates.empty()) continue;
      
      scorer_.assign(tree);
      evalb_ += scorer_(candidates.front());
    }
  }
  
  Task& task_;
  evalb_type& evalb_;
  queue_type& queue_;
  
  scorer_type scorer_;
};

template <typename Theta, typename Optimizer, typename Objective, typename Gen>
void learn_root(const Optimizer& optimizer,
		const Objective& objective,
		const option_type& option,
		const tree_set_type& trees,
		const tree_set_type& tests,
		const grammar_type& grammar,
		const signature_type& signature,
		Theta& theta_ret,
		Gen& gen)
{
  typedef OutputModel<Theta> output_model_type;

  typedef Task<Theta, Optimizer, Objective> task_type;
  
  typedef typename task_type::queue_tree_type     queue_tree_type;
  typedef typename task_type::queue_gradient_type queue_gradient_type;
  
  typedef typename task_type::loss_type    loss_type;
  typedef typename task_type::size_type    size_type;
  typedef typename task_type::encoded_type buffer_type;
  
  typedef boost::shared_ptr<buffer_type> buffer_ptr_type;
  typedef std::deque<buffer_ptr_type, std::allocator<buffer_ptr_type> >  buffer_set_type;
  typedef std::vector<buffer_set_type, std::allocator<buffer_set_type> > buffer_map_type;
  
  typedef utils::mpi_ostream tree_ostream_type;
  typedef utils::mpi_istream tree_istream_type;
  
  typedef utils::mpi_ostream_simple gradient_ostream_type;
  typedef utils::mpi_istream_simple gradient_istream_type;
  
  typedef boost::shared_ptr<tree_ostream_type> tree_ostream_ptr_type;
  typedef boost::shared_ptr<tree_istream_type> tree_istream_ptr_type;
  
  typedef boost::shared_ptr<gradient_ostream_type> gradient_ostream_ptr_type;
  typedef boost::shared_ptr<gradient_istream_type> gradient_istream_ptr_type;

  typedef std::vector<tree_ostream_ptr_type, std::allocator<tree_ostream_ptr_type> > tree_ostream_ptr_set_type;
  typedef std::vector<tree_istream_ptr_type, std::allocator<tree_istream_ptr_type> > tree_istream_ptr_set_type;
  
  typedef std::vector<gradient_ostream_ptr_type, std::allocator<gradient_ostream_ptr_type> > gradient_ostream_ptr_set_type;
  typedef std::vector<gradient_istream_ptr_type, std::allocator<gradient_istream_ptr_type> > gradient_istream_ptr_set_type;
  
  typedef std::vector<size_type, std::allocator<size_type> > working_set_type;

  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();

  bool perform_testing = ! tests.empty();
  MPI::COMM_WORLD.Bcast(&perform_testing, 1, utils::mpi_traits<bool>::data_type(), 0);

  Theta theta = theta_ret;
  
  working_set_type working(trees.size());
  for (size_type i = 0; i != trees.size(); ++ i)
    working[i] = i;
  
  typename output_model_type::queue_type queue_dumper;
  
  std::auto_ptr<boost::thread> dumper(new boost::thread(output_model_type(queue_dumper)));

  queue_tree_type     tree_mapper(1);
  queue_gradient_type gradient_mapper;
  queue_gradient_type gradient_reducer;
  
  task_type task(optimizer,
		 objective,
		 option,
		 grammar,
		 signature,
		 theta,
		 tree_mapper,
		 gradient_mapper,
		 gradient_reducer);
  
  std::string line;
  
  buffer_type     buffer;
  buffer_map_type buffers(mpi_size);
  
  tree_ostream_ptr_set_type tree_ostream(mpi_size);
  
  gradient_ostream_ptr_set_type gradient_ostream(mpi_size);
  gradient_istream_ptr_set_type gradient_istream(mpi_size);
  
  double evalb_max = 0;
  int zero_iter = 0;
  
  for (int t = 0; t < option.iteration_; ++ t) {
    if (debug)
      std::cerr << "iteration: " << (t + 1) << std::endl;
    
    // prepare iostreams...
    for (int rank = 0; rank != mpi_size; ++ rank)
      if (rank != mpi_rank) {
	tree_ostream[rank].reset(new tree_ostream_type(rank, tree_tag));
	
	gradient_ostream[rank].reset(new gradient_ostream_type(rank, gradient_tag));
	gradient_istream[rank].reset(new gradient_istream_type(rank, gradient_tag));
      }
    
    // create thread!
    boost::thread worker(boost::ref(task));
    
    std::auto_ptr<boost::progress_display> progress(debug
						    ? new boost::progress_display(working.size(), std::cerr, "", "", "")
						    : 0);
    
    utils::resource start;

    bool gradient_mapper_finished = false; // for gradients
    bool gradient_reducer_finished = false; // for gradients
    bool tree_finished = false; // for trees
    
    size_type id = 0;
    
    int non_found_iter = 0;
    for (;;) {
      bool found = false;

      // mapping of trees...
      for (int rank = 1; rank != mpi_size && id != working.size(); ++ rank)
	if (tree_ostream[rank]->test()) {
	  // encode tree as a line
	  
	  line.clear();
	  
	  boost::iostreams::filtering_ostream os;
	  os.push(boost::iostreams::back_inserter(line));
	  os << trees[working[id]];
	  os.reset();
	  
	  tree_ostream[rank]->write(line);
	  
	  if (progress.get())
	    ++ (*progress);
	  
	  ++ id;
	  found = true;
	}
      
      if (id != working.size() && tree_mapper.empty()) {
	tree_mapper.push(trees[working[id]]);
	
	if (progress.get())
	  ++ (*progress);
	
	++ id;
	found = true;
      }
      
      if (! tree_finished && id == working.size()) {
	tree_mapper.push(tree_type());
	tree_finished = true;
      }
      
      // terminate tree mapping
      if (tree_finished)
	for (int rank = 1; rank != mpi_size; ++ rank)
	  if (tree_ostream[rank] && tree_ostream[rank]->test()) {
	    if (! tree_ostream[rank]->terminated())
	      tree_ostream[rank]->terminate();
	    else
	      tree_ostream[rank].reset();
	    
	    found = true;
	  }
      
      // reduce gradients
      for (int rank = 0; rank != mpi_size; ++ rank)
	if (rank != mpi_rank && gradient_istream[rank] && gradient_istream[rank]->test()) {
	  if (gradient_istream[rank]->read(buffer))
	    gradient_reducer.push_swap(buffer);
	  else
	    gradient_istream[rank].reset();
	  
	  buffer.clear();
	  found = true;
	}
      
      // check termination...
      if (! gradient_reducer_finished
	  && std::count(gradient_istream.begin(), gradient_istream.end(), gradient_istream_ptr_type()) == mpi_size) {
	gradient_reducer.push(buffer_type());
	gradient_reducer_finished = true;
      }
      
      // bcast...
      // first, get the encoded buffer from mapper
      if (! gradient_mapper_finished && gradient_mapper.pop_swap(buffer, true)) {
	buffer_ptr_type buffer_ptr;
	
	if (! buffer.empty()) {
	  buffer_ptr.reset(new buffer_type());
	  buffer_ptr->swap(buffer);
	  buffer.clear();
	} else
	  gradient_mapper_finished = true;
	
	for (int rank = 0; rank != mpi_size; ++ rank) 
	  if (rank != mpi_rank)
	    buffers[rank].push_back(buffer_ptr);
	
	found = true;
      }
      
      // second, bcast...
      for (int rank = 0; rank != mpi_size; ++ rank)
	if (rank != mpi_rank && gradient_ostream[rank] && gradient_ostream[rank]->test() && ! buffers[rank].empty()) {
	  if (! buffers[rank].front()) {
	    // termination!
	    if (! gradient_ostream[rank]->terminated())
	      gradient_ostream[rank]->terminate();
	    else {
	      gradient_ostream[rank].reset();
	      buffers[rank].erase(buffers[rank].begin());
	    }
	  } else {
	    gradient_ostream[rank]->write(*(buffers[rank].front()));
	    buffers[rank].erase(buffers[rank].begin());
	  }
	  
	  found = true;
	}
      
      // termination condition
      if (tree_finished && gradient_reducer_finished && gradient_mapper_finished
	  && std::count(tree_ostream.begin(), tree_ostream.end(), tree_ostream_ptr_type()) == mpi_size
	  && std::count(gradient_istream.begin(), gradient_istream.end(), gradient_istream_ptr_type()) == mpi_size
	  && std::count(gradient_ostream.begin(), gradient_ostream.end(), gradient_ostream_ptr_type()) == mpi_size) break;
      
      non_found_iter = loop_sleep(found, non_found_iter);
    }
    
    worker.join();
    
    utils::resource end;
    
    loss_type loss      = task.loss_;
    size_type instances = task.instances_;
    size_type parsed    = task.parsed_;
    size_type updated   = task.updated_;
    
    for (int rank = 1; rank != mpi_size; ++ rank) {
      loss_type l;
      size_type i;
      size_type p;
      size_type u;
      
      boost::iostreams::filtering_istream is;
      is.push(utils::mpi_device_source(rank, loss_tag, 4096));
      is.read((char*) &l, sizeof(loss_type));
      is.read((char*) &i, sizeof(size_type));
      is.read((char*) &p, sizeof(size_type));
      is.read((char*) &u, sizeof(size_type));
      
      loss      += l;
      instances += i;
      parsed    += p;
      updated   += u;
    }
    
    if (debug)
      std::cerr << "loss: " << static_cast<double>(loss) << std::endl
		<< "instances: " << instances << std::endl
		<< "parsed: " << parsed << std::endl
		<< "updated: " << updated << std::endl;
    
    if (debug)
      std::cerr << "cpu time:    " << end.cpu_time() - start.cpu_time() << std::endl
		<< "user time:   " << end.user_time() - start.user_time() << std::endl;
    
    // shuffle trees!
    {
      boost::random_number_generator<boost::mt19937> rng(gen);
      
      std::random_shuffle(working.begin(), working.end(), rng);
    }
    
    // mixing
    if (mix_average_mode)
      average_model(theta);
    else if (mix_select_mode)
      select_model(theta);
    else
      bcast_model(theta);
    
    if (dump > 0 && !((t + 1) % dump))
      queue_dumper.push(std::make_pair(theta,
				       output_prefix.string() + "." + utils::lexical_cast<std::string>(t + 1)));

    if (perform_testing) {
      typedef Test<Theta, task_type> test_type;
      
      typedef typename test_type::queue_type queue_type;
      typedef typename test_type::evalb_type evalb_type;
      
      if (debug)
	std::cerr << "testing: " << (t + 1) << std::endl;
      
      queue_type queue(1);
      evalb_type evalb;
      
      std::auto_ptr<boost::progress_display> progress(debug
						      ? new boost::progress_display(tests.size(), std::cerr, "", "", "")
						      : 0);
      
      std::auto_ptr<boost::thread> worker(new boost::thread(test_type(task, evalb, queue)));

      // prepare iostreams...
      for (int rank = 0; rank != mpi_size; ++ rank)
	if (rank != mpi_rank)
	  tree_ostream[rank].reset(new tree_ostream_type(rank, tree_tag));
      
      utils::resource start;
      
      bool tree_finished = false; // for trees
      
      size_type id = 0;
      
      int non_found_iter = 0;
      for (;;) {
	bool found = false;
	
	// mapping of trees...
	for (int rank = 1; rank != mpi_size && id != tests.size(); ++ rank)
	  if (tree_ostream[rank]->test()) {
	    // encode tree as a line
	    
	    line.clear();
	    
	    boost::iostreams::filtering_ostream os;
	    os.push(boost::iostreams::back_inserter(line));
	    os << tests[id];
	    os.reset();
	    
	    tree_ostream[rank]->write(line);
	    
	    if (progress.get())
	      ++ (*progress);
	    
	    ++ id;
	    found = true;
	  }
	
	if (id != tests.size() && tree_mapper.empty()) {
	  queue.push(tests[id]);
	  
	  if (progress.get())
	    ++ (*progress);
	  
	  ++ id;
	  found = true;
	}
	
	if (! tree_finished && id == tests.size()) {
	  queue.push(tree_type());
	  tree_finished = true;
	}
	
	// terminate tree mapping
	if (tree_finished)
	  for (int rank = 1; rank != mpi_size; ++ rank)
	    if (tree_ostream[rank] && tree_ostream[rank]->test()) {
	      if (! tree_ostream[rank]->terminated())
		tree_ostream[rank]->terminate();
	      else
		tree_ostream[rank].reset();
	      
	      found = true;
	    }
	
	// termination condition
	if (tree_finished && std::count(tree_ostream.begin(), tree_ostream.end(), tree_ostream_ptr_type()) == mpi_size) break;
	
	non_found_iter = loop_sleep(found, non_found_iter);
      }
      
      worker->join();
      
      utils::resource end;
      
      for (int rank = 1; rank != mpi_size; ++ rank) {
	evalb_type e;
	
	boost::iostreams::filtering_istream is;
	is.push(utils::mpi_device_source(rank, evalb_tag, 4096));
	
	is >> e;
	
	evalb += e;
      }
      
      const double evalb_curr = evalb();

      if (debug)
	std::cerr << "EVALB: " << evalb_curr << std::endl
		  << "test cpu time:    " << end.cpu_time() - start.cpu_time() << std::endl
		  << "test user time:   " << end.user_time() - start.user_time() << std::endl;
      
      if (evalb_curr > evalb_max) {
	evalb_max = evalb_curr;
	theta_ret = theta;
      }
    }

    MPI::COMM_WORLD.Bcast(&updated, 1, utils::mpi_traits<size_type>::data_type(), 0);
    
    if (! updated)
      ++ zero_iter;
    else
      zero_iter = 0;
    
    if (zero_iter >= 2) break;
  }
  
  queue_dumper.push(typename output_model_type::model_path_type());
  dumper->join();
  
  if (! perform_testing)
    theta_ret = theta;
  else
    bcast_model(theta_ret);
}

template <typename Theta, typename Optimizer, typename Objective>
void learn_others(const Optimizer& optimizer,
		  const Objective& objective,
		  const option_type& option,
		  const grammar_type& grammar,
		  const signature_type& signature,
		  Theta& theta_ret)
{
  typedef Task<Theta, Optimizer, Objective> task_type;
  
  typedef typename task_type::queue_tree_type     queue_tree_type;
  typedef typename task_type::queue_gradient_type queue_gradient_type;
  
  typedef typename task_type::loss_type    loss_type;
  typedef typename task_type::size_type    size_type;
  typedef typename task_type::encoded_type buffer_type;
  
  typedef boost::shared_ptr<buffer_type> buffer_ptr_type;
  typedef std::deque<buffer_ptr_type, std::allocator<buffer_ptr_type> >  buffer_set_type;
  typedef std::vector<buffer_set_type, std::allocator<buffer_set_type> > buffer_map_type;
  
  typedef utils::mpi_ostream tree_ostream_type;
  typedef utils::mpi_istream tree_istream_type;
  
  typedef utils::mpi_ostream_simple gradient_ostream_type;
  typedef utils::mpi_istream_simple gradient_istream_type;
  
  typedef boost::shared_ptr<tree_ostream_type> tree_ostream_ptr_type;
  typedef boost::shared_ptr<tree_istream_type> tree_istream_ptr_type;

  typedef boost::shared_ptr<gradient_ostream_type> gradient_ostream_ptr_type;
  typedef boost::shared_ptr<gradient_istream_type> gradient_istream_ptr_type;

  typedef std::vector<tree_ostream_ptr_type, std::allocator<tree_ostream_ptr_type> > tree_ostream_ptr_set_type;
  typedef std::vector<tree_istream_ptr_type, std::allocator<tree_istream_ptr_type> > tree_istream_ptr_set_type;
  
  typedef std::vector<gradient_ostream_ptr_type, std::allocator<gradient_ostream_ptr_type> > gradient_ostream_ptr_set_type;
  typedef std::vector<gradient_istream_ptr_type, std::allocator<gradient_istream_ptr_type> > gradient_istream_ptr_set_type;
  
  const int mpi_rank = MPI::COMM_WORLD.Get_rank();
  const int mpi_size = MPI::COMM_WORLD.Get_size();

  bool perform_testing = false;
  MPI::COMM_WORLD.Bcast(&perform_testing, 1, utils::mpi_traits<bool>::data_type(), 0);
  
  Theta theta = theta_ret;
  
  queue_tree_type     tree_mapper(1);
  queue_gradient_type gradient_mapper;
  queue_gradient_type gradient_reducer;
  
  task_type task(optimizer,
		 objective,
		 option,
		 grammar,
		 signature,
		 theta,
		 tree_mapper,
		 gradient_mapper,
		 gradient_reducer);
  
  std::string line;
  tree_type tree;
  
  buffer_type     buffer;
  buffer_map_type buffers(mpi_size);
  
  gradient_ostream_ptr_set_type gradient_ostream(mpi_size);
  gradient_istream_ptr_set_type gradient_istream(mpi_size);
  
  int zero_iter = 0;
  
  for (int t = 0; t < option.iteration_; ++ t) {
    tree_istream_ptr_type tree_istream(new tree_istream_type(0, tree_tag));
    
    // prepare iostreams...
    for (int rank = 0; rank != mpi_size; ++ rank)
      if (rank != mpi_rank) {
	gradient_ostream[rank].reset(new gradient_ostream_type(rank, gradient_tag));
	gradient_istream[rank].reset(new gradient_istream_type(rank, gradient_tag));
      }

    // create thread!
    boost::thread worker(boost::ref(task));
    
    bool gradient_mapper_finished = false; // for gradients
    bool gradient_reducer_finished = false; // for gradients
    
    int non_found_iter = 0;
    for (;;) {
      bool found = false;
      
      // read trees mapped from root
      if (tree_istream && tree_istream->test() && tree_mapper.empty()) {
	if (tree_istream->read(line)) {
	  tree.assign(line);

	  tree_mapper.push_swap(tree);
	} else {
	  tree_istream.reset();
	  
	  tree_mapper.push(tree_type());
	}
	
	found = true;
      }
      
      // reduce gradients
      for (int rank = 0; rank != mpi_size; ++ rank)
	if (rank != mpi_rank && gradient_istream[rank] && gradient_istream[rank]->test()) {
	  if (gradient_istream[rank]->read(buffer))
	    gradient_reducer.push_swap(buffer);
	  else
	    gradient_istream[rank].reset();
	  
	  buffer.clear();
	  found = true;
	}
      
      // check termination...
      if (! gradient_reducer_finished
	  && std::count(gradient_istream.begin(), gradient_istream.end(), gradient_istream_ptr_type()) == mpi_size) {
	gradient_reducer.push(buffer_type());
	gradient_reducer_finished = true;
      }
      
      // bcast gradients
      // first, get the encoded buffer from mapper
      if (! gradient_mapper_finished && gradient_mapper.pop_swap(buffer, true)) {
	buffer_ptr_type buffer_ptr;
	
	if (! buffer.empty()) {
	  buffer_ptr.reset(new buffer_type());
	  buffer_ptr->swap(buffer);
	  buffer.clear();
	} else
	  gradient_mapper_finished = true;
	
	for (int rank = 0; rank != mpi_size; ++ rank) 
	  if (rank != mpi_rank)
	    buffers[rank].push_back(buffer_ptr);
	
	found = true;
      }
      
      // second, bcast...
      for (int rank = 0; rank != mpi_size; ++ rank)
	if (rank != mpi_rank && gradient_ostream[rank] && gradient_ostream[rank]->test() && ! buffers[rank].empty()) {
	  if (! buffers[rank].front()) {
	    // termination!
	    if (! gradient_ostream[rank]->terminated())
	      gradient_ostream[rank]->terminate();
	    else {
	      gradient_ostream[rank].reset();
	      buffers[rank].erase(buffers[rank].begin());
	    }
	  } else {
	    gradient_ostream[rank]->write(*(buffers[rank].front()));
	    buffers[rank].erase(buffers[rank].begin());
	  }
	  
	  found = true;
	}
      
      // termination condition
      if (! tree_istream
	  && gradient_reducer_finished
	  && gradient_mapper_finished
	  && std::count(gradient_istream.begin(), gradient_istream.end(), gradient_istream_ptr_type()) == mpi_size
	  && std::count(gradient_ostream.begin(), gradient_ostream.end(), gradient_ostream_ptr_type()) == mpi_size) break;
      
      non_found_iter = loop_sleep(found, non_found_iter);
    }
    
    worker.join();
    
    // send additional information...
    // loss and parsed...
    {
      boost::iostreams::filtering_ostream os;
      os.push(utils::mpi_device_sink(0, loss_tag, 4096));
      os.write((char*) &task.loss_, sizeof(task.loss_));
      os.write((char*) &task.instances_, sizeof(task.instances_));
      os.write((char*) &task.parsed_, sizeof(task.parsed_));
      os.write((char*) &task.updated_, sizeof(task.updated_));
    }
    
    // mixing
    if (mix_average_mode)
      average_model(theta);
    else if (mix_select_mode)
      select_model(theta);
    else
      bcast_model(theta);

    if (perform_testing) {
      typedef Test<Theta, task_type> test_type;  
      
      typedef typename test_type::queue_type queue_type;
      typedef typename test_type::evalb_type evalb_type;
      
      queue_type queue(1);
      evalb_type evalb;
      
      std::auto_ptr<boost::thread> worker(new boost::thread(test_type(task, evalb, queue)));
      
      tree_istream_ptr_type tree_istream(new tree_istream_type(0, tree_tag));

      int non_found_iter = 0;
      for (;;) {
	bool found = false;
	
	// read trees mapped from root
	if (tree_istream && tree_istream->test() && queue.empty()) {
	  if (tree_istream->read(line)) {
	    tree.assign(line);
	    
	    queue.push_swap(tree);
	  } else {
	    tree_istream.reset();
	    
	    queue.push(tree_type());
	  }
	  
	  found = true;
	}

	if (! tree_istream) break;
	
	non_found_iter = loop_sleep(found, non_found_iter);
      }
      
      worker->join();
      
      boost::iostreams::filtering_ostream os;
      os.push(utils::mpi_device_sink(0, evalb_tag, 4096));
      os << evalb;
    }
    
    size_type updated = 0;
    
    MPI::COMM_WORLD.Bcast(&updated, 1, utils::mpi_traits<size_type>::data_type(), 0);
    
    if (! updated)
      ++ zero_iter;
    else
      zero_iter = 0;
    
    if (zero_iter >= 2) break;
  }

  if (! perform_testing)
    theta_ret = theta;
  else
    bcast_model(theta_ret);
}


template <typename Theta, typename Optimizer, typename Objective, typename Gen>
void learn(const Optimizer& optimizer,
	   const Objective& objective,
	   const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen);

template <typename Theta, typename Optimizer, typename Gen>
void learn(const Optimizer& optimizer,
	   const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen)
{
  
  if (option.margin_derivation())
    learn(optimizer, rnnp::objective::MarginDerivation(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.margin_evalb())
    learn(optimizer, rnnp::objective::MarginEvalb(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.margin_early())
    learn(optimizer, rnnp::objective::MarginEarly(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.margin_late())
    learn(optimizer, rnnp::objective::MarginLate(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.margin_max())
    learn(optimizer, rnnp::objective::MarginMax(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.violation_early())
    learn(optimizer, rnnp::objective::ViolationEarly(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.violation_late())
    learn(optimizer, rnnp::objective::ViolationLate(), option, trees, tests, grammar, signature, theta, gen);
  else if (option.violation_max())
    learn(optimizer, rnnp::objective::ViolationMax(), option, trees, tests, grammar, signature, theta, gen);
  else
    throw std::runtime_error("unsupported objective");
}

template <typename Theta, typename Gen>
void learn(const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen)
{
  if (MPI::COMM_WORLD.Get_rank() == 0 && debug)
    std::cerr << "learning: " << option << std::endl;

  if (option.optimize_adagrad())
    learn(rnnp::optimize::AdaGrad<Theta>(theta, option.lambda_, option.eta0_), option, trees, tests, grammar, signature, theta, gen);
  else if (option.optimize_adadec())
    learn(rnnp::optimize::AdaDec<Theta>(theta, option.lambda_, option.eta0_), option, trees, tests, grammar, signature, theta, gen);
  else if (option.optimize_adadelta())
    learn(rnnp::optimize::AdaDelta<Theta>(theta, option.lambda_, option.eta0_), option, trees, tests, grammar, signature, theta, gen);
  else if (option.optimize_sgd())
    learn(rnnp::optimize::SGD<Theta>(theta, option.lambda_, option.eta0_), option, trees, tests, grammar, signature, theta, gen);
  else
    throw std::runtime_error("unknown optimizer");
}

template <typename Theta, typename Gen>
void learn(const option_set_type& optimizations,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen)
{
  if (! model_file.empty())
    theta.read(model_file);
  else {
    if (randomize)
      theta.random(gen);
    
    if (! embedding_file.empty())
      theta.embedding(embedding_file);
    
    bcast_model(theta);
  }
  
  if (MPI::COMM_WORLD.Get_rank() == 0 && debug) {
    const size_t terminals = std::count(theta.vocab_terminal_.begin(), theta.vocab_terminal_.end(), true);
    const size_t non_terminals = (theta.vocab_category_.size()
				  - std::count(theta.vocab_category_.begin(), theta.vocab_category_.end(),
					       model_type::symbol_type()));
    
    std::cerr << "terminals: " << terminals
	      << " non-terminals: " << non_terminals
	      << std::endl;
  }

  int iter = 0;
  
  option_set_type::const_iterator oiter_end = optimizations.end();
  for (option_set_type::const_iterator oiter = optimizations.begin(); oiter != oiter_end; ++ oiter, ++ iter) {
    output_prefix = output_file.string() + "." + utils::lexical_cast<std::string>(iter);
    
    learn(*oiter, trees, tests, grammar, signature, theta, gen);
  }

  if (MPI::COMM_WORLD.Get_rank() == 0)
    theta.write(output_file);
}


template <typename Theta, typename Optimizer, typename Objective, typename Gen>
void learn(const Optimizer& optimizer,
	   const Objective& objective,
	   const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   Theta& theta,
	   Gen& gen)
{
  if (MPI::COMM_WORLD.Get_rank() == 0)
    learn_root(optimizer, objective, option, trees, tests, grammar, signature, theta, gen);
  else
    learn_others(optimizer, objective, option, grammar, signature, theta);
}

void read_data(const path_type& path,
	       tree_set_type& trees)
{
  if (MPI::COMM_WORLD.Get_rank() != 0) return;
 
  utils::compress_istream is(path, 1024 * 1024);
  
  tree_type tree;
  while (is >> tree)
    if (! tree.empty())
      trees.push_back(tree);
}

void options(int argc, char** argv)
{
  namespace po = boost::program_options;
 
  po::options_description opts_config("configuration options");
  opts_config.add_options()
    ("input",     po::value<path_type>(&input_file)->default_value(input_file), "input file")
    ("test",      po::value<path_type>(&test_file),                             "test file")
    ("output",    po::value<path_type>(&output_file),                           "output file")
    
    ("grammar",    po::value<path_type>(&grammar_file),                                    "grammar file")
    ("signature",  po::value<std::string>(&signature_name)->default_value(signature_name), "language specific signature")
    
    ("model1",    po::bool_switch(&model_model1), "parsing by model1")
    ("model2",    po::bool_switch(&model_model2), "parsing by model2 (default)")
    ("model3",    po::bool_switch(&model_model3), "parsing by model3")
    ("model4",    po::bool_switch(&model_model4), "parsing by model4")
    ("model5",    po::bool_switch(&model_model5), "parsing by model5")

    ("model",     po::value<path_type>(&model_file),                              "model file")
    ("hidden",    po::value<int>(&hidden_size)->default_value(hidden_size),       "hidden dimension")
    ("embedding", po::value<int>(&embedding_size)->default_value(embedding_size), "embedding dimension")
    
    ("beam",      po::value<int>(&beam_size)->default_value(beam_size),           "beam size")
    ("kbest",     po::value<int>(&kbest_size)->default_value(kbest_size),         "kbest size")
    ("unary",     po::value<int>(&unary_size)->default_value(unary_size),         "unary size")
    
    ("binarize-left",  po::bool_switch(&binarize_left),  "left recursive (or left heavy) binarization (default)")
    ("binarize-right", po::bool_switch(&binarize_right), "right recursive (or right heavy) binarization")
    
    ("randomize",      po::bool_switch(&randomize),           "randomize model parameters")
    ("word-embedding", po::value<path_type>(&embedding_file), "word embedding file")
    
    ("learn", po::value<opt_set_type>(&optimize_options)->composing(), "learning option(s)")
    
    ("mix-none",    po::bool_switch(&mix_none_mode),    "no mixing")
    ("mix-average", po::bool_switch(&mix_average_mode), "mixing weights by averaging")
    ("mix-select",  po::bool_switch(&mix_select_mode),  "select weights by L1")

    ("dump", po::value<int>(&dump)->default_value(dump), "output model file during training");
  
  po::options_description opts_command("command line options");
  opts_command.add_options()
    ("config",  po::value<path_type>(),                    "configuration file")    
    
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
