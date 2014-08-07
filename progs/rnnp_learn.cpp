//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <stdexcept>
#include <iostream>
#include <memory>

#include <rnnp/evalb.hpp>
#include <rnnp/tree.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/signature.hpp>
#include <rnnp/feature_set.hpp>
#include <rnnp/model_traits.hpp>
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
#include <rnnp/optimize/adagradrda.hpp>
#include <rnnp/optimize/adadec.hpp>
#include <rnnp/optimize/adadelta.hpp>
#include <rnnp/optimize/sgd.hpp>

#include <rnnp/loss.hpp>

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

typedef rnnp::Sentence   sentence_type;
typedef rnnp::Tree       tree_type;
typedef rnnp::Grammar    grammar_type;
typedef rnnp::Signature  signature_type;
typedef rnnp::FeatureSet feature_set_type;
typedef rnnp::Model      model_type;

typedef rnnp::LearnOption option_type;

typedef boost::filesystem::path path_type;
typedef std::vector<std::string, std::allocator<std::string> > feat_set_type;

typedef std::vector<std::string, std::allocator<std::string> > opt_set_type;
typedef std::vector<option_type, std::allocator<option_type> > option_set_type;

typedef utils::chunk_vector<tree_type, 4096 / sizeof(tree_type), std::allocator<tree_type> > tree_set_type;

path_type input_file = "-";
path_type test_file;
path_type output_file;

path_type grammar_file;
std::string signature_name = "none";
feat_set_type feature_functions;

bool model_model1 = false;
bool model_model2 = false;
bool model_model3 = false;
bool model_model4 = false;
bool model_model5 = false;
bool model_model6 = false;
bool model_model7 = false;

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

int threads = 1;

bool feature_function_list = false;

int debug = 0;

template <typename Theta, typename Gen>
void learn(const option_set_type& options,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   const feature_set_type& feats,
	   Theta& theta,
	   Gen& gen);

void read_data(const path_type& path_input,
	       tree_set_type& trees);

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

    if (int(mix_none_mode) + mix_average_mode + mix_select_mode > 1)
      throw std::runtime_error("you can specify only one of mix-{none,average,select}");
    if (int(mix_none_mode) + mix_average_mode + mix_select_mode == 0)
      mix_none_mode = true;
    
    if (model_file.empty()) {
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5 + model_model6 + model_model7 > 1)
	throw std::runtime_error("either one of --model{1,2,3,4,5}");
      
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5 + model_model6 + model_model7 == 0)
	model_model2 = true;
    } else {
      if (int(model_model1) + model_model2 + model_model3 + model_model4 + model_model5 + model_model6 + model_model7)
	throw std::runtime_error("model file is specified via --model, but with --model{1,2,3,4,5,6, 7}?");
      
      if (! boost::filesystem::exists(model_file))
	throw std::runtime_error("no model file? " + model_file.string());
      
      switch (model_type::model(model_file)) {
      case rnnp::model::MODEL1: model_model1 = true; break;
      case rnnp::model::MODEL2: model_model2 = true; break;
      case rnnp::model::MODEL3: model_model3 = true; break;
      case rnnp::model::MODEL4: model_model4 = true; break;
      case rnnp::model::MODEL5: model_model5 = true; break;
      case rnnp::model::MODEL6: model_model6 = true; break;
      case rnnp::model::MODEL7: model_model7 = true; break;
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
    
    if (debug)
      std::cerr << "# of training data: " << trees.size() << std::endl;

    tree_set_type tests;
    if (! test_file.empty())
      read_data(test_file, tests);
    
    if (debug && ! tests.empty())
      std::cerr << "# of test data: " << tests.size() << std::endl;
    
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

    feature_set_type feats(feature_functions.begin(), feature_functions.end());
    
    if (debug)
      std::cerr << "# of features: " << feats.size() << std::endl;
    
    if (model_model1) {
      if (debug)
	std::cerr << "model1" << std::endl;
      
      rnnp::model::Model1 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model2) {
      if (debug)
	std::cerr << "model2" << std::endl;
      
      rnnp::model::Model2 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model3) {
      if (debug)
	std::cerr << "model3" << std::endl;
      
      rnnp::model::Model3 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model4) {
      if (debug)
	std::cerr << "model4" << std::endl;
      
      rnnp::model::Model4 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model5) {
      if (debug)
	std::cerr << "model5" << std::endl;
      
      rnnp::model::Model5 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model6) {
      if (debug)
	std::cerr << "model6" << std::endl;
      
      rnnp::model::Model6 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else if (model_model7) {
      if (debug)
	std::cerr << "model7" << std::endl;
      
      rnnp::model::Model7 theta(hidden_size, embedding_size, grammar);
      
      learn(optimizations, trees, tests, grammar, *signature, feats, theta, generator);
    } else
      throw std::runtime_error("no model?");
    
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}

template <typename Theta, typename Optimizer, typename Objective>
struct Task
{
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;

  typedef typename rnnp::model_traits<Theta>::model_type    model_type;
  typedef typename rnnp::model_traits<Theta>::gradient_type gradient_type;

  typedef rnnp::Loss loss_type;

  typedef std::vector<size_type, std::allocator<size_type> > working_set_type;
  
  typedef utils::lockfree_list_queue<size_type, std::allocator<size_type> >           queue_mapper_type;
  typedef utils::lockfree_list_queue<gradient_type*, std::allocator<gradient_type*> > queue_merger_type;
  typedef std::vector<queue_merger_type, std::allocator<queue_merger_type> >          queue_merger_set_type;

  typedef std::deque<gradient_type, std::allocator<gradient_type> > gradient_set_type;
  
  typedef rnnp::Evalb       evalb_type;
  typedef rnnp::EvalbScorer scorer_type;
  
  Task(const Optimizer& optimizer,
       const Objective& objective,
       const option_type& option,
       const tree_set_type& trees,
       const working_set_type& working,
       const grammar_type& grammar,
       const signature_type& signature,
       const feature_set_type& feats,
       const model_type& theta,
       queue_mapper_type& mapper,
       queue_merger_set_type& mergers)
    : optimizer_(optimizer),
      objective_(objective),
      option_(option),
      trees_(trees),
      working_(working),
      grammar_(grammar),
      signature_(signature),
      feats_(feats),
      theta_(theta),
      mapper_(mapper),
      mergers_(mergers),
      parser_(beam_size, unary_size),
      parser_oracle_(beam_size, unary_size, binarize_left)
  { }
  
  Optimizer   optimizer_;
  Objective   objective_;
  option_type option_;
  
  const tree_set_type&    trees_;
  const working_set_type& working_;
  
  working_set_type working_curr_;
  
  const grammar_type&     grammar_;
  const signature_type&   signature_;
  const feature_set_type& feats_;
  model_type theta_;
  
  queue_mapper_type&     mapper_;
  queue_merger_set_type& mergers_;

  rnnp::Parser       parser_;
  rnnp::ParserOracle parser_oracle_;

  loss_type loss_;
  size_type instances_;
  size_type parsed_;
  size_type updated_;
  
  evalb_type  evalb_;
  scorer_type scorer_;
  
  gradient_set_type gradients_;
  
  size_type      shard_;
  
  void operator()()
  {
    clear();
    
    const size_type shard_size = mergers_.size();
    const size_type batch_size = option_.batch_;

    signature_type::signature_ptr_type signature(signature_.clone());
    feature_set_type feats(feats_.clone());

    rnnp::Parser::derivation_set_type candidates;
    rnnp::Parser::derivation_set_type oracles;

    rnnp::Derivation derivation;
    
    size_type batch = 0;
    gradient_type* grad = 0;
    
    size_type merge_finished = 0;
    bool learn_finished = false;

    int non_found_iter = 0;
    
    while (merge_finished != shard_size || ! learn_finished) {
      bool found = false;
      
      if (merge_finished != shard_size)
	while (mergers_[shard_].pop(grad, true)) {
	  found = true;
	  
	  if (! grad)
	    ++ merge_finished;
	  else {
	    optimizer_(theta_, *grad, option_);
	    grad->increment();
	  }
	}
      
      if (! learn_finished && ((grad = allocate()) != 0) && mapper_.pop(batch, true)) {
	found = true;
	
	if (batch == size_type(-1)) {
	  // send termination!
	  for (size_type i = 0; i != shard_size; ++ i)
	    mergers_[i].push(0);
	  
	  learn_finished = true;
	} else {
	  grad->clear();
	  
	  const size_type first = batch * batch_size;
	  const size_type last  = utils::bithack::min(first + batch_size, working_.size());
	  
	  for (size_type id = first; id != last; ++ id) {
	    const tree_type& tree = trees_[working_[id]];

	    parser_oracle_(tree, grammar_, *signature, feats, theta_, kbest_size, oracles);
	    
	    parser_(parser_oracle_.oracle_.sentence_, grammar_, *signature, feats, theta_, kbest_size, candidates);
	    
	    parsed_ += (! candidates.empty());
	    ++ instances_;

	    if (! candidates.empty()) {
	      scorer_.assign(tree);
	      evalb_ += scorer_(candidates.front());
	    }
	    
	    if (debug >= 3) {
	      if (! candidates.empty()) {
		derivation.assign(candidates.front());
		
		std::cerr << "shard: " << shard_ << " parse: " << derivation.tree_ << std::endl;
	      } else
		std::cerr << "shard: " << shard_ << " no parse" << std::endl;
	    }
	    
	    const size_type count_curr = grad->count_;
	    
	    loss_ += objective_(theta_, parser_, parser_oracle_, option_, *grad);
	    
	    if (count_curr != grad->count_)
	      working_curr_.push_back(working_[id]);
	  }
	  
	  if (debug >= 2)
	    std::cerr << "shard: " << shard_ << " loss: " << static_cast<double>(loss_) << std::endl;
	  
	  loss_ += objective_(*grad);
	  
	  updated_ += grad->count_;

	  optimizer_(theta_, *grad, option_);
	  grad->increment();
	  
	  for (size_type i = 0; i != shard_size; ++ i)
	    if (i != shard_)
	      mergers_[i].push(grad);
	}
      }
      
      non_found_iter = loop_sleep(found, non_found_iter);
    }
  }

  gradient_type* allocate()
  {
    gradient_type* grad = 0;
    
    for (size_type j = 0; j != gradients_.size(); ++ j)
      if (gradients_[j].shared() >= mergers_.size() || gradients_[j].shared() == 0) {
	grad = &gradients_[j];
	break;
      }
    
    if (! grad && gradients_.size() < 128) {
      gradients_.push_back(gradient_type(theta_));
      grad = &gradients_.back();
    }
    
    return grad;
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

    evalb_.clear();
    
    working_curr_.clear();
  }
};

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
  
  typedef utils::lockfree_list_queue<size_type, std::allocator<size_type> > queue_type;

  Test(const tree_set_type& trees,
       Task& task,
       evalb_type& evalb,
       queue_type& queue)
    : trees_(trees),
      task_(task),
      evalb_(evalb),
      queue_(queue) {}
  
  void operator()()
  {
    signature_type::signature_ptr_type signature(task_.signature_.clone());
    feature_set_type feats(task_.feats_.clone());
    
    rnnp::Parser::derivation_set_type candidates;
    
    evalb_.clear();
    
    size_type id = 0;
    
    for (;;) {
      queue_.pop(id);
      
      if (id == size_type(-1)) break;
      
      task_.parser_(trees_[id].leaf(), task_.grammar_, *signature, feats, task_.theta_, kbest_size, candidates);

      if (candidates.empty()) continue;
      
      scorer_.assign(trees_[id]);
      evalb_ += scorer_(candidates.front());
    }
  }
  
  const tree_set_type& trees_;
  Task& task_;
  evalb_type& evalb_;
  queue_type& queue_;
  
  scorer_type scorer_;
};

template <typename Theta, typename Optimizer, typename Objective, typename Gen>
void learn(const Optimizer& optimizer,
	   const Objective& objective,
	   const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   const feature_set_type& feats,
	   Theta& theta,
	   Gen& gen);

template <typename Theta, typename Optimizer, typename Gen>
void learn(const Optimizer& optimizer,
	   const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   const feature_set_type& feats,
	   Theta& theta,
	   Gen& gen)
{
  
  if (option.margin_derivation())
    learn(optimizer, rnnp::objective::MarginDerivation(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.margin_evalb())
    learn(optimizer, rnnp::objective::MarginEvalb(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.margin_early())
    learn(optimizer, rnnp::objective::MarginEarly(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.margin_late())
    learn(optimizer, rnnp::objective::MarginLate(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.margin_max())
    learn(optimizer, rnnp::objective::MarginMax(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.violation_early())
    learn(optimizer, rnnp::objective::ViolationEarly(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.violation_late())
    learn(optimizer, rnnp::objective::ViolationLate(), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.violation_max())
    learn(optimizer, rnnp::objective::ViolationMax(), option, trees, tests, grammar, signature, feats, theta, gen);
  else
    throw std::runtime_error("unsupported objective");
}

template <typename Theta, typename Gen>
void learn(const option_type& option,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   const feature_set_type& feats,
	   Theta& theta,
	   Gen& gen)
{
  if (debug)
    std::cerr << "learning: " << option << std::endl;

  if (option.optimize_adagrad())
    learn(rnnp::optimize::AdaGrad<Theta>(theta, option.lambda_, option.eta0_, option.epsilon_), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.optimize_adagradrda())
    learn(rnnp::optimize::AdaGradRDA<Theta>(theta, option.lambda_, option.eta0_, option.epsilon_), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.optimize_adadec())
    learn(rnnp::optimize::AdaDec<Theta>(theta, option.lambda_, option.eta0_, option.epsilon_, option.gamma_), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.optimize_adadelta())
    learn(rnnp::optimize::AdaDelta<Theta>(theta, option.lambda_, option.eta0_, option.epsilon_, option.gamma_), option, trees, tests, grammar, signature, feats, theta, gen);
  else if (option.optimize_sgd())
    learn(rnnp::optimize::SGD<Theta>(theta, option.lambda_, option.eta0_), option, trees, tests, grammar, signature, feats, theta, gen);
  else
    throw std::runtime_error("unknown optimizer");
}

template <typename Theta, typename Gen>
void learn(const option_set_type& optimizations,
	   const tree_set_type& trees,
	   const tree_set_type& tests,
	   const grammar_type& grammar,
	   const signature_type& signature,
	   const feature_set_type& feats,
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
  
  int iter = 0;

  option_set_type::const_iterator oiter_end = optimizations.end();
  for (option_set_type::const_iterator oiter = optimizations.begin(); oiter != oiter_end; ++ oiter, ++ iter) {
    output_prefix = output_file.string() + "." + utils::lexical_cast<std::string>(iter);
    
    learn(*oiter, trees, tests, grammar, signature, feats, theta, gen);
  }

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
	   const feature_set_type& feats,
	   Theta& theta,
	   Gen& gen)
{
  typedef OutputModel<Theta> output_model_type;

  typedef Task<Theta, Optimizer, Objective> task_type;
  typedef std::vector<task_type, std::allocator<task_type> > task_set_type;
  
  typedef typename task_type::queue_mapper_type     queue_mapper_type;
  typedef typename task_type::queue_merger_set_type queue_merger_set_type;
  
  typedef typename task_type::loss_type loss_type;
  typedef typename task_type::size_type size_type;

  typedef typename task_type::working_set_type working_set_type;
  
  typedef std::vector<size_type, std::allocator<size_type> > batch_set_type;

  const bool perform_testing = ! tests.empty();

  working_set_type working(trees.size());
  for (size_type i = 0; i != trees.size(); ++ i)
    working[i] = i;

  typename output_model_type::queue_type queue_dumper;
  
  std::auto_ptr<boost::thread> dumper(new boost::thread(output_model_type(queue_dumper)));
  
  queue_mapper_type     mapper(threads);
  queue_merger_set_type mergers(threads);
  
  task_set_type tasks(threads, task_type(optimizer,
					 objective,
					 option,
					 trees,
					 working,
					 grammar,
					 signature,
					 feats,
					 theta,
					 mapper,
					 mergers));
  
  // assign shard id
  for (size_type shard = 0; shard != tasks.size(); ++ shard)
    tasks[shard].shard_ = shard;
  
  double evalb_max = 0;
  int zero_iter = 0;
  
  for (int t = 0; t < option.iteration_; ++ t) {
    if (debug)
      std::cerr << "iteration: " << (t + 1) << std::endl;

    const size_type batches_size = (working.size() + option.batch_ - 1) / option.batch_;
    
    std::auto_ptr<boost::progress_display> progress(debug
						    ? new boost::progress_display(batches_size, std::cerr, "", "", "")
						    : 0);
    
    utils::resource start;

    boost::thread_group workers;

    for (size_type i = 0; i != tasks.size(); ++ i)
      workers.add_thread(new boost::thread(boost::ref(tasks[i])));
    
    for (size_type b = 0; b != batches_size; ++ b) {
      mapper.push(b);
      
      if (debug)
	++ (*progress);
    }
    
    // termination
    for (size_type i = 0; i != tasks.size(); ++ i)
      mapper.push(size_type(-1));
    
    workers.join_all();
    
    utils::resource end;
    
    loss_type loss;
    size_type instances = 0;
    size_type parsed    = 0;
    size_type updated   = 0;

    typename task_type::evalb_type evalb;
    
    if (option.shrinking_) {
      working.clear();
      for (size_type i = 0; i != tasks.size(); ++ i) {
	loss      += tasks[i].loss_;
	instances += tasks[i].instances_;
	parsed    += tasks[i].parsed_;
	updated   += tasks[i].updated_;
	evalb     += tasks[i].evalb_;
	
	working.insert(working.end(), tasks[i].working_curr_.begin(), tasks[i].working_curr_.end());
      }
      
      if (working.empty() || t + 2 >= option.iteration_) {
	working.resize(trees.size());
	for (size_type i = 0; i != trees.size(); ++ i)
	  working[i] = i;
      }
    } else {
      for (size_type i = 0; i != tasks.size(); ++ i) {
	loss      += tasks[i].loss_;
	instances += tasks[i].instances_;
	parsed    += tasks[i].parsed_;
	updated   += tasks[i].updated_;
	evalb     += tasks[i].evalb_;
      }
    }
    
    if (debug)
      std::cerr << "loss: " << static_cast<double>(loss) << std::endl
		<< "evalb: " << evalb() << std::endl
		<< "instances: " << instances
		<< " parsed: " << parsed
		<< " updated: " << updated << std::endl;
    
    if (debug)
      std::cerr << "cpu time:    " << end.cpu_time() - start.cpu_time() << std::endl
		<< "user time:   " << end.user_time() - start.user_time() << std::endl;
    
    // shuffle trees!
    {
      boost::random_number_generator<boost::mt19937> rng(gen);
      
      std::random_shuffle(working.begin(), working.end(), rng);
    }
    
    // mixing
    if (mix_average_mode) {
      for (size_type i = 1; i < tasks.size(); ++ i)
	tasks.front().theta_ += tasks[i].theta_;
      
      tasks.front().theta_ *= 1.0 / tasks.size();
      
      for (size_type i = 1; i < tasks.size(); ++ i)
	tasks[i].theta_ = tasks.front().theta_;
    } else if (mix_select_mode) {
      size_type shard_min = 0;
      double l1_min = tasks.front().theta_.l1();
      
      for (size_type i = 1; i < tasks.size(); ++ i) {
	const double l1 = tasks[i].theta_.l1();
	
	if (l1 < l1_min) {
	  shard_min = i;
	  l1_min = l1;
	}
      }
      
      if (debug)
	std::cerr << "mix select: " << shard_min << " L1: " << l1_min << std::endl;
      
      for (size_type i = 0; i < tasks.size(); ++ i)
	if (i != shard_min)
	  tasks[i].theta_ = tasks[shard_min].theta_;
    } else {
      for (size_type i = 1; i < tasks.size(); ++ i)
	tasks[i].theta_ = tasks.front().theta_;
    }
    
    if (dump > 0 && !((t + 1) % dump))
      queue_dumper.push(std::make_pair(tasks.front().theta_,
				       output_prefix.string() + "." + utils::lexical_cast<std::string>(t + 1)));

    if (perform_testing) {
      typedef Test<Theta, task_type> test_type;
      
      typedef typename test_type::queue_type queue_type;
      typedef typename test_type::evalb_type evalb_type;
      
      typedef std::vector<evalb_type, std::allocator<evalb_type> > evalb_set_type;
      
      if (debug)
	std::cerr << "testing: " << (t + 1) << std::endl;
      
      queue_type queue(threads);
      evalb_set_type evalb(threads);
      
      std::auto_ptr<boost::progress_display> progress(debug
						      ? new boost::progress_display(tests.size(), std::cerr, "", "", "")
						      : 0);
      
      boost::thread_group workers;
      
      utils::resource start;
      
      for (size_type i = 0; i != tasks.size(); ++ i)
	workers.add_thread(new boost::thread(test_type(tests, tasks[i], evalb[i], queue)));
      
      for (size_type id = 0; id != tests.size(); ++ id) {
	queue.push(id);
	
	if (debug)
	  ++ (*progress);
      }
      
      for (size_type i = 0; i != tasks.size(); ++ i)
	queue.push(size_type(-1));
      
      workers.join_all();

      utils::resource end;
      
      for (size_type i = 1; i != tasks.size(); ++ i)
	evalb.front() += evalb[i];

      const double evalb_curr = evalb.front()();
      
      if (debug)
	std::cerr << "EVALB: " << evalb_curr << std::endl
		  << "test cpu time:    " << end.cpu_time() - start.cpu_time() << std::endl
		  << "test user time:   " << end.user_time() - start.user_time() << std::endl;
      
      if (evalb_curr > evalb_max) {
	evalb_max = evalb_curr;
	theta = tasks.front().theta_;
      } else if (t && evalb_curr < evalb_max && option.decay_) {
	for (size_type i = 0; i != tasks.size(); ++ i)
	  tasks[i].optimizer_.decay();
      }
    }
    
    if (! updated)
      ++ zero_iter;
    else
      zero_iter = 0;
    
    if (zero_iter >= 2) break;
  }
  
  queue_dumper.push(typename output_model_type::model_path_type());
  dumper->join();
  
  // copy the model!
  if (! perform_testing)
    theta = tasks.front().theta_;
}

void read_data(const path_type& path,
	       tree_set_type& trees)
{
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
    ("feature",    po::value<feat_set_type>(&feature_functions)->composing(),              "feature function(s)")
    
    ("model1",    po::bool_switch(&model_model1), "parsing by model1")
    ("model2",    po::bool_switch(&model_model2), "parsing by model2 (default)")
    ("model3",    po::bool_switch(&model_model3), "parsing by model3")
    ("model4",    po::bool_switch(&model_model4), "parsing by model4")
    ("model5",    po::bool_switch(&model_model5), "parsing by model5")
    ("model6",    po::bool_switch(&model_model6), "parsing by model6")
    ("model7",    po::bool_switch(&model_model7), "parsing by model7")

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
    ("threads", po::value<int>(&threads)->default_value(threads), "# of threads")

    ("feature-list", po::bool_switch(&feature_function_list), "list of feature functions")
    
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
