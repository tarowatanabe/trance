//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

//
// collect grammar
//
//
// we will peform witten-bell smoothing or KN smoothing...?
//

#include <iostream>

#include <rnnp/symbol.hpp>
#include <rnnp/tree.hpp>
#include <rnnp/rule.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/binarize.hpp>
#include <rnnp/signature.hpp>

#include "utils/compress_stream.hpp"
#include "utils/compact_map.hpp"
#include "utils/unordered_map.hpp"
#include "utils/bithack.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

typedef boost::filesystem::path path_type;

typedef rnnp::Symbol  symbol_type;
typedef rnnp::Tree    tree_type;
typedef rnnp::Rule    rule_type;

typedef uint64_t count_type;
typedef double   prob_type;
typedef utils::compact_map<symbol_type, count_type,
			   utils::unassigned<symbol_type>, utils::unassigned<symbol_type>,
			   boost::hash<symbol_type>, std::equal_to<symbol_type>,
			   std::allocator<std::pair<const symbol_type, count_type> > > unigram_type;


struct GrammarCount
{
  typedef utils::unordered_map<rule_type, count_type,
			       boost::hash<rule_type>, std::equal_to<rule_type>,
			       std::allocator<std::pair<const rule_type, count_type> > >::type rule_set_type;
  
  
  symbol_type  goal_;
  unigram_type sentence_;
  
  rule_set_type binary_;
  rule_set_type unary_;
  rule_set_type preterminal_;
};

struct GrammarPCFG
{
  typedef utils::unordered_map<rule_type, prob_type,
			       boost::hash<rule_type>, std::equal_to<rule_type>,
			       std::allocator<std::pair<const rule_type, prob_type> > >::type rule_set_type;
  typedef utils::unordered_map<symbol_type, rule_set_type,
			       boost::hash<symbol_type>, std::equal_to<symbol_type>,
			       std::allocator<std::pair<const symbol_type, rule_set_type> > >::type rule_map_type;
  
  symbol_type goal_;
  symbol_type sentence_;
  
  rule_map_type binary_;
  rule_map_type unary_;
  rule_map_type preterminal_;
};

typedef GrammarCount grammar_count_type;
typedef GrammarPCFG  grammar_pcfg_type;

typedef rnnp::Signature signature_type;

void collect_rules(const path_type& path,
		   grammar_count_type& grammar,
		   unigram_type& unigram,
		   const bool left=true);
void cutoff_terminal(const signature_type& signature,
		     grammar_count_type& grammar,
		     unigram_type& unigram);
void estimate(const grammar_count_type& counts,
	      grammar_pcfg_type& pcfg);
void output_grammar(const path_type& path,
		    const grammar_pcfg_type& grammar);

path_type input_file = "-";
path_type output_file = "-";

std::string signature_name = "none";

bool binarize_left = false;
bool binarize_right = false;

bool split_preterminal = false;

int cutoff = 3;

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
    
    grammar_count_type grammar;
    unigram_type unigram;

    collect_rules(input_file, grammar, unigram, binarize_left);

    if (cutoff > 0)
      cutoff_terminal(*signature_type::create(signature_name), grammar, unigram);

    grammar_pcfg_type grammar_pcfg;

    estimate(grammar, grammar_pcfg);
    
    output_grammar(output_file, grammar_pcfg);
  }
  catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}

struct CollectRules
{
  CollectRules(grammar_count_type& grammar,
	       unigram_type& unigram,
	       bool left)
    : grammar_(grammar), unigram_(unigram), left_(left), unary_max_(0) {}

  void operator()(const tree_type& tree)
  {
    if (left_)
      rnnp::binarize_left(tree, binarized_);
    else
      rnnp::binarize_right(tree, binarized_);
    
    if (grammar_.goal_ == symbol_type())
      grammar_.goal_ = binarized_.label_;
    else if (grammar_.goal_ != binarized_.label_)
      throw std::runtime_error("different goal: previous = " + static_cast<const std::string&>(grammar_.goal_)
			       + " current = " + static_cast<const std::string&>(binarized_.label_));
    
    switch (binarized_.antecedent_.size()) {
    case 1:
      ++ grammar_.sentence_[binarized_.antecedent_.front().label_];
      break;
    case 2:
      ++ grammar_.sentence_[binarized_.antecedent_.front().label_];
      ++ grammar_.sentence_[binarized_.antecedent_.back().label_];
      break;
    default: 
      throw std::runtime_error("invalid binary tree");
    }

    int unary = 0;
    extract(binarized_, unary);
  }
  
  void extract(const tree_type& tree, int& unary)
  {
    switch (tree.antecedent_.size()) {
    case 1: {
      rule_type rule(tree.label_, 1);
      rule.rhs_.front() = tree.antecedent_.front().label_;
      
      if (rule.unary()) {
	++ unary;
	++ grammar_.unary_[rule];
	
	extract(tree.antecedent_.front(), unary);
      } else if (rule.preterminal()) {
	++ grammar_.preterminal_[rule];
	
	++ unigram_[rule.rhs_.front()];
      } else
	std::runtime_error("invalid rule: " + rule.string());
    } break;
    case 2: {
      unary_max_ = utils::bithack::max(unary_max_, unary);
      
      rule_type rule(tree.label_, 2);
      rule.rhs_.front() = tree.antecedent_.front().label_;
      rule.rhs_.back()  = tree.antecedent_.back().label_;

      if (! rule.binary())
	throw std::runtime_error("invalid rule: " + rule.string());
      
      ++ grammar_.binary_[rule];

      int unary_left = 0;
      int unary_right = 0;
      
      extract(tree.antecedent_.front(), unary_left);
      extract(tree.antecedent_.back(),  unary_right);
    } break;
    default:
      throw std::runtime_error("invalid binary tree");
    }
  }

  tree_type binarized_;
  
  grammar_count_type& grammar_;
  unigram_type& unigram_;
  bool left_;

  int unary_max_;
};

void collect_rules(const path_type& path,
		   grammar_count_type& grammar,
		   unigram_type& unigram,
		   const bool left)
{
  CollectRules collect(grammar, unigram, left);

  tree_type tree;

  utils::compress_istream is(path, 1024 * 1024);
  
  while (is >> tree)
    if (! tree.empty())
      collect(tree);

  if (debug)
    std::cerr << "maximum unary size: " << collect.unary_max_ << std::endl;
}

void cutoff_terminal(const signature_type& signature,
		     grammar_count_type& grammar,
		     unigram_type& unigram)
{
  unigram_type unigram_cutoff;

  // add workaround for penntreebank
  unigram_type::const_iterator uiter_end = unigram.end();
  for (unigram_type::const_iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
    if (uiter->second >= cutoff || static_cast<const std::string&>(uiter->first).find("``", 0, 2) != std::string::npos)
      unigram_cutoff.insert(*uiter);
    else
      unigram_cutoff[signature(uiter->first)] += uiter->second;
  }
  
  unigram.swap(unigram_cutoff);
  
  grammar_count_type::rule_set_type preterminal;
  grammar_count_type::rule_set_type preterminal_oov;
  bool has_fallback = false;
  
  grammar_count_type::rule_set_type::const_iterator piter_end = grammar.preterminal_.end();
  for (grammar_count_type::rule_set_type::const_iterator piter = grammar.preterminal_.begin(); piter != piter_end; ++ piter)
    if (unigram.find(piter->first.rhs_.front()) != unigram.end())
      preterminal.insert(*piter);
    else {
      if (debug >= 2)
	std::cerr << "removing preterminal: " << piter->first << std::endl;
      
      const symbol_type sig = signature(piter->first.rhs_.front());
      
      preterminal[rule_type(piter->first.lhs_, rule_type::rhs_type(1, sig))] += piter->second;
      preterminal_oov[rule_type(piter->first.lhs_, rule_type::rhs_type(1, symbol_type::UNK))] += piter->second;
      
      has_fallback |= (sig == symbol_type::UNK);
    }

  if (! has_fallback)
    preterminal.insert(preterminal_oov.begin(), preterminal_oov.end());
  
  grammar.preterminal_.swap(preterminal);
}

struct Estimate
{
  typedef rule_type::lhs_type lhs_type;
  typedef rule_type::rhs_type rhs_type;

  typedef std::pair<count_type, prob_type> count_prob_type;

  struct rhs_hash : public utils::hashmurmur3<size_t>
  {
    typedef utils::hashmurmur3<size_t> hasher_type;
    
    size_t operator()(const rhs_type& x) const
    {
      return hasher_type::operator()(x.begin(), x.end(), 0);
    }
  };
  
  typedef utils::unordered_map<rhs_type, count_prob_type,
			       rhs_hash, std::equal_to<rhs_type>,
			       std::allocator<std::pair<const rhs_type, count_prob_type> > >::type rhs_set_type;
  
  typedef utils::unordered_map<rule_type, count_prob_type,
			       boost::hash<rule_type>, std::equal_to<rule_type>,
			       std::allocator<std::pair<const rule_type, count_prob_type> > >::type rule_set_type;
  typedef utils::unordered_map<symbol_type, rule_set_type,
			       boost::hash<symbol_type>, std::equal_to<symbol_type>,
			       std::allocator<std::pair<const symbol_type, rule_set_type> > >::type rule_map_type;

  
};

void estimate(const grammar_count_type& counts,
	      grammar_pcfg_type& pcfg)
{
  typedef Estimate estimate_type;

  // assign goal and sentence
  pcfg.goal_ = counts.goal_;
  
  count_type count = 0;
  if (counts.sentence_.empty())
    throw std::runtime_error("invalid pre-goal label");
  
  unigram_type::const_iterator siter_end = counts.sentence_.end();
  for (unigram_type::const_iterator siter = counts.sentence_.begin(); siter != siter_end; ++ siter)
    if (siter->second > count) {
      pcfg.sentence_ = siter->first;
      count = siter->second;
    }
  
  //
  // now, we perform actual estimation....
  //
  
  // First, restrcture..
  estimate_type::rule_map_type unary;
  estimate_type::rule_map_type binary;
  estimate_type::rule_map_type preterminal;

  // unary..
  grammar_count_type::rule_set_type::const_iterator uiter_end = counts.unary_.end();
  for (grammar_count_type::rule_set_type::const_iterator uiter = counts.unary_.begin(); uiter != uiter_end; ++ uiter)
    unary[uiter->first.lhs_][uiter->first].first = uiter->second;
  
  // binary...
  grammar_count_type::rule_set_type::const_iterator biter_end = counts.binary_.end();
  for (grammar_count_type::rule_set_type::const_iterator biter = counts.binary_.begin(); biter != biter_end; ++ biter)
    binary[biter->first.lhs_][biter->first].first = biter->second;
  
  // preterminal..
  grammar_count_type::rule_set_type::const_iterator piter_end = counts.preterminal_.end();
  for (grammar_count_type::rule_set_type::const_iterator piter = counts.preterminal_.begin(); piter != piter_end; ++ piter)
    preterminal[piter->first.lhs_][piter->first].first = piter->second;
  
  // Second, collect lower order count
  
  if (split_preterminal) {
    
    {
      estimate_type::rhs_set_type unigram;

      count_type t1[5];
      count_type t2[5];
      std::fill(t1, t1 + 5, count_type(0));
      std::fill(t2, t2 + 5, count_type(0));
      
      {
	estimate_type::rule_map_type::const_iterator uiter_end = unary.end();
	for (estimate_type::rule_map_type::const_iterator uiter = unary.begin(); uiter != uiter_end; ++ uiter) {
	  estimate_type::rule_set_type::const_iterator riter_end = uiter->second.end();
	  for (estimate_type::rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter) {
	    ++ unigram[riter->first.rhs_].first;
	    
	    if (riter->second.first <= 4)
	      ++ t2[riter->second.first];
	  }
	}
	
	estimate_type::rule_map_type::const_iterator biter_end = binary.end();
	for (estimate_type::rule_map_type::const_iterator biter = binary.begin(); biter != biter_end; ++ biter) {
	  estimate_type::rule_set_type::const_iterator riter_end = biter->second.end();
	  for (estimate_type::rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter) {
	    ++ unigram[riter->first.rhs_].first;
	    
	    if (riter->second.first <= 4)
	      ++ t2[riter->second.first];
	  }
	}
      }
      
      count_type total = 0;
      
      {
	estimate_type::rhs_set_type::const_iterator uiter_end = unigram.end();
	for (estimate_type::rhs_set_type::const_iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	  if (uiter->second.first <= 4)
	    ++ t1[uiter->second.first];
	  
	  total += uiter->second.first;
	}
      }
      
      // estimate discount
      double d1[4];
      double d2[4];
      std::fill(d1, d1 + 4, double(0));
      std::fill(d2, d2 + 4, double(0));
      
      for (int k = 1; k != 4; ++ k)
	d1[k] = double(k) - double((k + 1) * t1[1] * t1[k+1]) / double((t1[1] + 2 * t1[2]) * t1[k]);
      
      for (int k = 1; k != 4; ++ k)
	d2[k] = double(k) - double((k + 1) * t2[1] * t2[k+1]) / double((t2[1] + 2 * t2[2]) * t2[k]);

      if (debug) {
	std::cerr << "grammar" << std::endl;
	
	for (int k = 1; k != 4; ++ k) {
	  if (k > 1)
	    std::cerr << ' ';
	  std::cerr << "D1[" << k << "] = " << d1[k];
	}
	std::cerr << std::endl;
	
	for (int k = 1; k != 4; ++ k) {
	  if (k > 1)
	    std::cerr << ' ';
	  std::cerr << "D2[" << k << "] = " << d2[k];
	}
	std::cerr << std::endl;
      }
      
      // estimate unigram probability...
      {
	const double factor = 1.0 / total;
	const double uniform = 1.0 / unigram.size();
	
	double backoff = 0.0;
	
	estimate_type::rhs_set_type::iterator uiter_end = unigram.end();
	for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	  backoff += d1[utils::bithack::min(uiter->second.first, count_type(3))];
	
	double sum = 0.0;
	
	for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	  uiter->second.second = factor * (double(uiter->second.first)
					   - d1[utils::bithack::min(uiter->second.first, count_type(3))]
					   + backoff * uniform);
	  
	  sum += uiter->second.second;
	}
	
	// make sure that we are correctly normalized
	const double factor_sum = 1.0 / sum;
	for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	  uiter->second.second *= factor_sum;
      }
      
      // estimate rule probability
      estimate_type::rule_map_type::iterator uiter_end = unary.end();
      for (estimate_type::rule_map_type::iterator uiter = unary.begin(); uiter != uiter_end; ++ uiter) {
	count_type total = 0;
	
	estimate_type::rule_set_type::iterator riter_end = uiter->second.end();
	for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	  total += riter->second.first;
	
	const double factor = 1.0 / total;
	
	double backoff = 0.0;
	for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	  backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
	for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter) {
	  const prob_type lower = unigram[riter->first.rhs_].second;
	  
	  riter->second.second = factor * (double(riter->second.first)
					   - d2[utils::bithack::min(riter->second.first, count_type(3))]
					   + backoff * lower);
	}
      }
      
      estimate_type::rule_map_type::iterator biter_end = binary.end();
      for (estimate_type::rule_map_type::iterator biter = binary.begin(); biter != biter_end; ++ biter) {
	count_type total = 0;
	
	estimate_type::rule_set_type::iterator riter_end = biter->second.end();
	for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	  total += riter->second.first;
	
	const double factor = 1.0 / total;
	
	double backoff = 0.0;
	for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	  backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
	for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter) {
	  const prob_type lower = unigram[riter->first.rhs_].second;
	  
	  riter->second.second = factor * (double(riter->second.first)
					   - d2[utils::bithack::min(riter->second.first, count_type(3))]
					   + backoff * lower);
	}
      }
    }
    
    {
      // collect modified count and count of count
      
      estimate_type::rhs_set_type unigram;
      
      count_type t1[5];
      count_type t2[5];
      std::fill(t1, t1 + 5, count_type(0));
      std::fill(t2, t2 + 5, count_type(0));
      
      {
	estimate_type::rule_map_type::const_iterator piter_end = preterminal.end();
	for (estimate_type::rule_map_type::const_iterator piter = preterminal.begin(); piter != piter_end; ++ piter) {
	  estimate_type::rule_set_type::const_iterator riter_end = piter->second.end();
	  for (estimate_type::rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter) {
	    ++ unigram[riter->first.rhs_].first;
	    
	    if (riter->second.first <= 4)
	      ++ t2[riter->second.first];
	  }
	}
      }
      
      count_type total = 0;

      {
	estimate_type::rhs_set_type::const_iterator uiter_end = unigram.end();
	for (estimate_type::rhs_set_type::const_iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	  if (uiter->second.first <= 4)
	    ++ t1[uiter->second.first];
	  
	  total += uiter->second.first;
	}
      }
      
      // estimate discount
      double d1[4];
      double d2[4];
      std::fill(d1, d1 + 4, double(0));
      std::fill(d2, d2 + 4, double(0));
      
      for (int k = 1; k != 4; ++ k)
	d1[k] = double(k) - double((k + 1) * t1[1] * t1[k+1]) / double((t1[1] + 2 * t1[2]) * t1[k]);
      
      for (int k = 1; k != 4; ++ k)
	d2[k] = double(k) - double((k + 1) * t2[1] * t2[k+1]) / double((t2[1] + 2 * t2[2]) * t2[k]);
      
      if (debug) {
	std::cerr << "pre-terminal" << std::endl;
	
	for (int k = 1; k != 4; ++ k) {
	  if (k > 1)
	    std::cerr << ' ';
	  std::cerr << "D1[" << k << "] = " << d1[k];
	}
	std::cerr << std::endl;
	
	for (int k = 1; k != 4; ++ k) {
	  if (k > 1)
	    std::cerr << ' ';
	  std::cerr << "D2[" << k << "] = " << d2[k];
	}
	std::cerr << std::endl;
      }
      
      // estimate unigram probability...
      const double factor = 1.0 / total;
      const double uniform = 1.0 / unigram.size();
      
      double backoff = 0.0;
      
      estimate_type::rhs_set_type::iterator uiter_end = unigram.end();
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	backoff += d1[utils::bithack::min(uiter->second.first, count_type(3))];

      double sum = 0.0;
      
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	uiter->second.second = factor * (double(uiter->second.first)
					 - d1[utils::bithack::min(uiter->second.first, count_type(3))]
					 + backoff * uniform);
	
	sum += uiter->second.second;
      }
      
      // make sure that we are correctly normalized
      const double factor_sum = 1.0 / sum;
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	uiter->second.second *= factor_sum;
      
      // estimate rule probability...
      estimate_type::rule_map_type::iterator piter_end = preterminal.end();
      for (estimate_type::rule_map_type::iterator piter = preterminal.begin(); piter != piter_end; ++ piter) {
	count_type total = 0;
	
	estimate_type::rule_set_type::iterator riter_end = piter->second.end();
	for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	  total += riter->second.first;
	
	const double factor = 1.0 / total;
	
	double backoff = 0.0;
	for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	  backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
	for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter) {
	  const prob_type lower = unigram[riter->first.rhs_].second;
	  
	  riter->second.second = factor * (double(riter->second.first)
					   - d2[utils::bithack::min(riter->second.first, count_type(3))]
					   + backoff * lower);
	}
      }
    }
  } else {
    estimate_type::rhs_set_type unigram;

    count_type t1[5];
    count_type t2[5];
    std::fill(t1, t1 + 5, count_type(0));
    std::fill(t2, t2 + 5, count_type(0));
      
    {
      estimate_type::rule_map_type::const_iterator uiter_end = unary.end();
      for (estimate_type::rule_map_type::const_iterator uiter = unary.begin(); uiter != uiter_end; ++ uiter) {
	estimate_type::rule_set_type::const_iterator riter_end = uiter->second.end();
	for (estimate_type::rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter) {
	  ++ unigram[riter->first.rhs_].first;
	    
	  if (riter->second.first <= 4)
	    ++ t2[riter->second.first];
	}
      }
      
      estimate_type::rule_map_type::const_iterator biter_end = binary.end();
      for (estimate_type::rule_map_type::const_iterator biter = binary.begin(); biter != biter_end; ++ biter) {
	estimate_type::rule_set_type::const_iterator riter_end = biter->second.end();
	for (estimate_type::rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter) {
	  ++ unigram[riter->first.rhs_].first;
	    
	  if (riter->second.first <= 4)
	    ++ t2[riter->second.first];
	}
      }
      
      estimate_type::rule_map_type::const_iterator piter_end = preterminal.end();
      for (estimate_type::rule_map_type::const_iterator piter = preterminal.begin(); piter != piter_end; ++ piter) {
	estimate_type::rule_set_type::const_iterator riter_end = piter->second.end();
	for (estimate_type::rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter) {
	  ++ unigram[riter->first.rhs_].first;
	    
	  if (riter->second.first <= 4)
	    ++ t2[riter->second.first];
	}
      }
    }
      
    count_type total = 0;
      
    {
      estimate_type::rhs_set_type::const_iterator uiter_end = unigram.end();
      for (estimate_type::rhs_set_type::const_iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	if (uiter->second.first <= 4)
	  ++ t1[uiter->second.first];
	  
	total += uiter->second.first;
      }
    }
      
    // estimate discount
    double d1[4];
    double d2[4];
    std::fill(d1, d1 + 4, double(0));
    std::fill(d2, d2 + 4, double(0));
      
    for (int k = 1; k != 4; ++ k)
      d1[k] = double(k) - double((k + 1) * t1[1] * t1[k+1]) / double((t1[1] + 2 * t1[2]) * t1[k]);
      
    for (int k = 1; k != 4; ++ k)
      d2[k] = double(k) - double((k + 1) * t2[1] * t2[k+1]) / double((t2[1] + 2 * t2[2]) * t2[k]);

    if (debug) {
      for (int k = 1; k != 4; ++ k) {
	if (k > 1)
	  std::cerr << ' ';
	std::cerr << "D1[" << k << "] = " << d1[k];
      }
      std::cerr << std::endl;
      
      for (int k = 1; k != 4; ++ k) {
	if (k > 1)
	  std::cerr << ' ';
	std::cerr << "D2[" << k << "] = " << d2[k];
      }
      std::cerr << std::endl;
    }
    
    // estimate unigram probability...
    {
      const double factor = 1.0 / total;
      const double uniform = 1.0 / unigram.size();
      
      double backoff = 0.0;

      estimate_type::rhs_set_type::iterator uiter_end = unigram.end();
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	backoff += d1[utils::bithack::min(uiter->second.first, count_type(3))];

      double sum = 0.0;
	
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter) {
	uiter->second.second = factor * (double(uiter->second.first)
					 - d1[utils::bithack::min(uiter->second.first, count_type(3))]
					 + backoff * uniform);
	
	sum += uiter->second.second;
      }
      
      // make sure that we are correctly normalized
      const double factor_sum = 1.0 / sum;
      for (estimate_type::rhs_set_type::iterator uiter = unigram.begin(); uiter != uiter_end; ++ uiter)
	uiter->second.second *= factor_sum;
    }
      
    // estimate rule probability
    estimate_type::rule_map_type::iterator uiter_end = unary.end();
    for (estimate_type::rule_map_type::iterator uiter = unary.begin(); uiter != uiter_end; ++ uiter) {
      count_type total = 0;
	
      estimate_type::rule_set_type::iterator riter_end = uiter->second.end();
      for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	total += riter->second.first;
	
      const double factor = 1.0 / total;
	
      double backoff = 0.0;
      for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
      for (estimate_type::rule_set_type::iterator riter = uiter->second.begin(); riter != riter_end; ++ riter) {
	const prob_type lower = unigram[riter->first.rhs_].second;
	  
	riter->second.second = factor * (double(riter->second.first)
					 - d2[utils::bithack::min(riter->second.first, count_type(3))]
					 + backoff * lower);
      }
    }
      
    estimate_type::rule_map_type::iterator biter_end = binary.end();
    for (estimate_type::rule_map_type::iterator biter = binary.begin(); biter != biter_end; ++ biter) {
      count_type total = 0;
	
      estimate_type::rule_set_type::iterator riter_end = biter->second.end();
      for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	total += riter->second.first;
	
      const double factor = 1.0 / total;
	
      double backoff = 0.0;
      for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
      for (estimate_type::rule_set_type::iterator riter = biter->second.begin(); riter != riter_end; ++ riter) {
	const prob_type lower = unigram[riter->first.rhs_].second;
	  
	riter->second.second = factor * (double(riter->second.first)
					 - d2[utils::bithack::min(riter->second.first, count_type(3))]
					 + backoff * lower);
      }
    }

    estimate_type::rule_map_type::iterator piter_end = preterminal.end();
    for (estimate_type::rule_map_type::iterator piter = preterminal.begin(); piter != piter_end; ++ piter) {
      count_type total = 0;
	
      estimate_type::rule_set_type::iterator riter_end = piter->second.end();
      for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	total += riter->second.first;
	
      const double factor = 1.0 / total;
	
      double backoff = 0.0;
      for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	backoff += d2[utils::bithack::min(riter->second.first, count_type(3))];
	
      for (estimate_type::rule_set_type::iterator riter = piter->second.begin(); riter != riter_end; ++ riter) {
	const prob_type lower = unigram[riter->first.rhs_].second;
	  
	riter->second.second = factor * (double(riter->second.first)
					 - d2[utils::bithack::min(riter->second.first, count_type(3))]
					 + backoff * lower);
      }
    }
  }

  {
    estimate_type::rule_map_type::const_iterator uiter_end = unary.end();
    for (estimate_type::rule_map_type::const_iterator uiter = unary.begin(); uiter != uiter_end; ++ uiter) {
      estimate_type::rule_set_type::const_iterator riter_end = uiter->second.end();
      for (estimate_type::rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	pcfg.unary_[riter->first.lhs_][riter->first] = riter->second.second;
    }
    
    estimate_type::rule_map_type::const_iterator biter_end = binary.end();
    for (estimate_type::rule_map_type::const_iterator biter = binary.begin(); biter != biter_end; ++ biter) {
      estimate_type::rule_set_type::const_iterator riter_end = biter->second.end();
      for (estimate_type::rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	pcfg.binary_[riter->first.lhs_][riter->first] = riter->second.second;
    }
    
    estimate_type::rule_map_type::const_iterator piter_end = preterminal.end();
    for (estimate_type::rule_map_type::const_iterator piter = preterminal.begin(); piter != piter_end; ++ piter) {
      estimate_type::rule_set_type::const_iterator riter_end = piter->second.end();
      for (estimate_type::rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	pcfg.preterminal_[riter->first.lhs_][riter->first] = riter->second.second;
    }
  }
}

void output_grammar(const path_type& path,
		    const grammar_pcfg_type& grammar)
{
  utils::compress_ostream os(path, 1024 * 1024);
  os.precision(10);
  
  os << grammar.goal_ << '\n';
  os << grammar.sentence_ << '\n';
  os << '\n';

  grammar_pcfg_type::rule_map_type::const_iterator uiter_end = grammar.unary_.end();
  for (grammar_pcfg_type::rule_map_type::const_iterator uiter = grammar.unary_.begin(); uiter != uiter_end; ++ uiter) {
    grammar_pcfg_type::rule_set_type::const_iterator riter_end = uiter->second.end();
    for (grammar_pcfg_type::rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
      os << riter->first << " ||| " << std::log(riter->second) << '\n';
  }
  os << '\n';
  
  grammar_pcfg_type::rule_map_type::const_iterator biter_end = grammar.binary_.end();
  for (grammar_pcfg_type::rule_map_type::const_iterator biter = grammar.binary_.begin(); biter != biter_end; ++ biter) {
    grammar_pcfg_type::rule_set_type::const_iterator riter_end = biter->second.end();
    for (grammar_pcfg_type::rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
      os << riter->first << " ||| " << std::log(riter->second) << '\n';
  }
  os << '\n';

  grammar_pcfg_type::rule_map_type::const_iterator piter_end = grammar.preterminal_.end();
  for (grammar_pcfg_type::rule_map_type::const_iterator piter = grammar.preterminal_.begin(); piter != piter_end; ++ piter) {
    grammar_pcfg_type::rule_set_type::const_iterator riter_end = piter->second.end();
    for (grammar_pcfg_type::rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
      os << riter->first << " ||| " << std::log(riter->second) << '\n';
  }
}


void options(int argc, char** argv)
{
  namespace po = boost::program_options;

  po::options_description desc("options");
  desc.add_options()
    ("input",     po::value<path_type>(&input_file)->default_value(input_file),   "input file")
    ("output",    po::value<path_type>(&output_file)->default_value(output_file), "output")

    ("signature", po::value<std::string>(&signature_name)->default_value(signature_name), "language specific signature")
    
    ("binarize-left",  po::bool_switch(&binarize_left),  "left recursive (or left heavy) binarization (default)")
    ("binarize-right", po::bool_switch(&binarize_right), "right recursive (or right heavy) binarization")
    
    ("split-preterminal", po::bool_switch(&split_preterminal), "split preterminals from other non-terminals (like Penn-treebank)")
    
    ("cutoff",    po::value<int>(&cutoff)->default_value(cutoff),           "OOV cutoff")
    
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


