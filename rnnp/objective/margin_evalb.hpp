// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN_EVALB__HPP__
#define __RNNP__OBJECTIVE__MARGIN_EVALB__HPP__ 1

#include <limits>

#include <rnnp/evalb.hpp>
#include <rnnp/objective/margin.hpp>
#include <rnnp/semiring/log.hpp>

#include <utils/compact_set.hpp>

namespace rnnp
{
  namespace objective
  {
    //
    // compute margin by expected evalb from orcle!
    //
    
    struct MarginEvalb : public objective::Margin
    {
      typedef rnnp::semiring::Log<double> weight_type;
      
      typedef EvalbScorer scorer_type;
      typedef std::vector<scorer_type, std::allocator<scorer_type> > scorer_set_type;

      typedef std::vector<double, std::allocator<double> >           score_set_type;
      typedef std::vector<double, std::allocator<double> >           margin_set_type;
      typedef std::vector<weight_type, std::allocator<weight_type> > prob_set_type;
      
      scorer_set_type evalb_;
      score_set_type  score_;
      margin_set_type margin_;
      prob_set_type   prob_;

      double margin(const model_type& theta,
		    const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option,
		    gradient_type& g)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	
	size_type step_back = candidates.agenda_.size() - 1;
	for (/**/; step_back > 0; -- step_back)
	  if (! candidates.agenda_[step_back].empty() && ! oracles.agenda_[step_back].empty()) break;
	
	// this should not happen...
	if (step_back == 0) return 0.0;
	
	const size_type kbest_candidate_size = candidates.agenda_[step_back].size();
	const size_type kbest_oracle_size    = oracles.agenda_[step_back].size();
	
	// first, collect oracle evalb stats...
	evalb_.clear();
	evalb_.resize(kbest_oracle_size);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  evalb_[o].assign(oracles.agenda_[step_back][o]);
	
	score_.clear();
	score_.resize(kbest_candidate_size, 0);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) 
	    score_[c] = std::max(score_[c], evalb_[o](candidates.agenda_[step_back][c])());
	
	weight_type Z;
	
	margin_.clear();
	margin_.resize(kbest_candidate_size);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const double margin = candidates.agenda_[step_back][c].score() * option.scale_;
	  
	  margin_[c] = margin;
	  Z += semiring::traits<weight_type>::exp(margin);
	}
	
	weight_type expectation;
      
	prob_.clear();
	prob_.resize(kbest_candidate_size);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const weight_type prob = semiring::traits<weight_type>::exp(margin_[c]) / Z;
	  
	  prob_[c] = prob;
	  expectation += score_[c] * prob;
	}
	
	const double objective = - expectation;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const weight_type loss = - (score_[c] - expectation) * prob_[c];
	  
	  if (loss == weight_type()) continue;
	  
	  backward_[candidates.agenda_[step_back][c]].loss_ += loss;
	  states_[candidates.agenda_[step_back][c].step()].insert(candidates.agenda_[step_back][c]);
	}
	
	return objective;
      }
    };
  };
};

#endif

