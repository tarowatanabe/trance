// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION_MAX__HPP__
#define __RNNP__OBJECTIVE__VIOLATION_MAX__HPP__ 1

// this is a max violation..

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct ViolationMax : public objective::Margin
    {
      typedef rnnp::semiring::Logprob<double> weight_type;
    
      double margin(const model_type& theta,
		    const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option,
		    gradient_type& g)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	
	double error_max = 0.0;
	size_type step_max = size_type(-1);
	
	sentence_type cand;
	
	// check the state with max-violation
	for (size_type step = 0; step != oracles.agenda_.size(); ++ step) 
	  if (! candidates.agenda_[step].empty() && ! oracles.agenda_[step].empty()) {
	    const double score_candidate = candidates.agenda_[step].back().score();
	    const double score_oracle    = oracles.agenda_[step].back().score();
	    
	    const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	    
	    if (error > error_max) {
	      error_max = error;
	      step_max = step;
	    }
	  }
	
	if (step_max == size_type(-1)) return 0.0;
	
	const size_type kbest_candidate_size = candidates.agenda_[step_max].size();
	const size_type kbest_oracle_size    = oracles.agenda_[step_max].size();
	
	weight_type Z_candidate;
	weight_type Z_oracle;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  Z_candidate += rnnp::semiring::traits<weight_type>::exp(candidates.agenda_[step_max][c].score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += rnnp::semiring::traits<weight_type>::exp(oracles.agenda_[step_max][o].score());
	
	double loss = 0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    const state_type& state_candidate = candidates.agenda_[step_max][c];
	    const state_type& state_oracle    = oracles.agenda_[step_max][o];
	    
	    const double& score_candidate = state_candidate.score();
	    const double& score_oracle    = state_oracle.score();
	    
	    const bool suffered = score_candidate > score_oracle;
	    const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	    
	    if (! suffered || error <= 0.0) continue;
	    
	    const weight_type prob_candidate = semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
	    const weight_type prob_oracle    = semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
	    
	    const double loss_factor = prob_candidate * prob_oracle;
	    
	    backward_type& backward_candidate = backward_[state_candidate];
	    backward_type& backward_oracle    = backward_[state_oracle];
	    
	    backward_candidate.loss_ += loss_factor;
	    backward_oracle.loss_    -= loss_factor;
	    
	    loss += error * loss_factor;
	  }
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  states_[candidates.agenda_[step_max][c].step()].insert(candidates.agenda_[step_max][c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  states_[oracles.agenda_[step_max][o].step()].insert(oracles.agenda_[step_max][o]);
	
	return loss;
      }
    };
  };
};

#endif
