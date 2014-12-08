// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__OBJECTIVE__MARGIN_MAX__HPP__
#define __TRANCE__OBJECTIVE__MARGIN_MAX__HPP__ 1

#include <trance/objective/margin.hpp>

#include <trance/semiring/logprob.hpp>

namespace trance
{
  namespace objective
  {
    struct MarginMax : public objective::Margin
    {
      typedef trance::semiring::Logprob<double> weight_type;

      double margin(const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");

	if (option.margin_all_) {
	  const size_type kbest_candidate_size = candidates.agenda_.back().size();
	  const size_type kbest_oracle_size    = oracles.agenda_.back().size();
	  
	  weight_type Z_candidate;
	  weight_type Z_oracle;

	  double score_max = - std::numeric_limits<double>::infinity();

	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    score_max = std::max(score_max, oracles.agenda_.back()[o].score());
	    Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_.back()[o].score());
	  }
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_max)
	      Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_.back()[c].score());
	  
	  bool found = false;
	  double loss = 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_max)
	      for (size_type o = 0; o != kbest_oracle_size; ++ o) {
		state_type state_candidate = candidates.agenda_.back()[c];
		state_type state_oracle    = oracles.agenda_.back()[o];
		
		double error_max = 0.0;
		state_type state_candidate_max;
		state_type state_oracle_max;
		
		while (state_candidate && state_oracle) {
		  if (state_candidate.step() > state_oracle.step())
		    state_candidate = state_candidate.derivation();
		  else if (state_oracle.step() > state_candidate.step())
		    state_oracle = state_oracle.derivation();
		  else {
		    const double score_candidate = state_candidate.score();
		    const double score_oracle    = state_oracle.score();
		    
		    const bool suffered = score_candidate > score_oracle;
		    const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
		    
		    if (suffered && error > error_max) {
		      error_max = error;
		      state_candidate_max = state_candidate;
		      state_oracle_max    = state_oracle;
		    }
		    
		    state_candidate = state_candidate.derivation();
		    state_oracle    = state_oracle.derivation();
		  }
		}
		
		if (state_candidate_max && state_oracle_max) {
		  const double& score_candidate = candidates.agenda_.back()[c].score();
		  const double& score_oracle    = oracles.agenda_.back()[o].score();
		  
		  const weight_type prob_candidate = trance::semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
		  const weight_type prob_oracle    = trance::semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
		  
		  const double loss_factor = prob_candidate * prob_oracle;
		  
		  backward_[state_candidate_max].loss_ += loss_factor;
		  backward_[state_oracle_max].loss_    -= loss_factor;
		  
		  loss += error_max * loss_factor;
		  found = true;
		}
	      }
	  
	  if (! found) return 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_max)
	      states_[candidates.agenda_.back()[c].step()].insert(candidates.agenda_.back()[c]);
	  
	  for (size_type o = 0; o != kbest_oracle_size; ++ o)
	    states_[oracles.agenda_.back()[o].step()].insert(oracles.agenda_.back()[o]);
	  
	  return loss;
	} else {
	  state_type state_candidate = candidates.agenda_.back().back();
	  state_type state_oracle    = oracles.agenda_.back().back();
	  
	  double error_max = 0.0;
	  state_type state_candidate_max;
	  state_type state_oracle_max;
	  
	  while (state_candidate && state_oracle) {
	    if (state_candidate.step() > state_oracle.step())
	      state_candidate = state_candidate.derivation();
	    else if (state_oracle.step() > state_candidate.step())
	      state_oracle = state_oracle.derivation();
	    else {
	      const double score_candidate = state_candidate.score();
	      const double score_oracle    = state_oracle.score();
	      
	      const bool suffered = score_candidate > score_oracle;
	      const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	      
	      if (suffered && error > error_max) {
		error_max = error;
		state_candidate_max = state_candidate;
		state_oracle_max    = state_oracle;
	      }
	      
	      state_candidate = state_candidate.derivation();
	      state_oracle    = state_oracle.derivation();
	    }
	  }
	  
	  if (! state_candidate_max || ! state_oracle_max) return 0.0;
	  
	  backward_[state_candidate_max].loss_ += 1.0;
	  backward_[state_oracle_max].loss_    -= 1.0;
	  
	  states_[state_candidate_max.step()].insert(state_candidate_max);
	  states_[state_oracle_max.step()].insert(state_oracle_max);
	  
	  return error_max;
	}
      }
    };
  };
};

#endif
