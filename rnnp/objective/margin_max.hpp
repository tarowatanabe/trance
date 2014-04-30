// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN_MAX__HPP__
#define __RNNP__OBJECTIVE__MARGIN_MAX__HPP__ 1

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct MarginMax : public objective::Margin
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

	size_type step_back = candidates.agenda_.size() - 1;
	for (/**/; step_back > 0; -- step_back)
	  if (! candidates.agenda_[step_back].empty() && ! oracles.agenda_[step_back].empty()) break;
	
	// this should not happen...
	if (step_back == 0) return 0.0;
	
	const size_type kbest_candidate_size = candidates.agenda_[step_back].size();
	const size_type kbest_oracle_size    = oracles.agenda_[step_back].size();
	
	weight_type Z_candidate;
	weight_type Z_oracle;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_[step_back][c].score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_[step_back][o].score());
      
	bool found = false;
	double loss = 0.0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    state_type state_candidate = candidates.agenda_[step_back][c];
	    state_type state_oracle    = oracles.agenda_[step_back][o];
	      
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
	      const double& score_candidate = candidates.agenda_[step_back][c].score();
	      const double& score_oracle    = oracles.agenda_[step_back][o].score();
	      
	      const weight_type prob_candidate = rnnp::semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
	      const weight_type prob_oracle    = rnnp::semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
		
	      const double loss_factor = prob_candidate * prob_oracle;
		
	      backward_type& backward_candidate = backward_[state_candidate_max];
	      backward_type& backward_oracle    = backward_[state_oracle_max];
		
	      backward_candidate.loss_ += loss_factor;
	      backward_oracle.loss_    -= loss_factor;
	      
	      loss += error_max * loss_factor;
	      found = true;
	    }
	  }
	
	if (! found) return 0.0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  states_[candidates.agenda_[step_back][c].step()].insert(candidates.agenda_[step_back][c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  states_[oracles.agenda_[step_back][o].step()].insert(oracles.agenda_[step_back][o]);
	
	return loss;
      }
    };
  };
};

#endif
