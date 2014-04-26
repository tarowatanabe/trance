// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION_LATE__HPP__
#define __RNNP__OBJECTIVE__VIOLATION_LATE__HPP__ 1

// this is a max violation..

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct ViolationLate : public objective::Margin
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
	
	double error_late = 0.0;
	size_type step_late = size_type(-1);
	
	// check the state with late-violation
	for (difference_type step = oracles.agenda_.size() - 1; step >= 0; -- step)
	  if (! candidates.agenda_[step].empty() && ! oracles.agenda_[step].empty()) {
	    const double beam_candidate = candidates.agenda_[step].front().score();
	    const double beam_oracle    = oracles.agenda_[step].back().score();
	    
	    if (beam_oracle >= beam_candidate) continue;
	    
	    const double score_candidate = candidates.agenda_[step].back().score();
	    const double score_oracle    = oracles.agenda_[step].back().score();
	    
	    const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	    
	    if (error > 0.0) {
	      error_late = error;
	      step_late = step;
	      break;
	    }
	  }
	
	if (step_late == size_type(-1)) return 0.0;
	
	const size_type kbest_candidate_size = candidates.agenda_[step_late].size();
	const size_type kbest_oracle_size    = oracles.agenda_[step_late].size();
	
	weight_type Z_candidate;
	weight_type Z_oracle;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  Z_candidate += rnnp::semiring::traits<weight_type>::exp(candidates.agenda_[step_late][c].score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += rnnp::semiring::traits<weight_type>::exp(oracles.agenda_[step_late][o].score());
	
	double loss = 0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    const state_type& state_candidate = candidates.agenda_[step_late][c];
	    const state_type& state_oracle    = oracles.agenda_[step_late][o];
	    
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
	  states_[candidates.agenda_[step_late][c].step()].insert(candidates.agenda_[step_late][c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  states_[oracles.agenda_[step_late][o].step()].insert(oracles.agenda_[step_late][o]);
	
	return loss;
      }
    };
  };
};

#endif
