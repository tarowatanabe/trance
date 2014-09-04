// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN_LATE__HPP__
#define __RNNP__OBJECTIVE__MARGIN_LATE__HPP__ 1

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct MarginLate : public objective::Margin
    {
      typedef rnnp::semiring::Logprob<double> weight_type;
      
      double margin(const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	
	size_type step_finished = candidates.agenda_.size() - 1;
	for (size_type step = 0; step != oracles.agenda_.size(); ++ step)
	  if (! candidates.agenda_[step].empty() && ! oracles.agenda_[step].empty()) {
	    bool non_finished = false;
	    
	    for (size_type o = 0; o != oracles.agenda_[step].size(); ++ o)
	      non_finished |= ! oracles.agenda_[step][o].operation().finished();
	    
	    for (size_type c = 0; c != candidates.agenda_[step].size(); ++ c)
	      non_finished |= ! candidates.agenda_[step][c].operation().finished();
	    
	    if (! non_finished) {
	      step_finished = step;
	      break;
	    }
	  }
	
	// this should not happen...
	if (step_finished == 0) return 0.0;
	
	if (option.margin_all_) {
	  const size_type kbest_candidate_size = candidates.agenda_[step_finished].size();
	  const size_type kbest_oracle_size    = oracles.agenda_[step_finished].size();
	  
	  weight_type Z_candidate;
	  weight_type Z_oracle;

	  double score_min = std::numeric_limits<double>::infinity();

	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    score_min = std::min(score_min, oracles.agenda_[step_finished][o].score());
	    Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_[step_finished][o].score());
	  }
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_[step_finished][c].score() > score_min)
	      Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_[step_finished][c].score());
	  
	  bool found = false;
	  double loss = 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_[step_finished][c].score() > score_min)
	      for (size_type o = 0; o != kbest_oracle_size; ++ o) {
		state_type state_candidate = candidates.agenda_[step_finished][c];
		state_type state_oracle    = oracles.agenda_[step_finished][o];
		
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
		    
		    if (suffered && error > 0.0) {
		      const double& score_candidate = candidates.agenda_[step_finished][c].score();
		      const double& score_oracle    = oracles.agenda_[step_finished][o].score();
		      
		      const weight_type prob_candidate = rnnp::semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
		      const weight_type prob_oracle    = rnnp::semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
		      
		      const double loss_factor = prob_candidate * prob_oracle;
		      
		      backward_[state_candidate].loss_ += loss_factor;
		      backward_[state_oracle].loss_    -= loss_factor;
		      
		      loss += error * loss_factor;
		      found = true;
		      break;
		    }
		    
		    state_candidate = state_candidate.derivation();
		    state_oracle    = state_oracle.derivation();
		  }
		}
	      }
	  
	  if (! found) return 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_[step_finished][c].score() > score_min)
	      states_[candidates.agenda_[step_finished][c].step()].insert(candidates.agenda_[step_finished][c]);
	  
	  for (size_type o = 0; o != kbest_oracle_size; ++ o)
	    states_[oracles.agenda_[step_finished][o].step()].insert(oracles.agenda_[step_finished][o]);
	  
	  return loss;
	} else {
	  state_type state_candidate = candidates.agenda_[step_finished].back();
	  state_type state_oracle    = oracles.agenda_[step_finished].back();
	  
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
	      
	      if (suffered && error > 0.0) {
		backward_[state_candidate].loss_ += 1.0;
		backward_[state_oracle].loss_    -= 1.0;
		
		states_[state_candidate.step()].insert(state_candidate);
		states_[state_oracle.step()].insert(state_oracle);
		
		return error;
	      }
	      
	      state_candidate = state_candidate.derivation();
	      state_oracle    = state_oracle.derivation();
	    }
	  }
	  
	  return 0.0;
	}
      }
    };
  };
};

#endif
