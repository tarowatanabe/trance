// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN_DERIVATION__HPP__
#define __RNNP__OBJECTIVE__MARGIN_DERIVATION__HPP__ 1

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct MarginDerivation : public objective::Margin
    {
      typedef rnnp::semiring::Logprob<double> weight_type;
      
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

	  double score_min = std::numeric_limits<double>::infinity();
	  
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    score_min = std::min(score_min, oracles.agenda_.back()[o].score());
	    Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_.back()[o].score());
	  }
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_min)
	      Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_.back()[c].score());
	  
	  bool found = false;
	  double loss = 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_min) 
	      for (size_type o = 0; o != kbest_oracle_size; ++ o) {
		const state_type& state_candidate = candidates.agenda_.back()[c];
		const state_type& state_oracle    = oracles.agenda_.back()[o];
		
		const double& score_candidate = state_candidate.score();
		const double& score_oracle    = state_oracle.score();
		
		const bool suffered = score_candidate > score_oracle;
		const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
		
		if (! suffered || error <= 0.0) continue;
		
		const weight_type prob_candidate = rnnp::semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
		const weight_type prob_oracle    = rnnp::semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
		
		const double loss_factor = prob_candidate * prob_oracle;
		
		backward_[state_candidate].loss_ += loss_factor;
		backward_[state_oracle].loss_    -= loss_factor;
		
		loss += error * loss_factor;
		found = true;
	      }
	  
	  if (! found) return 0.0;
	  
	  for (size_type c = 0; c != kbest_candidate_size; ++ c)
	    if (candidates.agenda_.back()[c].score() > score_min)
	      states_[candidates.agenda_.back()[c].step()].insert(candidates.agenda_.back()[c]);
	  
	  for (size_type o = 0; o != kbest_oracle_size; ++ o)
	    states_[oracles.agenda_.back()[o].step()].insert(oracles.agenda_.back()[o]);
	  
	  return loss;
	} else {
	  const state_type& state_candidate = candidates.agenda_.back().back();
	  const state_type& state_oracle    = oracles.agenda_.back().back();
	  
	  backward_[state_candidate].loss_ += 1.0;
	  backward_[state_oracle].loss_    -= 1.0;
	  
	  states_[state_candidate.step()].insert(state_candidate);
	  states_[state_oracle.step()].insert(state_oracle);
	  
	  return std::max(1.0 - (state_oracle.score() - state_candidate.score()), 0.0);
	}
      }
    };
  };
};

#endif

