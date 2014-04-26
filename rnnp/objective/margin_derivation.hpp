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
      
      double margin(const model_type& theta,
		    const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option,
		    gradient_type& g)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");

	const size_type kbest_candidate_size = candidates.agenda_.back().size();
	const size_type kbest_oracle_size    = oracles.agenda_.back().size();
	
	if (! kbest_candidate_size || ! kbest_oracle_size)
	  return 0.0;
	
	weight_type Z_candidate;
	weight_type Z_oracle;

	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_.back()[c].score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_.back()[o].score());
	
	bool found = false;
	double loss = 0.0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
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
	    
	    backward_type& backward_candidate = backward_[state_candidate];
	    backward_type& backward_oracle    = backward_[state_oracle];
	      
	    backward_candidate.loss_ += loss_factor;
	    backward_oracle.loss_    -= loss_factor;
	      
	    loss += error * loss_factor;
	    found = true;
	  }
	
	if (! found) return 0.0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  states_[candidates.agenda_.back()[c].step()].insert(candidates.agenda_.back()[c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  states_[oracles.agenda_.back()[o].step()].insert(oracles.agenda_.back()[o]);
	
	return loss;
      }
    };
  };
};

#endif

