// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TMT__OBJECTIVE__MARGIN_EARLY__HPP__
#define __TMT__OBJECTIVE__MARGIN_EARLY__HPP__ 1

#include <tmt/objective/margin.hpp>

#include <tmt/semiring/logprob.hpp>

namespace tmt
{
  namespace objective
  {
    struct MarginEarly : public objective::Margin
    {
      typedef tmt::semiring::Logprob<double> weight_type;
      
      double margin(const bitext_type& bitext,
		    const model_type& theta,
		    const decoder_type& candidates,
		    const decoder_type& oracles,
		    const option_type& option,
		    gradient_type& g)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	
	const size_type kbest_candidate_size = candidates.agenda_.back().size();
	const size_type kbest_oracle_size    = oracles.agenda_.back().size();
	
	if (! kbest_candidate_size || ! kbest_oracle_size)
	  return 0.0;
	
	if (! std::count(error_.begin(), error_.end(), true))
	  return 0.0;
	
	weight_type Z_candidate;
	weight_type Z_oracle;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  if (error_[c])
	    Z_candidate += semiring::traits<weight_type>::exp(candidates.agenda_.back()[c].state().score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += semiring::traits<weight_type>::exp(oracles.agenda_.back()[o].state().score());
	
	bool found = false;
	double loss = 0.0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  if (error_[c])
	    for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	      node_type node_candidate = candidates.agenda_.back()[c];
	      node_type node_oracle    = oracles.agenda_.back()[o];
	      
	      double error_early = 0.0;
	      node_type node_candidate_early;
	      node_type node_oracle_early;
	      
	      while (node_candidate && node_oracle) {
		if (node_candidate.state().step() > node_oracle.state().step())
		  node_candidate = node_candidate.derivation();
		else if (node_oracle.state().step() > node_candidate.state().step())
		  node_oracle = node_oracle.derivation();
		else {
		  const double score_candidate = node_candidate.state().score();
		  const double score_oracle    = node_oracle.state().score();
		  
		  const bool suffered = score_candidate > score_oracle;
		  const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
		  
		  if (suffered && error > 0.0) {
		    error_early = error;
		    node_candidate_early = node_candidate;
		    node_oracle_early    = node_oracle;
		  }
		  
		  node_candidate = node_candidate.derivation();
		  node_oracle    = node_oracle.derivation();
		}
	      }
	      
	      if (node_candidate_early && node_oracle_early) {
		const double& score_candidate = candidates.agenda_.back()[c].state().score();
		const double& score_oracle    = oracles.agenda_.back()[o].state().score();
		
		const weight_type prob_candidate = tmt::semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
		const weight_type prob_oracle    = tmt::semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;

		const double loss_factor = prob_candidate * prob_oracle;
	      
		backward_type& backward_candidate = backward_[node_candidate_early.state()];
		backward_type& backward_oracle    = backward_[node_oracle_early.state()];
	      
		backward_candidate.loss_ += loss_factor;
		backward_oracle.loss_    -= loss_factor;
	      
		loss += error_early * loss_factor;
		found = true;
	      }
	    }

	if (! found) return 0.0;

	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  if (error_[c])
	    nodes_[candidates.agenda_.back()[c].state().step()].insert(candidates.agenda_.back()[c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  nodes_[oracles.agenda_.back()[o].state().step()].insert(oracles.agenda_.back()[o]);
      
	return loss;
      }
    };
  };
};

#endif
