// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TMT__OBJECTIVE__VIOLATION_MAX__HPP__
#define __TMT__OBJECTIVE__VIOLATION_MAX__HPP__ 1

// this is a max violation..

#include <tmt/objective/margin.hpp>

#include <tmt/semiring/logprob.hpp>

namespace tmt
{
  namespace objective
  {
    struct ViolationMax : public objective::Margin
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
      
	double error_max = 0.0;
	size_type step_max = size_type(-1);

	sentence_type cand;
      
	// check the state with max-violation
	for (size_type step = 0; step != oracles.agenda_.size(); ++ step) 
	  if (! candidates.agenda_[step].empty() && ! oracles.agenda_[step].empty()) {
#if 0
	    // do not check whether an oracle falling off the beam
	    const double beam_candidate = candidates.agenda_[step].front().state().score();
	    const double beam_oracle    = oracles.agenda_[step].back().state().score();
	    
	    if (beam_oracle >= beam_candidate) continue;
#endif

	    derivation_.assign(theta, candidates.agenda_[step].back());
	    
	    cand = derivation_.target_;
	    
	    derivation_.assign(theta, oracles.agenda_[step].back());
	    
	    if (cand == derivation_.target_) continue;
	    
	    const double score_candidate = candidates.agenda_[step].back().state().score();
	    const double score_oracle    = oracles.agenda_[step].back().state().score();
	    
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
	  Z_candidate += tmt::semiring::traits<weight_type>::exp(candidates.agenda_[step_max][c].state().score());
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  Z_oracle += tmt::semiring::traits<weight_type>::exp(oracles.agenda_[step_max][o].state().score());
	
	double loss = 0;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    const node_type& node_candidate = candidates.agenda_[step_max][c];
	    const node_type& node_oracle    = oracles.agenda_[step_max][o];
	    
	    const double& score_candidate = node_candidate.state().score();
	    const double& score_oracle    = node_oracle.state().score();
	    
	    const bool suffered = score_candidate > score_oracle;
	    const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	    
	    if (! suffered || error <= 0.0) continue;
	    
	    const weight_type prob_candidate = semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
	    const weight_type prob_oracle    = semiring::traits<weight_type>::exp(score_oracle) / Z_oracle;
	    
	    const double loss_factor = prob_candidate * prob_oracle;
	    
	    backward_type& backward_candidate = backward_[node_candidate.state()];
	    backward_type& backward_oracle    = backward_[node_oracle.state()];
	    
	    backward_candidate.loss_ += loss_factor;
	    backward_oracle.loss_    -= loss_factor;
	    
	    loss += error * loss_factor;
	  }
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c)
	  nodes_[candidates.agenda_[step_max][c].state().step()].insert(candidates.agenda_[step_max][c]);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  nodes_[oracles.agenda_[step_max][o].state().step()].insert(oracles.agenda_[step_max][o]);
	
	return loss;
      }
    };
  };
};

#endif
