// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION__HPP__
#define __RNNP__OBJECTIVE__VIOLATION__HPP__ 1

// this is a violation..

#include <rnnp/objective/margin.hpp>

#include <rnnp/semiring/logprob.hpp>

namespace rnnp
{
  namespace objective
  {
    struct Violation : public objective::Margin
    {
      typedef rnnp::semiring::Logprob<double> weight_type;

      double heap_score_max(const heap_type& heap,
			    const heap_type& heap_goal) const
      {
	if (! heap.empty() && ! heap_goal.empty())
	  return std::max(heap.back().score(), heap_goal.back().score());
	else if (! heap.empty())
	  return heap.back().score();
	else
	  return heap_goal.back().score();
      }

      double heap_score_min(const heap_type& heap,
			    const heap_type& heap_goal) const
      {
	if (! heap.empty() && ! heap_goal.empty())
	  return std::min(heap.front().score(), heap_goal.front().score());
	else if (! heap.empty())
	  return heap.front().score();
	else
	  return heap_goal.front().score();
      }
      
      // virtual function
      virtual size_type violation(const model_type& theta,
				  const parser_type& candidates,
				  const parser_oracle_type& oracles,
				  const option_type& option) = 0;

      double margin(const state_type& state_candidate,
		    const state_type& state_oracle,
		    const weight_type& Z_candidate,
		    const weight_type& Z_oracle)
      {
	const double& score_candidate = state_candidate.score();
	const double& score_oracle    = state_oracle.score();
	
	const bool suffered = score_candidate > score_oracle;
	const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	
	if (! suffered || error <= 0.0) return 0.0;
	
	const weight_type prob_candidate = semiring::traits<weight_type>::exp(score_candidate) / Z_candidate;
	const weight_type prob_oracle    = semiring::traits<weight_type>::exp(score_oracle)    / Z_oracle;
	
	const double loss_factor = prob_candidate * prob_oracle;
	
	backward_type& backward_candidate = backward_[state_candidate];
	backward_type& backward_oracle    = backward_[state_oracle];
	
	backward_candidate.loss_ += loss_factor;
	backward_oracle.loss_    -= loss_factor;
	
	return error * loss_factor;
      }
      
      double margin(const model_type& theta,
		    const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option,
		    gradient_type& g)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	if (candidates.agenda_goal_.size() != oracles.agenda_goal_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");

	const size_type step = violation(theta, candidates, oracles, option);
	
	if (step == size_type(-1)) return 0.0;
	
	weight_type Z_candidate;
	weight_type Z_oracle;
	
	for (size_type c = 0; c != candidates.agenda_[step].size(); ++ c)
	  Z_candidate += rnnp::semiring::traits<weight_type>::exp(candidates.agenda_[step][c].score());
	for (size_type c = 0; c != candidates.agenda_goal_[step].size(); ++ c)
	  Z_candidate += rnnp::semiring::traits<weight_type>::exp(candidates.agenda_goal_[step][c].score());
	
	for (size_type o = 0; o != oracles.agenda_[step].size(); ++ o)
	  Z_oracle += rnnp::semiring::traits<weight_type>::exp(oracles.agenda_[step][o].score());
	for (size_type o = 0; o != oracles.agenda_goal_[step].size(); ++ o)
	  Z_oracle += rnnp::semiring::traits<weight_type>::exp(oracles.agenda_goal_[step][o].score());
	
	double loss = 0;
	
	for (size_type c = 0; c != candidates.agenda_[step].size(); ++ c) {
	  for (size_type o = 0; o != oracles.agenda_[step].size(); ++ o)
	    loss += margin(candidates.agenda_[step][c], oracles.agenda_[step][o], Z_candidate, Z_oracle);
	  
	  for (size_type o = 0; o != oracles.agenda_goal_[step].size(); ++ o)
	    loss += margin(candidates.agenda_[step][c], oracles.agenda_goal_[step][o], Z_candidate, Z_oracle);
	}
	
	for (size_type c = 0; c != candidates.agenda_goal_[step].size(); ++ c) {
	  for (size_type o = 0; o != oracles.agenda_[step].size(); ++ o)
	    loss += margin(candidates.agenda_goal_[step][c], oracles.agenda_[step][o], Z_candidate, Z_oracle);
	  
	  for (size_type o = 0; o != oracles.agenda_goal_[step].size(); ++ o)
	    loss += margin(candidates.agenda_goal_[step][c], oracles.agenda_goal_[step][o], Z_candidate, Z_oracle);
	}
	
	for (size_type c = 0; c != candidates.agenda_[step].size(); ++ c)
	  states_[candidates.agenda_[step][c].step()].insert(candidates.agenda_[step][c]);
	for (size_type c = 0; c != candidates.agenda_goal_[step].size(); ++ c)
	  states_[candidates.agenda_goal_[step][c].step()].insert(candidates.agenda_goal_[step][c]);
	
	for (size_type o = 0; o != oracles.agenda_[step].size(); ++ o)
	  states_[oracles.agenda_[step][o].step()].insert(oracles.agenda_[step][o]);
	for (size_type o = 0; o != oracles.agenda_goal_[step].size(); ++ o)
	  states_[oracles.agenda_goal_[step][o].step()].insert(oracles.agenda_goal_[step][o]);
	
	return loss;
      }
    };
  };
};

#endif
