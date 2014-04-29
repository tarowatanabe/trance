// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION_EARLY__HPP__
#define __RNNP__OBJECTIVE__VIOLATION_EARLY__HPP__ 1

// this is a max violation..

#include <rnnp/objective/violation.hpp>

namespace rnnp
{
  namespace objective
  {
    struct ViolationEarly : public objective::Violation
    {
      size_type violation(const model_type& theta,
			  const parser_type& candidates,
			  const parser_oracle_type& oracles,
			  const option_type& option)
      {
	double error_early = 0.0;
	size_type step_early = size_type(-1);
	
	// check the state with early-violation
	for (size_type step = 0; step != oracles.agenda_.size(); ++ step) {
	  
	  if (candidates.agenda_[step].empty() && candidates.agenda_goal_[step].empty()) continue;
	  if (oracles.agenda_[step].empty()    && oracles.agenda_goal_[step].empty()) continue;
	  
	  const double beam_candidate = heap_score_min(candidates.agenda_[step], candidates.agenda_goal_[step]);
	  const double beam_oracle    = heap_score_max(oracles.agenda_[step],    oracles.agenda_goal_[step]);
	  
	  if (beam_oracle >= beam_candidate) continue;
	  
	  const double score_candidate = heap_score_max(candidates.agenda_[step], candidates.agenda_goal_[step]);
	  const double score_oracle    = heap_score_max(oracles.agenda_[step],    oracles.agenda_goal_[step]);
	  
	  const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	  
	  if (error > 0.0) {
	    error_early = error;
	    step_early = step;
	    break;
	  }
	}
	
	return step_early;
      }
    };
  };
};

#endif
