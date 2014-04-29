// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION_LATE__HPP__
#define __RNNP__OBJECTIVE__VIOLATION_LATE__HPP__ 1

// this is a max violation..

#include <rnnp/objective/violation.hpp>

namespace rnnp
{
  namespace objective
  {
    struct ViolationLate : public objective::Violation
    {
      size_type violation(const model_type& theta,
			  const parser_type& candidates,
			  const parser_oracle_type& oracles,
			  const option_type& option)
      {
	double error_late = 0.0;
	size_type step_late = size_type(-1);
	
	// check the state with late-violation
	for (difference_type step = oracles.agenda_.size() - 1; step >= 0; -- step) {
	  
	  if (candidates.agenda_[step].empty() && candidates.agenda_goal_[step].empty()) continue;
	  if (oracles.agenda_[step].empty()    && oracles.agenda_goal_[step].empty()) continue;
	  
	  const double beam_candidate = heap_score_min(candidates.agenda_[step], candidates.agenda_goal_[step]);
	  const double beam_oracle    = heap_score_max(oracles.agenda_[step],    oracles.agenda_goal_[step]);
	  
	  if (beam_oracle >= beam_candidate) continue;
	  
	  const double score_candidate = heap_score_max(candidates.agenda_[step], candidates.agenda_goal_[step]);
	  const double score_oracle    = heap_score_max(oracles.agenda_[step],    oracles.agenda_goal_[step]);
	  
	  const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	  
	  if (error > 0.0) {
	    error_late = error;
	    step_late = step;
	    break;
	  }
	}
	
	return step_late;
      }
    };
  };
};

#endif
