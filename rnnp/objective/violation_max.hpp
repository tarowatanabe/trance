// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__VIOLATION_MAX__HPP__
#define __RNNP__OBJECTIVE__VIOLATION_MAX__HPP__ 1

// this is a max violation..

#include <rnnp/objective/violation.hpp>

namespace rnnp
{
  namespace objective
  {

    struct ViolationMax : public objective::Violation
    {
      
      size_type violation(const model_type& theta,
			  const parser_type& candidates,
			  const parser_oracle_type& oracles,
			  const option_type& option)
      {
	double error_max = 0.0;
	size_type step_max = size_type(-1);
	
	// check the state with max-violation
	for (size_type step = 0; step != oracles.agenda_.size(); ++ step) {
	  
	  if (candidates.agenda_[step].empty() && candidates.agenda_goal_[step].empty()) continue;
	  if (oracles.agenda_[step].empty()    && oracles.agenda_goal_[step].empty()) continue;
	  
	  const double score_candidate = heap_score_max(candidates.agenda_[step], candidates.agenda_goal_[step]);
	  const double score_oracle    = heap_score_max(oracles.agenda_[step],    oracles.agenda_goal_[step]);
	  
	  if (score_oracle >= score_candidate) continue;
	  
	  const double error = std::max(1.0 - (score_oracle - score_candidate), 0.0);
	  
	  if (error > error_max) {
	    error_max = error;
	    step_max = step;
	  }
	}
	
	return step_max;
      }
    };
  };
};

#endif
