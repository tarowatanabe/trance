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
	  
	  bool violated = false;
	  
	  if (! oracles.agenda_[step].empty() && ! oracles.agenda_goal_[step].empty()) {
	    if (candidates.agenda_[step].empty())
	      violated |= true;
	    else {
	      const double beam_candidate = candidates.agenda_[step].front().score();
	      const double beam_oracle    = oracles.agenda_[step].back().score();
	      
	      violated |= (beam_candidate > beam_oracle);
	    }
	    
	    if (candidates.agenda_goal_[step].empty())
	      violated |= true;
	    else {
	      const double beam_candidate = candidates.agenda_goal_[step].front().score();
	      const double beam_oracle    = oracles.agenda_goal_[step].back().score();
	      
	      violated |= (beam_candidate > beam_oracle);
	    }
	    	    
	  } else if (! oracles.agenda_[step].empty()) {
	    if (candidates.agenda_[step].empty())
	      violated |= true;
	    else {
	      const double beam_candidate = candidates.agenda_[step].front().score();
	      const double beam_oracle    = oracles.agenda_[step].back().score();
	      
	      violated |= (beam_candidate > beam_oracle);
	    }
	  } else {
	    if (candidates.agenda_goal_[step].empty())
	      violated |= true;
	    else {
	      const double beam_candidate = candidates.agenda_goal_[step].front().score();
	      const double beam_oracle    = oracles.agenda_goal_[step].back().score();
	      
	      violated |= (beam_candidate > beam_oracle);
	    }
	  }
	  
	  if (! violated) continue;
	  
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
