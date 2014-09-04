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
      size_type violation(const parser_type& candidates,
			  const parser_oracle_type& oracles,
			  const option_type& option)
      {
	double error_early = 0.0;
	size_type step_early = size_type(-1);
	size_type step_finished = oracles.agenda_.size() - 1;
	
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
	
	// check the state with early-violation
	for (size_type step = 0; step != step_finished + 1; ++ step) 
	  if (! candidates.agenda_[step].empty() && ! oracles.agenda_[step].empty()) {
	    
	    const double beam_candidate = candidates.agenda_[step].front().score();
	    const double beam_oracle    = oracles.agenda_[step].back().score();
	    
	    if (beam_candidate <= beam_oracle) continue;
	    
	    const double score_candidate = candidates.agenda_[step].back().score();
	    const double score_oracle    = oracles.agenda_[step].back().score();
	    
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
