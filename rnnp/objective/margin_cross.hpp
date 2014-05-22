// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN_CROSS__HPP__
#define __RNNP__OBJECTIVE__MARGIN_CROSS__HPP__ 1

#include <limits>

#include <rnnp/symbol.hpp>
#include <rnnp/span.hpp>
#include <rnnp/objective/margin.hpp>
#include <rnnp/semiring/log.hpp>

#include <utils/compact_set.hpp>

namespace rnnp
{
  namespace objective
  {
    //
    // compute margin by expected cross-bracketing from orcle!
    //
    
    struct MarginCross : public objective::Margin
    {
    public:
      typedef rnnp::semiring::Log<double> weight_type;
      
      typedef rnnp::Span   span_type;
      
      typedef utils::compact_set<span_type,
				 utils::unassigned<span_type>, utils::unassigned<span_type>,
				 utils::hashmurmur3<size_t>, std::equal_to<span_type>,
				 std::allocator<span_type> > span_set_type;
      typedef std::vector<span_set_type, std::allocator<span_set_type> > span_map_type;

      typedef std::vector<size_type, std::allocator<size_type> >     error_set_type;
      typedef std::vector<double, std::allocator<double> >           margin_set_type;
      typedef std::vector<weight_type, std::allocator<weight_type> > prob_set_type;
      
      span_map_type   span_;
      span_set_type   span_candidate_;
      error_set_type  error_;
      margin_set_type margin_;
      prob_set_type   prob_;

      void collect_statistics(state_type state,
			      span_set_type& span)
      {
	span.clear();
	
	while (state) {
	  switch (state.operation().operation()) {
	  case operation_type::REDUCE:
	  case operation_type::UNARY:
	    if (! state.label().binarized())
	      span.insert(state.span());
	    break;
	  default:
	    break;
	  }
	  state = state.derivation();
	}
      }
      
      double margin(const parser_type& candidates,
		    const parser_oracle_type& oracles,
		    const option_type& option)
      {
	if (candidates.agenda_.size() != oracles.agenda_.size())
	  throw std::runtime_error("invalid candidate and oracle pair");
	
	size_type step_back = candidates.agenda_.size() - 1;
	for (/**/; step_back > 0; -- step_back)
	  if (! candidates.agenda_[step_back].empty() && ! oracles.agenda_[step_back].empty()) break;
	
	// this should not happen...
	if (step_back == 0) return 0.0;
	
	const size_type kbest_candidate_size = candidates.agenda_[step_back].size();
	const size_type kbest_oracle_size    = oracles.agenda_[step_back].size();
	
	// first, collect oracle span stats...
	span_.clear();
	span_.resize(kbest_oracle_size);
	
	for (size_type o = 0; o != kbest_oracle_size; ++ o)
	  collect_statistics(oracles.agenda_[step_back][o], span_[o]);
	
	error_.clear();
	error_.resize(kbest_candidate_size, std::numeric_limits<size_type>::max());
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  collect_statistics(candidates.agenda_[step_back][c], span_candidate_);

	  for (size_type o = 0; o != kbest_oracle_size; ++ o) {
	    size_type error = 0;
	    
	    span_set_type::const_iterator oiter_begin = span_[o].begin();
	    span_set_type::const_iterator oiter_end   = span_[o].end();
	    
	    span_set_type::const_iterator citer_end = span_candidate_.end();
	    for (span_set_type::const_iterator citer = span_candidate_.begin(); citer != citer_end; ++ citer) {
	      
	      bool crossed = false;
	      for (span_set_type::const_iterator oiter = oiter_begin; oiter != oiter_end && ! crossed; ++ oiter)
		crossed = ((oiter->first_ < citer->first_ && citer->first_ < oiter->last_ && oiter->last_ < citer->last_)
			   || (citer->first_ < oiter->first_ && oiter->first_ < citer->last_ && citer->last_ < oiter->last_));
	      
	      error += crossed;
	    }
	    
	    // F-measure
	    error_[c] = utils::bithack::min(error_[c], error);
	  }
	}
	
	weight_type Z;
	
	margin_.clear();
	margin_.resize(kbest_candidate_size);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const double margin = candidates.agenda_[step_back][c].score() * option.scale_;
	  
	  margin_[c] = margin;
	  Z += semiring::traits<weight_type>::exp(margin);
	}
	
	weight_type expectation;
      
	prob_.clear();
	prob_.resize(kbest_candidate_size);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const weight_type prob = semiring::traits<weight_type>::exp(margin_[c]) / Z;
	  
	  prob_[c] = prob;
	  expectation += double(error_[c]) * prob;
	}
	
	const double objective = expectation;
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  const weight_type loss = (error_[c] - expectation) * prob_[c];
	  
	  if (loss == weight_type()) continue;
	  
	  backward_[candidates.agenda_[step_back][c]].loss_ += loss;
	  states_[candidates.agenda_[step_back][c].step()].insert(candidates.agenda_[step_back][c]);
	}
	
	return objective;
      }
    };
  };
};

#endif

