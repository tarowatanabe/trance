// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__PARSER_ORACLE__HPP__
#define __RNNP__PARSER_ORACLE__HPP__ 1

#include <rnnp/parser.hpp>
#include <rnnp/oracle.hpp>
#include <rnnp/tree.hpp>

namespace rnnp
{
  class ParserOracle : public Parser
  {
  public:
    typedef Tree   tree_type;
    typedef Oracle oracle_type;

  public:
    ParserOracle(size_type beam_size, size_type unary_size, bool left)
      : Parser(beam_size, unary_size), left_(left) {}
    
  public:
    void operator()(const tree_type& input,
		    const grammar_type& grammar,
		    const model_type& theta,
		    const size_type kbest,
		    derivation_set_type& derivations)
    {
      parse(input, grammar, theta, kbest, derivations, best_action_none());
    }

    template <typename BestAction>
    void operator()(const tree_type& input,
		    const grammar_type& grammar,
		    const model_type& theta,
		    const size_type kbest,
		    derivation_set_type& derivations,
		    const BestAction& best_action)
    {
      parse(input, grammar, theta, kbest, derivations, best_action);
    }
    
    template <typename BestAction>
    void parse(const tree_type& input,
	       const grammar_type& grammar,
	       const model_type& theta,
	       const size_type kbest,
	       derivation_set_type& derivations,
	       const BestAction& best_action)
    {
      derivations.clear();
      
      oracle_.assign(input, left_);
      
      if (oracle_.sentence_.empty()) return;
      
      initialize(oracle_.sentence_, grammar, theta);

      if (oracle_.actions_.size() >= agenda_.size())
	throw std::runtime_error("oracle operation sequence is longer than agenda size!");
      
      operation_axiom(theta);
      
      const size_type beam_goal = utils::bithack::max(beam_size_, kbest);
      const size_type unary_max = oracle_.sentence_.size() * unary_size_;
      const size_type step_last = oracle_.sentence_.size() * 2 + unary_max;
      
      for (size_type step = 0; step != step_last; ++ step) {
	heap_type& heap      = agenda_[step];
	heap_type& heap_goal = agenda_goal_[step];
	
	if (heap.empty() && heap_goal.empty()) break;
	
	prune(heap,      beam_size_);
	prune(heap_goal, beam_goal);
	
	// best_action
	if (! heap.empty() && ! heap_goal.empty()) {
	  if (heap_goal.back().score() >= heap.back().score())
	    best_action(step, heap_goal.back());
	  else
	    best_action(step, heap.back());
	} else if (! heap.empty())
	  best_action(step, heap.back());
	else
	  best_action(step, heap_goal.back());
	
	// process goal items
	heap_type::const_iterator giter_end = heap_goal.end();
	for (heap_type::const_iterator giter = heap_goal.begin(); giter != giter_end; ++ giter)
	  operation_idle(*giter, grammar.goal_, theta);
	
	// process others
	heap_type::const_iterator hiter_end = heap.end();
	for (heap_type::const_iterator hiter = heap.begin(); hiter != hiter_end; ++ hiter) {
	  const state_type& state = *hiter;
	  
	  if (step + 1 < oracle_.actions_.size()) {
	    const oracle_type::action_type& action = oracle_.actions_[step + 1];
	    
	    switch (action.operation_.operation()) {
	    case operation_type::SHIFT:
	      if (state.next() >= oracle_.sentence_.size())
		throw std::runtime_error("invalid shift!");
	      
	      operation_shift(state, action.head_, action.label_, theta);
	      break;
	    case operation_type::REDUCE:
	      if (! state.stack() || state.stack().label() == symbol_type::EPSILON)
		throw std::runtime_error("invalid reduction!");
	      
	      operation_reduce(state, action.label_, theta);
	      break;
	    case operation_type::UNARY:
	      if(state.unary() >= unary_max || state.operation().closure() >= unary_size_)
		throw std::runtime_error("invalid unary!");
	      
	      operation_unary(state, action.label_, theta);
	      break;
	    default:
	      throw std::runtime_error("invalid operation!");
	    }
	  } else {
	    // final...
	    if (state.stack()
		&& state.stack().label() == symbol_type::EPSILON
		&& state.label() == grammar.goal_
		&& state.next() == oracle_.sentence_.size())
	      operation_final(state, grammar.goal_, theta);
	    else
	      throw std::runtime_error("invalid final!");
	  }
	}
      }
      
      // compute the final kbest derivations
      if (! agenda_[step_last].empty())
	throw std::runtime_error("why non-final item exists in the last beam..?");

      heap_type& heap_goal = agenda_goal_[step_last];
      
      if (! heap_goal.empty()) {
	prune(heap_goal, kbest);
	
	derivations.insert(derivations.end(), heap_goal.rbegin(), heap_goal.rend());
      }
    }
    
  public:
    bool left_;
    
    oracle_type oracle_;
  };
};

#endif
