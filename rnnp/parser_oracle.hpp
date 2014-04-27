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

      if (oracle_.oracle_.size() >= agenda_.size())
	throw std::runtime_error("oracle operation sequence is longer than agenda size!");
      
      operation_axiom(theta, output_agenda(agenda_));
      
      const size_type unary_max = oracle_.sentence_.size() * unary_size_;
      const size_type step_last = oracle_.sentence_.size() * 2 + unary_max;
      
      for (size_type step = 0; step != step_last; ++ step) {
	heap_type& heap = agenda_[step];
	
	if (heap.empty()) continue;

	heap_type::iterator hiter_begin = heap.begin();
	heap_type::iterator hiter       = heap.end();
	heap_type::iterator hiter_end   = heap.end();
	
	std::make_heap(hiter_begin, hiter_end, heap_compare());
	
	for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != beam_size_; -- hiter)
	  std::pop_heap(hiter_begin, hiter, heap_compare());
	
	// deallocate unused states
	for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter)
	  state_allocator_.deallocate(*iter);
	
	if (hiter != hiter_end)
	  best_action(step, *(hiter_end - 1));
	
	for (heap_type::iterator iter = hiter; iter != hiter_end; ++ iter) {
	  const state_type& state = *iter;
	  
	  if (state.operation().finished())
	    operation_idle(state, grammar.goal_, theta, output_agenda(agenda_));
	  else {
	    // implement!
	    if (step + 1 < oracle_.oracle_.size()) {
	      const oracle_type::item_type& item = oracle_.oracle_[step + 1];
	      
	      switch (item.operation_.operation()) {
	      case operation_type::SHIFT:
		if (state.next() >= oracle_.sentence_.size())
		  throw std::runtime_error("invalid shift!");

		operation_shift(state, item.head_, item.label_, theta, output_agenda(agenda_));
		break;
	      case operation_type::REDUCE:
		if (! state.stack() || state.stack().label() == symbol_type::EPSILON)
		  throw std::runtime_error("invalid reduction!");

		operation_reduce(state, item.label_, theta, output_agenda(agenda_));
		break;
	      case operation_type::UNARY:
		if(state.unary() >= unary_max || state.operation().closure() >= unary_size_)
		  throw std::runtime_error("invalid unary!");

		operation_unary(state, item.label_, theta, output_agenda(agenda_));
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
		operation_final(state, grammar.goal_, theta, output_agenda(agenda_));
	      else
		throw std::runtime_error("invalid final!");
	    }
	  }
	}
	
	// erase deallocated states
	heap.erase(hiter_begin, hiter);
      }
      
      // compute the final kbest derivations
      heap_type& heap = agenda_[step_last];

      if (heap.empty())
	throw std::runtime_error("no oracle parse?");
      else {
	heap_type::iterator hiter_begin = heap.begin();
	heap_type::iterator hiter       = heap.end();
	heap_type::iterator hiter_end   = heap.end();
	
	std::make_heap(hiter_begin, hiter_end, heap_compare());
	for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != kbest; -- hiter) {
	  std::pop_heap(hiter_begin, hiter, heap_compare());
	  
#if 1
	  if (! (hiter - 1)->operation().finished()) {
	    std::cerr << "non-final operation? " << std::endl;
	    std::cerr << "input size: " << oracle_.sentence_.size() << " input: " << oracle_.sentence_ << std::endl;
	    
	    state_type stack = *(hiter - 1);
	    while (stack) {
	      std::cerr << "\tstack step: " << stack.step()
			<< " op: " << stack.operation()
			<< " next: " << stack.next()
			<< " label: " << stack.label()
			<< " span: " << stack.span() << std::endl;
	      
	      stack = stack.stack();
	    }
	    
	    state_type curr = *(hiter - 1);
	    while (curr) {
	      std::cerr << "\tderivation step: " << curr.step()
			<< " op: " << curr.operation()
			<< " next: " << curr.next()
			<< " label: " << curr.label() 
			<< " span: " << curr.span() << std::endl;
	      
	      curr = curr.derivation();
	    }
	  }
#endif
	  
	  derivations.push_back(*(hiter - 1));
	}
	
	// deallocate unused states
	for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter)
	  state_allocator_.deallocate(*iter);
	
	if (hiter != hiter_end)
	  best_action(step_last, *(hiter_end - 1));
	
	// erase deallocated states
	heap.erase(hiter_begin, hiter);
      }
    }
    
  public:
    bool left_;
    
    oracle_type oracle_;
  };
};

#endif
