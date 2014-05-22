// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__PARSER__HPP__
#define __RNNP__PARSER__HPP__ 1

#include <stdexcept>
#include <vector>
#include <queue>

#include <rnnp/parser/parser.hpp>

#include <rnnp/state.hpp>
#include <rnnp/allocator.hpp>
#include <rnnp/model_traits.hpp>

namespace rnnp
{
  class Parser
  {
  public:
    typedef parser::Parser::size_type       size_type;
    typedef parser::Parser::difference_type difference_type;

    typedef parser::Parser::span_type     span_type;
    typedef parser::Parser::sentence_type sentence_type;

    typedef parser::Parser::grammar_type   grammar_type;
    typedef parser::Parser::signature_type signature_type;
    
    typedef parser::Parser::model_type model_type;
    
    typedef model_type::symbol_type    symbol_type;
    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
    typedef parser::Parser::operation_type operation_type;
    typedef parser::Parser::state_type     state_type;

    typedef Allocator<state_type> state_allocator_type;
    
  public:
    // heap...
    typedef std::vector<state_type, std::allocator<state_type> > heap_type;
    typedef std::vector<heap_type, std::allocator<heap_type> > agenda_type;
    
    struct heap_compare
    {
      // compare by less so that better scored candidate preserved
      bool operator()(const state_type& x, const state_type& y) const
      {
	return x.score() < y.score();
      }
    };
    
    typedef std::vector<state_type, std::allocator<state_type> > derivation_set_type;
    
  public:
    Parser(size_type beam_size, size_type unary_size)
      : beam_size_(beam_size), unary_size_(unary_size) {}
    
  public:
    
    struct best_action_none
    {
      void operator()(const size_type& step, const state_type& state) const
      {
	
      }
    };
    
    template <typename Theta>
    void operator()(const sentence_type& input,
		    const grammar_type& grammar,
		    const signature_type& signature,
		    const Theta& theta,
		    const size_type kbest,
		    derivation_set_type& derivations)
    {
      parse(input, grammar, signature, theta, kbest, derivations, best_action_none());
    }

    template <typename Theta, typename BestAction>
    void operator()(const sentence_type& input,
		    const grammar_type& grammar,
		    const signature_type& signature,
		    const Theta& theta,
		    const size_type kbest,
		    derivation_set_type& derivations,
		    const BestAction& best_action)
    {
      parse(input, grammar, signature, theta, kbest, derivations, best_action);
    }
    
    template <typename Theta, typename BestAction>
    void parse(const sentence_type& input,
	       const grammar_type& grammar,
	       const signature_type& signature,
	       const Theta& theta,
	       const size_type kbest,
	       derivation_set_type& derivations,
	       const BestAction& best_action)
    {
      derivations.clear();
      
      if (input.empty()) return;
      
      initialize(input, theta);

      typename model_traits<Theta>::parser_type impl;
      
      impl.operation_axiom(*this, input, theta);
      
      const size_type unary_max = input.size() * unary_size_;
      const size_type step_last = input.size() * 2 + unary_max;
      
      // search
      for (size_type step = 0; step != step_last; ++ step) {
	heap_type& heap = agenda_[step];
	
	if (heap.empty()) break;
	
	prune(heap, beam_size_);
	
	// best_action
	best_action(step, heap.back());
	
	heap_type::const_iterator hiter_end = heap.end();
	for (heap_type::const_iterator hiter = heap.begin(); hiter != hiter_end; ++ hiter) {
	  const state_type& state = *hiter;
	  
	  if (state.operation().finished())
	    impl.operation_idle(*this, theta, state);
	  else {
	    // we perform shift..
	    if (state.next() < input.size()) {
	      const grammar_type::rule_set_type& rules = grammar.preterminal(signature, input[state.next()]);
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_shift(*this, theta, state, input[state.next()], riter->lhs_);
	    }
	    
	    // we perform unary
	    if (state.stack() && state.unary() < unary_max && state.operation().closure() < unary_size_) {
	      const grammar_type::rule_set_type& rules = grammar.unary(state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_unary(*this, theta, state, riter->lhs_);
	    }
	    
	    // final...
	    if (state.stack()
		&& state.stack().label() == symbol_type::EPSILON
		&& state.label() == grammar.goal_
		&& state.next() == input.size())
	      impl.operation_final(*this, theta, state);
	    
	    // we will perform reduce
	    if (state.stack() && state.stack().label() != symbol_type::EPSILON) {
	      const grammar_type::rule_set_type& rules = grammar.binary(state.stack().label(), state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_reduce(*this, theta, state, riter->lhs_);
	    }
	  }
	}
      }
      
      if (agenda_[step_last].empty()) {
	difference_type step_drop = step_last - 1;
	for (/**/; step_drop >= 0; -- step_drop)
	  if (! agenda_[step_drop].empty()) break;

	for (size_type step = step_drop; step != step_last; ++ step) {
	  heap_type& heap = agenda_[step];
	  
	  if (heap.empty()) break;
	  
	  if (step > step_drop) {
	    prune(heap, beam_size_);
	    
	    // best_action
	    best_action(step, heap.back());
	  }
	  
	  heap_type::const_iterator hiter_end = heap.end();
	  for (heap_type::const_iterator hiter = heap.begin(); hiter != hiter_end; ++ hiter) {
	    const state_type& state = *hiter;
	    
	    if (state.operation().finished())
	      impl.operation_idle(*this, theta, state);
	    else {
	      // we perform shift.... this should not happen, though..
	      if (state.next() < input.size()) {
		const grammar_type::rule_set_type& rules = grammar.preterminal(signature, input[state.next()]);
		
		grammar_type::rule_set_type::const_iterator riter_end = rules.end();
		for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		  impl.operation_shift(*this, theta, state, input[state.next()], riter->lhs_);
	      }
	      
	      // perform root
	      if (state.stack()
		  && state.stack().label() == symbol_type::EPSILON
		  && state.label() != grammar.goal_
		  && state.next() == input.size())
		impl.operation_unary(*this, theta, state, grammar.goal_);
	      
	      // final...
	      if (state.stack()
		  && state.stack().label() == symbol_type::EPSILON
		  && state.label() == grammar.goal_
		  && state.next() == input.size())
		impl.operation_final(*this, theta, state);
	      
	      // we will perform reduce
	      if (state.stack() && state.stack().label() != symbol_type::EPSILON) {
		if (state.stack().stack() && state.stack().stack().label() == symbol_type::EPSILON)
		  impl.operation_reduce(*this, theta, state, grammar.sentence_);
		else
		  impl.operation_reduce(*this, theta, state, grammar.sentence_binarized_);
	      }
	    }
	  }
	}
      }
      
      // compute the final kbest derivations
      heap_type& heap = agenda_[step_last];
      
      if (! heap.empty()) {
	prune(heap, kbest);
	
	best_action(step_last, heap.back());
	
	derivations.insert(derivations.end(), heap.rbegin(), heap.rend());
      }
    }
    
    void initialize(const sentence_type& input, const model_type& theta)
    {
      // # of operations is 2n + # of unary rules + final
      agenda_.clear();
      agenda_.resize(input.size() * 2 + input.size() * unary_size_ + 1);

      // state allocator
      state_allocator_.clear();
      state_allocator_.assign(state_type::size(theta.hidden_));
    }
    
    void prune(heap_type& heap, const size_type beam)
    {
      if (heap.empty()) return;

      heap_type::iterator hiter_begin = heap.begin();
      heap_type::iterator hiter       = heap.end();
      heap_type::iterator hiter_end   = heap.end();
      
      std::make_heap(hiter_begin, hiter_end, heap_compare());
      
      for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != beam; -- hiter)
	std::pop_heap(hiter_begin, hiter, heap_compare());
      
      // deallocate unused states
      for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter)
	state_allocator_.deallocate(*iter);
      
      // erase deallocated states
      heap.erase(hiter_begin, hiter);
    }

  public:
    size_type beam_size_;
    size_type unary_size_;
    
    agenda_type agenda_;
    
    // allocator
    state_allocator_type state_allocator_;
    
    // additional information required by some models...
    tensor_type queue_;
  };
};

#endif
