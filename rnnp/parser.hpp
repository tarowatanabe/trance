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
  class Parser : public parser::Parser
  {
  public:
    typedef Allocator<state_type> state_allocator_type;
    
    struct feature_vector_allocator_type
    {
      typedef utils::chunk_vector<feature_vector_type,
				  1024 * 16 / sizeof(feature_vector_type),
				  std::allocator<feature_vector_type> > feature_vector_set_type;
      typedef std::vector<feature_vector_type*, std::allocator<feature_vector_type*> > cache_type;
      
      feature_vector_type* allocate()
      {
	if (! cache_.empty()) {
	  feature_vector_type* allocated = cache_.back();
	  cache_.pop_back();
	  allocated->clear();
	  return allocated;
	}
	
	feature_vectors_.push_back(feature_vector_type());
	return &feature_vectors_.back();
      }

      void deallocate(const feature_vector_type* vec)
      {
	cache_.push_back(const_cast<feature_vector_type*>(vec));
      }
      
      void clear()
      {
	feature_vectors_.clear();
	cache_.clear();
      }
      
      feature_vector_set_type feature_vectors_;
      cache_type cache_;
    };

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
    Parser(size_type beam_size, size_type unary_size, bool terminate_early=false)
      : beam_size_(beam_size), unary_size_(unary_size), terminate_early_(terminate_early) {}
    
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
		    const feature_set_type& feats,
		    const Theta& theta,
		    const size_type kbest,
		    derivation_set_type& derivations)
    {
      parse(input, grammar, signature, feats, theta, kbest, derivations, best_action_none());
    }

    template <typename Theta, typename BestAction>
    void operator()(const sentence_type& input,
		    const grammar_type& grammar,
		    const signature_type& signature,
		    const feature_set_type& feats,
		    const Theta& theta,
		    const size_type kbest,
		    derivation_set_type& derivations,
		    const BestAction& best_action)
    {
      parse(input, grammar, signature, feats, theta, kbest, derivations, best_action);
    }
    
    template <typename Theta, typename BestAction>
    void parse(const sentence_type& input,
	       const grammar_type& grammar,
	       const signature_type& signature,
	       const feature_set_type& feats,
	       const Theta& theta,
	       const size_type kbest,
	       derivation_set_type& derivations,
	       const BestAction& best_action)
    {
      derivations.clear();
      
      if (input.empty()) return;
      
      initialize(input, feats, theta);

      typename model_traits<Theta>::parser_type impl;
      
      impl.operation_axiom(*this, input, feats, theta);
      
      const size_type unary_max = input.size() * unary_size_;
      const size_type step_last = input.size() * 2 + unary_max;

      size_type step_finished = step_last;
      
      // search
      for (size_type step = 0; step != step_last; ++ step) {
	heap_type& heap = agenda_[step];
	
	if (heap.empty()) break;
	
	prune(heap, feats, beam_size_);
	
	// best_action
	best_action(step, heap.back());

	bool non_finished = false;
	
	heap_type::const_iterator hiter_end = heap.end();
	for (heap_type::const_iterator hiter = heap.begin(); hiter != hiter_end; ++ hiter) {
	  const state_type& state = *hiter;
	  
	  if (state.operation().finished())
	    impl.operation_idle(*this, feats, theta, state);
	  else {
	    non_finished = true;
	    
	    // we perform shift..
	    if (state.next() < input.size()) {
	      const grammar_type::rule_set_type& rules = grammar.preterminal(signature, input[state.next()]);
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_shift(*this, feats, theta, state, input[state.next()], riter->lhs_);
	    }
	    
	    // we perform unary
	    if (state.stack() && state.unary() < unary_max && state.operation().closure() < unary_size_) {
	      const grammar_type::rule_set_type& rules = grammar.unary(state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_unary(*this, feats, theta, state, riter->lhs_);
	    }
	    
	    // final...
	    if (state.stack()
		&& state.stack().label() == symbol_type::AXIOM
		&& state.label() == grammar.goal_
		&& state.next() == input.size())
	      impl.operation_final(*this, feats, theta, state);
	    
	    // we will perform reduce
	    if (state.stack() && state.stack().label() != symbol_type::AXIOM) {
	      const grammar_type::rule_set_type& rules = grammar.binary(state.stack().label(), state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		impl.operation_reduce(*this, feats, theta, state, riter->lhs_);
	    }
	  }
	}
	
	if (terminate_early_ && ! non_finished) {
	  step_finished = step;
	  break;
	}
      }
      
      if (step_finished == step_last && agenda_[step_last].empty()) {
	difference_type step_drop = step_last - 1;
	for (/**/; step_drop >= 0; -- step_drop)
	  if (! agenda_[step_drop].empty()) break;
	
	for (size_type step = step_drop; step != step_last; ++ step) {
	  heap_type& heap = agenda_[step];
	  
	  if (heap.empty()) break;
	  
	  if (step > step_drop) {
	    prune(heap, feats, beam_size_);
	    
	    // best_action
	    best_action(step, heap.back());
	  }
	  
	  bool non_finished = false;
	  
	  heap_type::const_iterator hiter_end = heap.end();
	  for (heap_type::const_iterator hiter = heap.begin(); hiter != hiter_end; ++ hiter) {
	    const state_type& state = *hiter;
	    
	    if (state.operation().finished())
	      impl.operation_idle(*this, feats, theta, state);
	    else {
	      non_finished = true;
	      
	      // we perform shift.... this should not happen, though..
	      if (state.next() < input.size()) {
		const grammar_type::rule_set_type& rules = grammar.preterminal(signature, input[state.next()]);
		
		grammar_type::rule_set_type::const_iterator riter_end = rules.end();
		for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		  impl.operation_shift(*this, feats, theta, state, input[state.next()], riter->lhs_);
	      }
	      
	      // perform root
	      if (state.stack()
		  && state.stack().label() == symbol_type::AXIOM
		  && state.label() != grammar.goal_
		  && state.next() == input.size())
		impl.operation_unary(*this, feats, theta, state, grammar.goal_);
	      
	      // final...
	      if (state.stack()
		  && state.stack().label() == symbol_type::AXIOM
		  && state.label() == grammar.goal_
		  && state.next() == input.size())
		impl.operation_final(*this, feats, theta, state);
	      
	      // we will perform reduce
	      if (state.stack() && state.stack().label() != symbol_type::AXIOM) {
		if (state.stack().stack() && state.stack().stack().label() == symbol_type::AXIOM)
		  impl.operation_reduce(*this, feats, theta, state, grammar.sentence_);
		else
		  impl.operation_reduce(*this, feats, theta, state, grammar.sentence_binarized_);
	      }
	    }
	  }
	  
	  if (terminate_early_ && ! non_finished) {
	    step_finished = step;
	    break;
	  }
	}
      }
      
      // compute the final kbest derivations
      if (step_finished != step_last) {
	const heap_type& heap = agenda_[step_finished];
	
	if (! heap.empty())
	  derivations.insert(derivations.end(), heap.rend() - utils::bithack::min(kbest, heap.size()), heap.rend());
      } else {
	heap_type& heap = agenda_[step_last];
	
	if (! heap.empty()) {
	  prune(heap, feats, kbest);
	  
	  best_action(step_last, heap.back());
	  
	  derivations.insert(derivations.end(), heap.rbegin(), heap.rend());
	}
      }
    }
    
    void initialize(const sentence_type& input, const feature_set_type& feats, const model_type& theta)
    {
      // # of operations is 2n + # of unary rules + final
      agenda_.clear();
      agenda_.resize(input.size() * 2 + input.size() * unary_size_ + 1);

      // state allocator
      state_allocator_.clear();
      state_allocator_.assign(state_type::size(theta.hidden_));
      
      // feature(s)
      const_cast<feature_set_type&>(feats).initialize();
      feature_vector_allocator_.clear();
    }
    
    void prune(heap_type& heap, const feature_set_type& feats, const size_type beam)
    {
      if (heap.empty()) return;

      heap_type::iterator hiter_begin = heap.begin();
      heap_type::iterator hiter       = heap.end();
      heap_type::iterator hiter_end   = heap.end();
      
      std::make_heap(hiter_begin, hiter_end, heap_compare());
      
      for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != beam; -- hiter)
	std::pop_heap(hiter_begin, hiter, heap_compare());
      
      // deallocate unused states
      for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter) {
	const_cast<feature_set_type&>(feats).deallocate(iter->feature_state());
	feature_vector_allocator_.deallocate(iter->feature_vector());
	state_allocator_.deallocate(*iter);
      }
      
      // erase deallocated states
      heap.erase(hiter_begin, hiter);
    }

  public:
    size_type beam_size_;
    size_type unary_size_;
    bool terminate_early_;
    
    agenda_type agenda_;
    
    // allocator
    state_allocator_type          state_allocator_;
    feature_vector_allocator_type feature_vector_allocator_;
    
    // additional information required by some models...
    tensor_type queue_;
    tensor_type buffer_;
  };
};

#endif
