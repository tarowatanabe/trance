// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__PARSER__HPP__
#define __RNNP__PARSER__HPP__ 1

#include <stdexcept>
#include <vector>
#include <queue>

#include <rnnp/sentence.hpp>
#include <rnnp/symbol.hpp>
#include <rnnp/model.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/operation.hpp>
#include <rnnp/span.hpp>
#include <rnnp/state.hpp>
#include <rnnp/allocator.hpp>

namespace rnnp
{
  class Parser
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef Span span_type;
    
    typedef Sentence  sentence_type;
    
    typedef Grammar grammar_type;

    typedef grammar_type::rule_type rule_type;

    typedef Model model_type;
    
    typedef model_type::symbol_type    symbol_type;
    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
    typedef Operation operation_type;
    
    typedef State state_type;

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
    
    void operator()(const sentence_type& input,
		    const grammar_type& grammar,
		    const model_type& theta,
		    const size_type kbest,
		    derivation_set_type& derivations)
    {
      parse(input, grammar, theta, kbest, derivations, best_action_none());
    }

    template <typename BestAction>
    void operator()(const sentence_type& input,
		    const grammar_type& grammar,
		    const model_type& theta,
		    const size_type kbest,
		    derivation_set_type& derivations,
		    const BestAction& best_action)
    {
      parse(input, grammar, theta, kbest, derivations, best_action);
    }
    
    template <typename BestAction>
    void parse(const sentence_type& input,
	       const grammar_type& grammar,
	       const model_type& theta,
	       const size_type kbest,
	       derivation_set_type& derivations,
	       const BestAction& best_action)
    {
      derivations.clear();
      
      if (input.empty()) return;
      
      initialize(input, grammar, theta);
      
      operation_axiom(theta);
      
      const size_type unary_max = input.size() * unary_size_;
      const size_type step_last = input.size() * 2 + unary_max;
      
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
	    operation_idle(state, grammar.goal_, theta);
	  else {
	    // we perform shift..
	    if (state.next() < input.size()) {
	      const grammar_type::rule_set_type& rules = grammar.preterminal(input[state.next()]);
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		operation_shift(state, input[state.next()], riter->lhs_, theta);
	    }
	    
	    // we perform unary
	    if (state.unary() < unary_max && state.operation().closure() < unary_size_) {
	      const grammar_type::rule_set_type& rules = grammar.unary(state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		operation_unary(state, riter->lhs_, theta);
	    }
	    
	    // final...
	    if (state.stack()
		&& state.stack().label() == symbol_type::EPSILON
		&& state.label() == grammar.goal_
		&& state.next() == input.size())
	      operation_final(state, grammar.goal_, theta);
	    
	    // we will perform reduce
	    if (state.stack() && state.stack().label() != symbol_type::EPSILON) {
	      const grammar_type::rule_set_type& rules = grammar.binary(state.stack().label(), state.label());
	      
	      grammar_type::rule_set_type::const_iterator riter_end = rules.end();
	      for (grammar_type::rule_set_type::const_iterator riter = rules.begin(); riter != riter_end; ++ riter)
		operation_reduce(state, riter->lhs_, theta);
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
    
    void initialize(const sentence_type& input, const grammar_type& grammar, const model_type& theta)
    {
      // # of operations is 2n + # of unary rules + final
      agenda_.clear();
      agenda_.resize(input.size() * 2 + input.size() * unary_size_ + 1);

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
    
    void operation_shift(const state_type& state,
			 const word_type& head,
			 const symbol_type& label,
			 const model_type& theta)
    {
      const size_type offset1 = 0;
      const size_type offset2 = theta.hidden_;

      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = state.step() + 1;
      state_new.next()  = state.next() + 1;
      state_new.unary() = state.unary();
      
      state_new.operation() = operation_type::SHIFT;
      state_new.label()     = label;
      state_new.head()      = head;
      state_new.span()      = span_type(state.next(), state.next() + 1);
      
      state_new.stack()      = state;
      state_new.derivation() = state;
      state_new.reduced()    = state_type();
      
      const size_type offset_grammar        = theta.offset_grammar(label);
      const size_type offset_classification = theta.offset_classification(label);
      
      state_new.layer(theta.hidden_) = (theta.Bsh_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wsh_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_)
					   * state.layer(theta.hidden_))
					+ (theta.Wsh_.block(offset_grammar, offset2, theta.hidden_, theta.embedding_)
					   * theta.terminal_.col(theta.terminal(head)))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = state.score() + score;
      
      agenda_[state_new.step()].push_back(state_new);
    }
    
    void operation_reduce(const state_type& state,
			  const symbol_type& label,
			  const model_type& theta)
    {
      const size_type offset1 = 0;
      const size_type offset2 = theta.hidden_;
      
      const state_type state_reduced = state.stack();
      const state_type state_stack   = state_reduced.stack();

      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = state.step() + 1;
      state_new.next()  = state.next();
      state_new.unary() = state.unary();
      
      state_new.operation() = operation_type::REDUCE;
      state_new.label()     = label;
      state_new.head()      = symbol_type::EPSILON;
      state_new.span()      = span_type(state_reduced.span().first_, state.span().last_);
      
      state_new.stack()      = state_stack;
      state_new.derivation() = state;
      state_new.reduced()    = state_reduced;
      
      const size_type offset_grammar        = theta.offset_grammar(label);
      const size_type offset_classification = theta.offset_classification(label);
      
      state_new.layer(theta.hidden_) = (theta.Bre_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wre_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_)
					   * state.layer(theta.hidden_))
					+ (theta.Wre_.block(offset_grammar, offset2, theta.hidden_, theta.hidden_)
					   * state_reduced.layer(theta.hidden_))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = state.score() + score;
      
      agenda_[state_new.step()].push_back(state_new);
    }
    
    void operation_unary(const state_type& state,
			 const symbol_type& label,
			 const model_type& theta)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = state.step() + 1;
      state_new.next()  = state.next();
      state_new.unary() = state.unary() + 1;
      
      state_new.operation() = operation_type(operation_type::UNARY, state.operation().closure() + 1);
      state_new.label()     = label;
      state_new.head()      = symbol_type::EPSILON;
      state_new.span()      = state.span();
      
      state_new.stack()      = state.stack();
      state_new.derivation() = state;
      state_new.reduced()    = state_type();
      
      const size_type offset_grammar        = theta.offset_grammar(label);
      const size_type offset_classification = theta.offset_classification(label);
      
      state_new.layer(theta.hidden_) = (theta.Bu_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wu_.block(offset_grammar, 0, theta.hidden_, theta.hidden_)
					   * state.layer(theta.hidden_))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = state.score() + score;
      
      agenda_[state_new.step()].push_back(state_new);
    }
    
    void operation_final(const state_type& state,
			 const symbol_type& goal,
			 const model_type& theta)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = state.step() + 1;
      state_new.next()  = state.next();
      state_new.unary() = state.unary();
      
      state_new.operation() = operation_type::FINAL;
      state_new.label()     = goal;
      state_new.head()      = symbol_type::EPSILON;
      state_new.span()      = state.span();
      
      state_new.stack()      = state.stack();
      state_new.derivation() = state;
      state_new.reduced()    = state_type();
      
      state_new.layer(theta.hidden_) = (theta.Bf_
					+ theta.Wf_ * state.layer(theta.hidden_)
					).array().unaryExpr(model_type::activation());
      
      const size_type offset_classification = theta.offset_classification(goal);
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = state.score() + score;
      
      agenda_[state_new.step()].push_back(state_new);
    }

    void operation_idle(const state_type& state,
			const symbol_type& goal,
			const model_type& theta)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = state.step() + 1;
      state_new.next()  = state.next();
      state_new.unary() = state.unary();
      
      state_new.operation() = operation_type::IDLE;
      state_new.label()     = goal;
      state_new.head()      = symbol_type::EPSILON;
      state_new.span()      = state.span();
      
      state_new.stack()      = state.stack();
      state_new.derivation() = state;
      state_new.reduced()    = state_type();
      
      state_new.layer(theta.hidden_) = (theta.Bi_
					+ theta.Wi_ * state.layer(theta.hidden_)
					).array().unaryExpr(model_type::activation());
      
      const size_type offset_classification = theta.offset_classification(goal);
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = state.score() + score;
      
      agenda_[state_new.step()].push_back(state_new);
    }

    void operation_axiom(const model_type& theta)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step()  = 0;
      state_new.next()  = 0;
      state_new.unary() = 0;
      
      state_new.operation() = operation_type::AXIOM;
      state_new.label()     = symbol_type::EPSILON;
      state_new.head()      = symbol_type::EPSILON;
      state_new.span()      = span_type(-1, 0);
      
      state_new.stack()      = state_type();
      state_new.derivation() = state_type();
      state_new.reduced()    = state_type();

      state_new.score() = 0;
      state_new.layer(theta.hidden_) = theta.Ba_.array().unaryExpr(model_type::activation());
      
      agenda_[state_new.step()].push_back(state_new);
    }

  public:
    size_type beam_size_;
    size_type unary_size_;
    
    agenda_type agenda_;
    
    // allocator
    state_allocator_type state_allocator_;
  };
};

#endif
