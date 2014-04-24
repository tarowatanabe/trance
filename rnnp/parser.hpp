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
#include <rnnp/state.hpp>
#include <rnnp/node.hpp>
#include <rnnp/allocator.hpp>

namespace rnnp
{
  class Parser
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef Sentence  sentence_type;
    
    typedef Grammar grammar_type;

    typedef grammar_type::rule_type rule_type;

    typedef Model model_type;
    
    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
    typedef Operation operation_type;
    
    typedef State state_type;
    typedef Node  node_type;
    
  public:
    // heap...
    typedef std::vector<node_type, std::allocator<node_type> > heap_type;
    typedef std::vector<heap_type, std::allocator<heap_type> > agenda_type;
    
    struct heap_compare
    {
      // compare by less so that better scored candidate preserved
      bool operator()(const node_type& x, const node_type& y) const
      {
	return x.state().score() < y.state().score();
      }
    };
    
    typedef std::vector<node_type, std::allocator<node_type> > derivation_set_type;
    typedef std::vector<node_type, std::allocator<node_type> > stack_type;

    struct output_agenda
    {
      output_agenda(agenda_type& agenda) : agenda_(agenda) {}

      void operator()(const node_type& node)
      {
	agenda_[node.state().step()].push_back(node);
      }
      
      agenda_type& agenda_;
    };

    struct best_action_none
    {
      void operator()(const size_type& step, const node_type& node) const
      {
	
      }
    };

    struct action_none
    {
      void operator()(const node_type& node) const
      {
	
      }
    };
    
  public:

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
      
      initialize(input);

      operation_axiom(theta, output_agenda(agenda_), action_none());
      
      const size_type step_last = input.size() * 2 + input.size() * unary_size_;
      for (size_type step = 0; step != step_last; ++ step) {
	heap_type& heap = agenda_[step];
	
	if (heap.empty()) continue;
	
	heap_type::iterator hiter_begin = heap.begin();
	heap_type::iterator hiter       = heap.end();
	heap_type::iterator hiter_end   = heap.end();
	
	std::make_heap(hiter_begin, hiter_end, heap_compare());
	
	for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != beam; -- hiter)
	  std::pop_heap(hiter_begin, hiter, heap_compare());
	
	// deallocate unused nodes
	for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter) {
	  state_allocator_.deallocate(iter->state());
	  node_allocator_.deallocate(*iter);
	}
	
	if (hiter != hiter_end)
	  best_action(step, *(hiter_end - 1));
	
	for (heap_type::iterator iter = hiter; iter != hiter_end; ++ iter) {
	  const node_type& node = *iter;
	  
	  if (node.state().operation().finished())
	    operation_idle(node, grammar.goal_, theta, output_agenda(agenda_), action_none());
	  else {
	    // we perform shift..
	    if (node.state().next() < input.size()) {
	      const word_type word = input[node.state().next()];
	      
	      grammar_type::rule_set_preterminal_type::const_iterator piter = grammar.preterminal_.find(word);
	      
	      if (piter == grammar.preterminal_.end()) {
		piter = grammar.preterminal_.find(symbol_type::UNK);
		
		if (piter == grammar.preterminal_.end())
		  throw std::runtime_error("no fallback preterminal?");
	      }
	      
	      grammar_type::rule_set_type::const_iterator riter_end = piter->second.end();
	      for (grammar_type::rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
		operation_shift(node, *riter, theta, output_agenda(agenda_), action_none());
	    }
	    
	    // we perform unary
	    if (node.state().operation().closure() < unary_size_) {
	      grammar_type::rule_set_unary_type::const_iterator uiter = grammar.unary_.find(node.state().label());
	      
	      if (uiter != grammar.unary_.end()) {
		grammar_type::rule_set_type::const_iterator riter_end = uiter->second.end();
		for (grammar_type::rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
		  operation_unary(node, *riter, theta, output_agenda(agenda_), action_none());
	      }
	    }
	    
	    // final...
	    if (node.stack()
		&& node.stack().state().label() == symbol_type::EPSILON
		&& node.state().label() == grammar.goal_
		&& node.state().next() == input.size())
	      operation_final(node, grammar.goal_, theta, output_agenda(agenda_), action_none());
	    
	    // we will perform reduce
	    if (node.stack() && node.stack().state().label() != symbol_type::EPSILON) {
	      const symbol_type& left  = node.stack().state().label();
	      const symbol_type& right = node.state().label();
	      
	      grammar_type::rule_set_binary_type::const_iterator biter = grammar.binary_.find(std::make_pair(left, right)));
	    
	    if (biter != grammar.binary_.end()) {
	      grammar_type::rule_set_type::const_iterator riter_end = biter->second.end();
	      for (grammar_type::rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
		operation_reduce(node, *riter, theta, output_agenda(agenda_), action_none());
	    }
	  }
	}
	
	// erase deallocated nodes
	heap.erase(hiter_begin, hiter);
      }
      
      // compute the final kbest derivations
      heap_type& heap = agenda_[step_last];
      
      if (! heap.empty()) {
	heap_type::iterator hiter_begin = heap.begin();
	heap_type::iterator hiter       = heap.end();
	heap_type::iterator hiter_end   = heap.end();
	
	std::make_heap(hiter_begin, hiter_end, heap_compare());
	for (/**/; hiter_begin != hiter && std::distance(hiter, hiter_end) != kbest; -- hiter) {
	  std::pop_heap(hiter_begin, hiter, heap_compare());
	  
	  if ((hiter - 1)->stack()) {
	    std::cerr << "non-final operation? " << std::endl;
	    
	    node_type stack = *(hiter - 1);
	    while (stack) {
	      std::cerr << "\tstack op: " << stack.state().operation()
			<< " step: " << stack.state().step() << std::endl
			<< " next: " << stack.state().next() << std::endl
			<< " label: " << stack.state().label() << std::endl; 
	      
	      stack = stack.stack();
	    }
	    
	    node_type curr = *(hiter - 1);
	    while (curr) {
	      std::cerr << "\tderivation op: " << curr.state().operation()
			<< " step: " << curr.state().step() << std::endl
			<< " next: " << curr.state().next() << std::endl
			<< " label: " << curr.state().label() << std::endl; 
	      
	      curr = curr.derivation();
	    }
	  }
	  
	  derivations.push_back(*(hiter - 1));
	}
	
	// deallocate unused nodes
	for (heap_type::iterator iter = hiter_begin; iter != hiter; ++ iter) {
	  state_allocator_.deallocate(iter->state());
	  node_allocator_.deallocate(*iter);
	}

	if (hiter != hiter_end)
	  best_action(step_last, *(hiter_end - 1));

	// erase deallocated nodes
	heap.erase(hiter_begin, hiter);
      }
    }
    
    void initialize(const sentence_type& input)
    {
      // # of operations is 2n + # of unary rules
      agenda_.clear();
      agenda_.resize(input.size() * 2 + input.size() * unary_size_ + 1);
      
      state_allocator_.clear();
      node_allocator_.clear();
      
      state_allocator_.assign(state_type::size(theta.hidden_));
      node_allocator_.assign(node_type::size(theta.hidden_));
    }

    template <typename Output, typename Action>
    void operation_shift(const node_type& node,
			 const rule_type& rule,
			 const model_type& model,
			 Output output,
			 Action action)
    {
      if (! rule.preterminal())
	throw std::runtime_error("invalid preterminal rule: " + rule.string());

      const size_type offset1 = 0;
      const size_type offset2 = theta.hidden_;

      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = node.state().step() + 1;
      state_new.next() = node.state().next() + 1;
      
      state_new.label() = rule.lhs_;
      state_new.operation() = operation_type::SHIFT;
      
      const size_type offset_grammar        = theta.offset_grammar(rule.lhs_);
      const size_type offset_classification = theta.offset_classification(rule.lhs_);
      
      state_new.layer(theta.hidden_) = (theta.Bsh_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wsh_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_)
					   * node.state().layer(theta.hidden_))
					+ (theta.Wsh_.block(offset_grammar, offset2, theta.hidden_, theta.embedding_)
					   * model.terminal_.col(model.terminal(rule.rhs_.front())))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = node.state().score() + score;
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node.stack();
      node_new.derivation() = node;
      node_new.reduced()    = node_type();
      
      output(node_new);
      
      action(node_new);
    }
    
    template <typename Output, typename Action>
    void operation_reduce(const node_type& node,
			  const rule_type& rule,
			  const model_type& model,
			  Output output,
			  Action action)
    {
      if (! rule.binary())
	throw std::runtime_error("invalid binary rule: " + rule.string());
	    
      const size_type offset1 = 0;
      const size_type offset2 = theta.hidden_;
      
      const node_type node_reduced = node.stack();
      const node_type node_stack = node_reduced.stack();

      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = node.state().step() + 1;
      state_new.next() = node.state().next();
      
      state_new.label() = rule.lhs_;
      state_new.operation() = operation_type::REDUCE;
      
      const size_type offset_grammar        = theta.offset_grammar(rule.lhs_);
      const size_type offset_classification = theta.offset_classification(rule.lhs_);
      
      state_new.layer(theta.hidden_) = (theta.Bre_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wre_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_)
					   * node.state().layer(theta.hidden_))
					+ (theta.Wre_.block(offset_grammar, offset2, theta.hidden_, theta.hidden_)
					   * node_reduced.state().layer(theta.hidden_))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = node.state().score() + score;
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node_stack;
      node_new.derivation() = node;
      node_new.reduced()    = node_reduced;
      
      output(node_new);
      
      action(node_new);
    }
    
    template <typename Output, typename Action>
    void operation_unary(const node_type& node,
			 const rule_type& rule,
			 const model_type& model,
			 Output output,
			 Action action)
    {
      if (! rule.unary())
	throw std::runtime_error("invalid unary rule: " + rule.string());
      
      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = node.state().step() + 1;
      state_new.next() = node.state().next();
      
      state_new.label() = rule.lhs_;
      state_new.operation() = operation_type(operation_type::UNARY, node.state().operation().closure() + 1);
      
      const size_type offset_grammar        = theta.offset_grammar(goal);
      const size_type offset_classification = theta.offset_classification(goal);
      
      state_new.layer(theta.hidden_) = (theta.Bu_.block(offset_grammar, 0, theta.hidden_, 1)
					+ (theta.Wu_.block(offset_grammar, 0, theta.hidden_, theta.hidden_)
					   * node.state().layer(theta.hidden_))
					).array().unaryExpr(model_type::activation());
      
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = node.state().score() + score;
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node.stack();
      node_new.derivation() = node;
      node_new.reduced()    = node_type();
      
      output(node_new);
      
      action(node_new);
    }
    
    template <typename Output, typename Action>
    void operation_final(const node_type& node,
			 const symbol_type& goal,
			 const model_type& theta,
			 Output output,
			 Action action)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = node.state().step() + 1;
      state_new.next() = node.state().next();
      
      state_new.label() = goal;
      state_new.operation() = operation_type::FINAL;
      
      state_new.layer(theta.hidden_) = (theta.Bf_
					+ theta.Wf_ * node.state().layer(theta.hidden_)
					).array().unaryExpr(model_type::activation());
      
      const size_type offset_classification = theta.offset_classification(goal);
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = node.state().score() + score;
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node.stack();
      node_new.derivation() = node;
      node_new.reduced()    = node_type();
      
      output(node_new);
      
      action(node_new);
    }

    template <typename Output, typename Action>
    void operation_idle(const node_type& node,
			const symbol_type& goal,
			const model_type& theta,
			Output output,
			Action action)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = node.state().step() + 1;
      state_new.next() = node.state().next();
      
      state_new.label() = goal;
      state_new.operation() = operation_type::IDLE;
      
      state_new.layer(theta.hidden_) = (theta.Bi_
					+ theta.Wi_ * node.state().layer(theta.hidden_)
					).array().unaryExpr(model_type::activation());
      
      const size_type offset_classification = theta.offset_classification(goal);
      const double score = (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_) * state_new.layer(theta.hidden_))(0, 0);
      
      state_new.score() = node.state().score() + score;
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node.stack();
      node_new.derivation() = node;
      node_new.reduced()    = node_type();
      
      output(node_new);
      
      action(node_new);
    }

    template <typename Output, typename Action>
    void operation_axiom(const model_type& theta,
			 Output output,
			 Action action)
    {
      state_type state_new = state_allocator_.allocate();
      
      state_new.step() = 0;
      state_new.next() = 0;
      
      state_new.label() = symbol_type::EPSILON;
      state_new.operation() = operation_type::AXIOM;
      
      state_new.score() = 0;
      state_new.layer(theta.hidden_) = theta.Ba().array().unaryExpr(model_type::activation());
      
      node_type node_new = node_allocator_.allocate();
      
      node_new.state()      = state_new;
      node_new.stack()      = node_type();
      node_new.derivation() = node_type();
      node_new.reduced()    = node_type();
      
      output(node_new);
      
      action(node_new);
    }
        

  public:
    size_type beam_size_;
    size_type unary_size_;
    
    agenda_type agenda
    
    // allocator
    state_allocator_type          state_allocator_;
    node_allocator_type           node_allocator_;
  };
};

#endif
