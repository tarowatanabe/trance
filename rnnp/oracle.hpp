// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__ORACLE__HPP__
#define __RNNP__ORACLE__HPP__ 1

//
// transform tree structure into SR sequence
//

#include <rnnp/symbol.hpp>
#include <rnnp/sentence.hpp>
#include <rnnp/tree.hpp>
#include <rnnp/operation.hpp>
#include <rnnp/binarize.hpp>

namespace rnnp
{
  struct Oracle
  {
    typedef Symbol    symbol_type;
    typedef Sentence  sentence_type;
    typedef Tree      tree_type;
    typedef Operation operation_type;

    struct Action
    {
      operation_type operation_;
      symbol_type    label_;
      symbol_type    head_;
      
      Action() : operation_(operation_type::AXIOM), label_(), head_() {}
      Action(const operation_type& operation)
	: operation_(operation), label_(symbol_type::EPSILON), head_(symbol_type::EPSILON) {}
      Action(const operation_type& operation, const symbol_type& label)
	: operation_(operation), label_(label), head_(symbol_type::EPSILON) {}
      Action(const operation_type& operation, const symbol_type& label, const symbol_type& head)
	: operation_(operation), label_(label), head_(head) {}
    };
    typedef Action action_type;
    
    typedef std::vector<action_type, std::allocator<action_type> > action_set_type;
    
    Oracle() {}
    Oracle(const tree_type& tree, const bool left) { assign(tree, left); }
    
    void assign(const tree_type& tree, const bool left)
    {
      actions_.clear();
      actions_.push_back(action_type(operation_type::AXIOM));
      
      sentence_.clear();

      if (left)
	binarize_left(tree, binarized_);
      else
	binarize_right(tree, binarized_);

      oracle(binarized_);
    }

  private:
    void oracle(const tree_type& tree)
    {
      // post-traversal to compute SR sequence
      switch (tree.antecedent_.size()) {
      case 1:
	if (tree.antecedent_.front().label_.terminal()) {
	  if (! tree.antecedent_.front().antecedent_.empty())
	    throw std::runtime_error("invalid leaf");
	  
	  sentence_.push_back(tree.antecedent_.front().label_);
	  actions_.push_back(action_type(operation_type::SHIFT, tree.label_, tree.antecedent_.front().label_));
	} else {
	  oracle(tree.antecedent_.front());
	  actions_.push_back(action_type(operation_type(operation_type::UNARY, actions_.back().operation_.closure() + 1),
				      tree.label_));
	}
	break;
      case 2:
	oracle(tree.antecedent_.front());
	oracle(tree.antecedent_.back());
	actions_.push_back(action_type(operation_type::REDUCE, tree.label_));
	break;
      default:
	throw std::runtime_error("invalid tree structure");
      }
    }
    
  public:
    tree_type       binarized_;
    action_set_type actions_;
    sentence_type   sentence_;
  };
};

#endif
