// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__DERIVATION__HPP__
#define __TRANCE__DERIVATION__HPP__ 1

#include <trance/sentence.hpp>
#include <trance/span.hpp>
#include <trance/symbol.hpp>
#include <trance/parser.hpp>
#include <trance/tree.hpp>
#include <trance/forest.hpp>
#include <trance/debinarize.hpp>
#include <trance/sort.hpp>

#include <utils/compact_map.hpp>
#include <utils/compact_set.hpp>

namespace trance
{
  class Derivation
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;

    typedef Tree tree_type;
    
    typedef Parser parser_type;
    
    typedef parser_type::sentence_type  sentence_type;
    typedef parser_type::word_type      word_type;
    typedef parser_type::symbol_type    symbol_type;
    
    typedef parser_type::operation_type      operation_type;
    typedef parser_type::feature_vector_type feature_vector_type;
    
    typedef parser_type::state_type   state_type;
    
    typedef parser_type::derivation_set_type derivation_set_type;

    typedef Forest forest_type;

    typedef utils::compact_map<state_type, forest_type::id_type,
			       utils::unassigned<state_type>, utils::unassigned<state_type>,
			       boost::hash<state_type>, std::equal_to<state_type>,
			       std::allocator<std::pair<const state_type, forest_type::id_type> > > state_map_type;
    typedef utils::compact_set<state_type,
			       utils::unassigned<state_type>, utils::unassigned<state_type>,
			       boost::hash<state_type>, std::equal_to<state_type>,
			       std::allocator<state_type > > visited_type;
    
  public:
    Derivation() {}
    Derivation(const state_type& state) { assign(state); }
    Derivation(const derivation_set_type& derivations) { assign(derivations.begin(), derivations.end()); }
    template <typename Iterator>
    Derivation(Iterator first, Iterator last) { assign(first, last); }
    
  public:
    void assign(state_type state)
    {
      clear();
      
      assign(state, tree_binarized_);
      debinarize(tree_binarized_, tree_);
    }
    
    void assign(const derivation_set_type& derivations)
    {
      assign(derivations.begin(), derivations.end());
    }
    
    template <typename Iterator>
    void assign(Iterator first, Iterator last)
    {
      clear();
      
      if (first == last) return;
      
      forest_binarized_.goal_ = forest_binarized_.add_node().id_;
      
      for (/**/; first != last; ++ first)
	assign(*first, forest_binarized_);

      topologically_sort(forest_binarized_);

      debinarize(forest_binarized_, forest_);
    }
    
    void clear()
    {
      tree_binarized_.clear();
      tree_.clear();
      
      forest_binarized_.clear();
      forest_.clear();

      features_.clear();

      states_.clear();
      visited_.clear();
    }
    
  private:
    void assign(state_type state, tree_type& tree)
    {
      features_ += *state.feature_vector();

      switch (state.operation().operation()) {
      case operation_type::AXIOM:
	break;
      case operation_type::FINAL:
      case operation_type::IDLE:
	assign(state.derivation(), tree);
	break;
      case operation_type::UNARY:
	tree.label_ = state.label();
	tree.antecedent_.resize(1);
	assign(state.derivation(), tree.antecedent_.front());
	break;
      case operation_type::SHIFT:
	tree.label_ = state.label();
	tree.antecedent_ = tree_type::antecedent_type(1, state.head());
	break;
      case operation_type::REDUCE:
      case operation_type::REDUCE_LEFT:
      case operation_type::REDUCE_RIGHT:
	tree.label_ = state.label();
	tree.antecedent_.resize(2);
	assign(state.reduced(), tree.antecedent_.front());
	assign(state.derivation(), tree.antecedent_.back());
	break;
      }
    }

    void assign(state_type state, forest_type& forest)
    {
      double score_accumulated = 0;

      while (state) {
	if (! visited_.insert(state).second) break;
	
	switch (state.operation().operation()) {
	case operation_type::AXIOM:
	  break;
	case operation_type::FINAL:
	  score_accumulated += state.score() - state.derivation().score();
	  
	  states_.insert(std::make_pair(state.derivation(), forest.goal_));
	  break;
	case operation_type::IDLE:
	  score_accumulated += state.score() - state.derivation().score();
	  break;
	case operation_type::UNARY: {
	  std::pair<state_map_type::iterator, bool> parent = states_.insert(std::make_pair(state, 0));
	  if (parent.second)
	    parent.first->second = forest.add_node().id_;
	  
	  std::pair<state_map_type::iterator, bool> antecedent = states_.insert(std::make_pair(state.derivation(), 0));
	  if (antecedent.second)
	    antecedent.first->second = forest.add_node().id_;
	  
	  const forest_type::id_type&     tail = antecedent.first->second;
	  const forest_type::symbol_type& rhs  = state.derivation().label();
	  
	  forest_type::edge_type& edge = forest.add_edge(&tail, (&tail) + 1);
	  
	  edge.score_ = state.score() - state.derivation().score() + score_accumulated;
	  edge.rule_ = forest_type::rule_type(state.label(), &rhs, (&rhs) + 1);
	  
	  forest.connect_edge(edge.id_, parent.first->second);
	  score_accumulated = 0;
	} break;
	case operation_type::SHIFT: {
	  std::pair<state_map_type::iterator, bool> parent = states_.insert(std::make_pair(state, 0));
	  if (parent.second)
	    parent.first->second = forest.add_node().id_;

	  const forest_type::symbol_type& rhs = state.head();
	  
	  forest_type::edge_type& edge = forest.add_edge();
	  
	  edge.score_ = state.score() - state.derivation().score() + score_accumulated;
	  edge.rule_ = forest_type::rule_type(state.label(), &rhs, (&rhs) + 1);
	  
	  forest.connect_edge(edge.id_, parent.first->second);
	  score_accumulated = 0;
	} break;
	case operation_type::REDUCE:
	case operation_type::REDUCE_LEFT:
	case operation_type::REDUCE_RIGHT: {
	  std::pair<state_map_type::iterator, bool> parent = states_.insert(std::make_pair(state, 0));
	  if (parent.second)
	    parent.first->second = forest.add_node().id_;
	  
	  std::pair<state_map_type::iterator, bool> antecedent1 = states_.insert(std::make_pair(state.reduced(), 0));
	  if (antecedent1.second)
	    antecedent1.first->second = forest.add_node().id_;
	  
	  std::pair<state_map_type::iterator, bool> antecedent2 = states_.insert(std::make_pair(state.derivation(), 0));
	  if (antecedent2.second)
	    antecedent2.first->second = forest.add_node().id_;
	  
	  forest_type::id_type tail[2];
	  tail[0] = antecedent1.first->second;
	  tail[1] = antecedent2.first->second;
	  
	  forest_type::symbol_type rhs[2];
	  rhs[0] = state.reduced().label();
	  rhs[1] = state.derivation().label();
	  
	  forest_type::edge_type& edge = forest.add_edge(tail, tail + 2);
	  
	  edge.score_ = state.score() - state.derivation().score() + score_accumulated;
	  edge.rule_ = forest_type::rule_type(state.label(), rhs, rhs + 2);
	  
	  forest.connect_edge(edge.id_, parent.first->second);
	  score_accumulated = 0;
	} break;
	}
	
	state = state.derivation();
      }
    }

  public:    
    tree_type tree_binarized_;
    tree_type tree_;
    
    forest_type forest_binarized_;
    forest_type forest_;

    feature_vector_type features_;
    
    state_map_type states_;
    visited_type   visited_;
  };
};

#endif
