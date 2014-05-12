// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__DEBINARIZE__HPP__
#define __RNNP__DEBINARIZE__HPP__ 1

// debinarization

#include <algorithm>
#include <vector>
#include <iterator>

#include <rnnp/tree.hpp>
#include <rnnp/forest.hpp>
#include <rnnp/sort.hpp>

namespace rnnp
{
  struct DebinarizeTree
  {
    typedef Tree tree_type;

    typedef std::vector<tree_type, std::allocator<tree_type> > tree_set_type;
    
    void operator()(const tree_type& source, tree_type& target)
    {
      tree_set_type antecedent;
      
      debinarize(source.antecedent_.begin(), source.antecedent_.end(), std::back_inserter(antecedent));
      
      target.label_ = source.label_;
      target.antecedent_.resize(antecedent.size());
      
      std::swap_ranges(antecedent.begin(), antecedent.end(), target.antecedent_.begin());
    }
    
    template <typename Iterator, typename Output>
    void debinarize(Iterator first, Iterator last, Output output)
    {
      for (/**/; first != last; ++ first)
	if (first->label_.binarized())
	  debinarize(first->antecedent_.begin(), first->antecedent_.end(), output);
	else {
	  tree_set_type antecedent;
	  
	  debinarize(first->antecedent_.begin(), first->antecedent_.end(), std::back_inserter(antecedent));
	  
	  *output = tree_type(first->label_, antecedent.begin(), antecedent.end());
	  ++ output;
	}
    }
  };

  struct DebinarizeForest
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Forest forest_type;

    typedef forest_type::symbol_type symbol_type;
    typedef forest_type::rule_type   rule_type;
    
    typedef std::vector<bool, std::allocator<bool> > removed_type;
    
    typedef std::vector<int, std::allocator<int> > index_set_type;
    typedef std::vector<forest_type::id_type, std::allocator<forest_type::id_type> > tail_type;
    typedef std::vector<symbol_type, std::allocator<symbol_type> > rhs_type;
    
    struct filter_edge
    {
      filter_edge(const removed_type& removed) : removed_(removed) {}
  
      bool operator()(const forest_type::edge_type& edge) const
      {
        return removed_[edge.id_];
      }
      
      const removed_type& removed_;
     };

    void operator()(const forest_type& source, forest_type& target)
    {
      target = source;
      
      if (! source.valid())
	return;
      
      removed_type remove_edges(target.edges_.size(), false);
      removed_type remove_nodes(target.nodes_.size(), false);
      bool has_remove_nodes = false;
      
      // first, check whether we should remove nodes...
      forest_type::node_set_type::const_iterator niter_end = target.nodes_.end();
      for (forest_type::node_set_type::const_iterator niter = target.nodes_.begin(); niter != niter_end; ++ niter) {
	const forest_type::node_type& node = *niter;
	
	// this should not happen, though..
	if (node.edges_.empty()) continue;
	
	const forest_type::edge_type& edge = target.edges_[node.edges_.front()];
	
	// check if this is the binarized node!
	if (! edge.rule_.lhs_.binarized()) continue;
	
	remove_nodes[node.id_] = true;
	
	forest_type::node_type::edge_set_type::const_iterator eiter_end = node.edges_.end();
	for (forest_type::node_type::edge_set_type::const_iterator eiter = node.edges_.begin(); eiter != eiter_end; ++ eiter)
	  remove_edges[*eiter] = true;
	  
	has_remove_nodes = true;
      }

      if (! has_remove_nodes) return;
      
      tail_type tail;
      rhs_type  rhs;
      
      for (forest_type::node_set_type::const_iterator niter = target.nodes_.begin(); niter != niter_end; ++ niter) {
	const forest_type::node_type& node = *niter;
	
	const size_type edges_size = node.edges_.size();
	for (size_type e = 0; e != edges_size; ++ e) {
	  const forest_type::edge_type& edge = target.edges_[node.edges_[e]];
	  
	  // search for antecedent nodes, and seek the remove_nodes label..
	  // if found, try merge! 
	  // it is like apply-exact to form new edges....
	  
	  index_set_type j_ends(edge.tail_.size(), 0);
	  index_set_type j(edge.tail_.size(), 0);
	  
	  bool found_remove_nodes = false;
	  
	  for (size_type i = 0; i != edge.tail_.size(); ++ i) {
	    found_remove_nodes |= remove_nodes[edge.tail_[i]];
	    j_ends[i] = utils::bithack::branch(remove_nodes[edge.tail_[i]], target.nodes_[edge.tail_[i]].edges_.size(), size_type(0));
	  }
	  
	  if (! found_remove_nodes) continue;
	  
	  remove_edges[edge.id_] = true;
	  
	  for (;;) {
	    tail.clear();
	    rhs.clear();
	    
	    double score = edge.score_;

	    bool invalid = false;
	    int antecedent_index = 0;
	    
	    rule_type::rhs_type::const_iterator riter_end = edge.rule_.rhs_.end();
	    for (rule_type::rhs_type::const_iterator riter = edge.rule_.rhs_.begin(); riter != riter_end; ++ riter)
	      if (riter->non_terminal()) {
		
		if (j_ends[antecedent_index]) {
		  const forest_type::node_type& node_antecedent = target.nodes_[edge.tail_[antecedent_index]];
		  const forest_type::edge_type& edge_antecedent = target.edges_[node_antecedent.edges_[j[antecedent_index]]];
		  
		  score += edge_antecedent.score_;
		  
		  // special care is reqiured for gran-antecedents by converting indices....
		  
		  int grand_index = 0;
		  rule_type::rhs_type::const_iterator riter_end = edge_antecedent.rule_.rhs_.end();
		  for (rule_type::rhs_type::const_iterator riter = edge_antecedent.rule_.rhs_.begin(); riter != riter_end; ++ riter) {
		    if (riter->non_terminal()) {
		      invalid |= remove_nodes[edge_antecedent.tail_[grand_index]];
		      tail.push_back(edge_antecedent.tail_[grand_index]);
		      rhs.push_back(*riter);
		      
		      ++ grand_index;
		    } else
		      rhs.push_back(*riter);
		  }
		  
		} else {
		  tail.push_back(edge.tail_[antecedent_index]);
		  rhs.push_back(*riter);
		}
		
		++ antecedent_index;
	      } else
		rhs.push_back(*riter);
	    
	    if (! invalid) {
	      forest_type::edge_type& edge_new = target.add_edge(tail.begin(), tail.end());
	      
	      edge_new.rule_  = rule_type(edge.rule_.lhs_, rhs.begin(), rhs.end());
	      edge_new.score_ = score;
	      
	      target.connect_edge(edge_new.id_, edge.head_);
	      
	      if (remove_edges.size() < target.edges_.size())
		remove_edges.resize(target.edges_.size(), false);
	      
	      remove_edges[edge_new.id_] = remove_nodes[edge.head_];
	    } 
	    
	    // proceed to the next...
	    size_type index = 0;
	    for (/**/; index != j.size(); ++ index) 
	      if (j_ends[index]) {
		++ j[index];
		if (j[index] < j_ends[index]) break;
		j[index] = 0;
	      }
	    
	    // finished!
	    if (index == j.size()) break;
	  }
	}
      }
      
      remove_edges.resize(target.edges_.size(), false);
      
      forest_type sorted;
      topologically_sort(target, sorted, filter_edge(remove_edges));
      target.swap(sorted);
    }
  };

  inline
  void debinarize(const Tree& source, Tree& target)
  {
    DebinarizeTree debinarize;
    debinarize(source, target);
  }
  
  inline
  void debinarize(Tree& tree)
  {
    Tree debinarized;
    debinarize(tree, debinarized);
    debinarized.swap(tree);
  }

  inline
  void debinarize(const Forest& source, Forest& target)
  {
    DebinarizeForest debinarize;
    debinarize(source, target);
  }
  
  inline
  void debinarize(Forest& forest)
  {
    Forest debinarized;
    debinarize(forest, debinarized);
    debinarized.swap(forest);
  }
};

#endif
