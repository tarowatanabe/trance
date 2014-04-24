// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__DERIVATION__HPP__
#define __RNNP__DERIVATION__HPP__ 1

#include <vector>

#include <rnnp/sentence.hpp>
#include <rnnp/span.hpp>
#include <rnnp/alignment_span.hpp>
#include <rnnp/symbol.hpp>
#include <rnnp/parser.hpp>

#include <utils/indexed_set.hpp>
#include <utils/chunk_vector.hpp>

#include <boost/functional/hash/hash.hpp>

namespace rnnp
{
  class Derivation
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;
    
    typedef Parser parser_type;
    
    typedef parser_type::sentence_type  sentence_type;
    typedef parser_type::word_type      word_type;
    typedef parser_type::symbol_type    symbol_type;
    
    typedef parser_type::operation_type operation_type;
    
    typedef parser_type::state_type   state_type;
    typedef parser_type::node_type    node_type;

    typedef parser_type::tensor_type tensor_type;
    
    struct edge_type
    {
      sentence_type       target_;
      alignment_span_type alignment_;
      
      index_type first_;
      index_type second_;
      
      edge_type() : target_(), alignment_(), first_(index_type(-1)), second_(index_type(-1)) {}
    };
    
    typedef utils::indexed_set<state_type, boost::hash<state_type>, std::equal_to<state_type>,
			       std::allocator<state_type> > state_map_type;
    typedef utils::chunk_vector<edge_type, 1024 * 16 / sizeof(edge_type), std::allocator<edge_type> > hypergraph_type;
    typedef std::vector<node_type, std::allocator<node_type> > node_set_type;
    
  public:
    Derivation() {}
    template <typename Theta>
    Derivation(const Theta& theta, const node_type& node) { assign(theta, node); }
    
  public:
    template <typename Theta>
    void assign(const Theta& theta, node_type node)
    {
      clear();
      
      // first, construct state-map and hypergraph
      {
	node_type node_curr = node;
	while (node_curr != node_type()) {
	  std::pair<state_map_type::iterator, bool> result = state_map_.insert(node_curr.state());
	  if (! result.second)
	    throw std::runtime_error("the state is already inserted...?");
	  
	  hypergraph_.push_back(edge_type());
	  nodes_.push_back(node_curr);
	  node_curr = node_curr.derivation();
	}
      }
      
      // second, forward, bottom-up traversal to construct target and alignment
      for (difference_type id = nodes_.size() - 1; id >= 0; -- id) {
	const node_type& node = nodes_[id];
	edge_type& edge = hypergraph_[id];
	
	features_ += *node.state().feature_vector();
	
	switch (node.state().operation().operation()) {
	case operation_type::AXIOM:
	  break;
	case operation_type::SHIFT:
	  if (! features_nn_.cols())
	    features_nn_ = tensor_type::Zero(theta.hidden_, 1);
	  features_nn_ += node.state().layer(theta.hidden_);

	  edge.target_.reserve(node.state().target().size());
	  edge.target_.insert(edge.target_.end(), node.state().target().begin(), node.state().target().end());
	  
	  edge.alignment_.reserve(1);
	  edge.alignment_.push_back(span_pair_type(node.state().source(),
						   span_type(0, node.state().target().size())));
	  break;
	case operation_type::REDUCE_STRAIGHT:
	  {
	    if (! features_nn_.cols())
	      features_nn_ = tensor_type::Zero(theta.hidden_, 1);
	    features_nn_ += node.state().layer(theta.hidden_);
	    
	    edge.first_ =  find(node.reduced().state());
	    edge.second_ = find(node.derivation().state());
	    
	    const edge_type& antecedent1 = hypergraph_[edge.first_];
	    const edge_type& antecedent2 = hypergraph_[edge.second_];
	    
	    const size_type target1_size = antecedent1.target_.size();
	    const size_type target2_size = antecedent2.target_.size();
	    
	    const size_type align1_size = antecedent1.alignment_.size();
	    const size_type align2_size = antecedent2.alignment_.size();
	    
	    edge.target_.reserve(target1_size + target2_size);
	    edge.target_.insert(edge.target_.end(), antecedent1.target_.begin(), antecedent1.target_.end());
	    edge.target_.insert(edge.target_.end(), antecedent2.target_.begin(), antecedent2.target_.end());
	    
	    edge.alignment_.reserve(align1_size + align2_size);
	    edge.alignment_.insert(edge.alignment_.end(), antecedent1.alignment_.begin(), antecedent1.alignment_.end());
	    
	    alignment_span_type::const_iterator aiter_end = antecedent2.alignment_.end();
	    for (alignment_span_type::const_iterator aiter = antecedent2.alignment_.begin(); aiter != aiter_end; ++ aiter)
	      edge.alignment_.push_back(span_pair_type(aiter->source_,
						       span_type(aiter->target_.first_ + target1_size,
								 aiter->target_.last_  + target1_size)));
	  }
	  break;
	case operation_type::REDUCE_INVERSION:
	  {
	    if (! features_nn_.cols())
	      features_nn_ = tensor_type::Zero(theta.hidden_, 1);
	    features_nn_ += node.state().layer(theta.hidden_);
	    
	    edge.first_ =  find(node.derivation().state());
	    edge.second_ = find(node.reduced().state());

	    const edge_type& antecedent1 = hypergraph_[edge.first_];
	    const edge_type& antecedent2 = hypergraph_[edge.second_];
	    
	    const size_type target1_size = antecedent1.target_.size();
	    const size_type target2_size = antecedent2.target_.size();
	    
	    const size_type align1_size = antecedent1.alignment_.size();
	    const size_type align2_size = antecedent2.alignment_.size();
	    
	    edge.target_.reserve(target1_size + target2_size);
	    edge.target_.insert(edge.target_.end(), antecedent1.target_.begin(), antecedent1.target_.end());
	    edge.target_.insert(edge.target_.end(), antecedent2.target_.begin(), antecedent2.target_.end());
	    
	    edge.alignment_.reserve(align1_size + align2_size);
	    edge.alignment_.insert(edge.alignment_.end(), antecedent1.alignment_.begin(), antecedent1.alignment_.end());
	    
	    alignment_span_type::const_iterator aiter_end = antecedent2.alignment_.end();
	    for (alignment_span_type::const_iterator aiter = antecedent2.alignment_.begin(); aiter != aiter_end; ++ aiter)
	      edge.alignment_.push_back(span_pair_type(aiter->source_,
						       span_type(aiter->target_.first_ + target1_size,
								 aiter->target_.last_  + target1_size)));	    
	  }
	  break;
	}
      }
      
      // third, copy and sort!
      target_    = hypergraph_.front().target_;
      alignment_ = hypergraph_.front().alignment_;
      
      std::sort(alignment_.begin(), alignment_.end());
    }
    
    void clear()
    {
      target_.clear();
      alignment_.clear();
      features_.clear();
      features_nn_ = tensor_type();
      
      state_map_.clear();
      hypergraph_.clear();
      nodes_.clear();
    }
    
  private:
    index_type find(const state_type& state) const
    {
      state_map_type::const_iterator iter = state_map_.find(state);
      
      if (iter == state_map_.end())
	throw std::runtime_error("no state id?");
      
      return iter - state_map_.begin();
    }
    
  public:    
    sentence_type       target_;
    alignment_span_type alignment_;
    feature_vector_type features_;
    tensor_type         features_nn_;

    state_map_type      state_map_;
    hypergraph_type     hypergraph_;
    node_set_type       nodes_;
  };
};

#endif
