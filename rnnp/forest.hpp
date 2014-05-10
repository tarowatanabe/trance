// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__FOREST__HPP__
#define __RNNP__FOREST__HPP__ 1

#include <iostream>
#include <vector>
#include <string>

#include <rnnp/symbol.hpp>
#include <rnnp/rule.hpp>
#include <rnnp/tree.hpp>

#include <utils/hashmurmur3.hpp>
#include <utils/piece.hpp>
#include <utils/small_vector.hpp>
#include <utils/chunk_vector.hpp>

namespace rnnp
{
  class Forest
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  id_type;

    typedef Symbol symbol_type;
    typedef Symbol word_type;
    typedef Rule   rule_type;
    typedef double score_type;

    typedef Tree tree_type;
    
  public:
    static const id_type invalid = id_type(-1);

  public:
    struct Node
    {
      typedef std::vector<id_type, std::allocator<id_type> > edge_set_type;

      Node() : id_(invalid) {}
      
      edge_set_type edges_;
      id_type       id_;
    };
    
    struct Edge
    {
      typedef utils::small_vector<id_type, std::allocator<id_type> > node_set_type;

      Edge()
	: rule_(), score_(0), tail_(), head_(invalid), id_(invalid) {}
      template <typename Iterator>
      Edge(Iterator first, Iterator last)
	: rule_(), score_(0), tail_(first, last), head_(invalid), id_(invalid) {}
      
      rule_type  rule_;
      score_type score_;
      
      node_set_type tail_;
      id_type       head_;
      
      id_type id_;
    };

    typedef Node node_type;
    typedef Edge edge_type;
    
  public:
    typedef utils::chunk_vector<node_type, 4096 / sizeof(node_type), std::allocator<node_type> > node_set_type;
    typedef utils::chunk_vector<edge_type, 4096 / sizeof(edge_type), std::allocator<edge_type> > edge_set_type;

  public:
    Forest() : nodes_(), edges_(), goal_(invalid) {}
    Forest(const tree_type& tree) { assign(tree); }
    Forest(const utils::piece& x) { assign(x); }

  public:
    void assign(const utils::piece& x);
    bool assign(std::string::const_iterator& iter, std::string::const_iterator end);
    bool assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end);

    void assign(const tree_type& tree);

  public:
    void clear()
    {
      nodes_.clear();
      edges_.clear();
      goal_ = invalid;
    }

    void swap(Forest& x)
    {
      nodes_.swap(x.nodes_);
      edges_.swap(x.edges_);
      std::swap(goal_, x.goal_);
    }

    bool is_valid() const
    {
      return goal_ != invalid;
    }

  public:
    edge_type& add_edge()
    {
      const id_type edge_id = edges_.size();
      
      edges_.push_back(edge_type());
      edges_.back().id_ = edge_id;
      
      return edges_.back();
    }
    
    template <typename Iterator>
    edge_type& add_edge(Iterator first, Iterator last)
    {
      const id_type edge_id = edges_.size();
      
      edges_.push_back(edge_type(first, last));
      edges_.back().id_ = edge_id;
      
      return edges_.back();
    }
    
    node_type& add_node()
    {
      const id_type node_id = nodes_.size();
      
      nodes_.push_back(node_type());
      nodes_.back().id_ = node_id;
      
      return nodes_.back();
    }
    
    void connect_edge(const id_type edge, const id_type head)
    {
      edges_[edge].head_ = head;
      nodes_[head].edges_.push_back(edge);
    };

  public:
    friend
    std::ostream& operator<<(std::ostream& os, const Forest& forest);
    friend
    std::istream& operator>>(std::istream& is, Forest& forest);
    
  public:
    node_set_type nodes_;
    edge_set_type edges_;
    id_type goal_;
  };
};

#endif
