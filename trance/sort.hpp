// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__SORT__HPP__
#define __TRANCE__SORT__HPP__ 1

#include <stdexcept>

#include <trance/forest.hpp>

#include <utils/compact_set.hpp>

namespace trance
{
  struct SortTopologically
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Forest forest_type;
    
    typedef forest_type::id_type id_type;
    typedef forest_type::node_type node_type;
    typedef forest_type::edge_type edge_type;

    enum color_type {
      White,
      Gray,
      Black
    };
    
    struct dfs_type
    {
      id_type node_;
      size_type edge_;
      size_type tail_;

      dfs_type(const id_type& node) 
	: node_(node), edge_(0), tail_(0) {}      
      dfs_type(const id_type& node, const size_type& edge, const size_type& tail) 
	: node_(node), edge_(edge), tail_(tail) {}
    };
    
    typedef std::vector<int, std::allocator<int> > position_set_type;
    typedef std::vector<color_type, std::allocator<color_type> > color_set_type;
    typedef std::vector<dfs_type, std::allocator<dfs_type> > stack_type;
    
    struct id_unassigned
    {
      id_type operator()() const { return id_type(-1); }
    };
    
    typedef utils::compact_set<id_type,
			       id_unassigned, id_unassigned,
			       boost::hash<id_type>, std::equal_to<id_type>,
			       std::allocator<id_type> > id_unique_type;

    struct no_filter_edge
    {
      bool operator()(const edge_type& edge) const
      {
	return false;
      }
    };

    struct filter_edge
    {
      std::vector<bool, std::allocator<bool> > removed_;
      
      filter_edge(size_t size) : removed_(size, false) {}
      
      bool operator()(const edge_type& edge) const
      {
	return removed_[edge.id_];
      }
    };

  public:
    void operator()(const forest_type& forest, forest_type& sorted)
    {
      return operator()(forest, sorted, no_filter_edge());
    }
    
    template <typename Filter>
    void operator()(const forest_type& forest, forest_type& sorted, Filter filter)
    {
      sorted.clear();
      
      if (! forest.valid()) return;
      
      id_unique_type edges_cycle;
      id_unique_type nodes_empty;
      
      position_set_type position_node(forest.nodes_.size(), -1);
      position_set_type position_edge(forest.edges_.size(), -1);
      color_set_type color(forest.nodes_.size(), White);
      stack_type stack;
      
      stack.reserve(forest.nodes_.size());
      stack.push_back(dfs_type(forest.goal_));
      
      size_type node_count = 0;
      size_type edge_count = 0;
      
      while (! stack.empty()) {
	dfs_type dfs = stack.back();
	
	stack.pop_back();
	
	const node_type* curr_node = &(forest.nodes_[dfs.node_]);
	
	while (dfs.edge_ != curr_node->edges_.size()) {
	  const edge_type& curr_edge = forest.edges_[curr_node->edges_[dfs.edge_]];
	  
	  if (dfs.tail_ == curr_edge.tail_.size() || filter(curr_edge)) {
	    // reach end: proceed to the next edge with pos_tail initialized to the first tail
	    ++ dfs.edge_;
	    dfs.tail_ = 0;
	    continue;
	  }
	  
	  const id_type    tail_node  = curr_edge.tail_[dfs.tail_];
	  const color_type tail_color = color[tail_node];
	  
	  switch (tail_color) {
	  case White:
	    ++ dfs.tail_;
	    stack.push_back(dfs);
	    
	    dfs.node_ = tail_node;
	    dfs.edge_ = 0;
	    dfs.tail_ = 0;
	    
	    curr_node = &(forest.nodes_[dfs.node_]);
	    color[dfs.node_] = Gray;
	    break;
	  case Black:
	    ++ dfs.tail_;
	    break;
	  case Gray:
	    // cycle detected...
	    // we will force cutting this cycle!
	    ++ dfs.tail_;
	    edges_cycle.insert(curr_edge.id_);
	    break;
	  }
	}
	
	for (size_type i = 0; i != curr_node->edges_.size(); ++ i)
	  if (! filter(forest.edges_[curr_node->edges_[i]]))
	    position_edge[curr_node->edges_[i]] = edge_count ++;
	
	color[dfs.node_] = Black;
	position_node[dfs.node_] = node_count ++;
      }
      
      // construct nodes
      for (size_type i = 0; i != position_node.size(); ++ i)
	if (position_node[i] >= 0)
	  sorted.add_node();
      
      // construct edges
      for (size_type i = 0; i != position_edge.size(); ++ i)
	if (position_edge[i] >= 0) {
	  const edge_type& edge_old = forest.edges_[i];
	  
	  edge_type& edge_new = sorted.add_edge(edge_old);
	  
	  edge_type::node_set_type::iterator titer_end = edge_new.tail_.end();
	  for (edge_type::node_set_type::iterator titer = edge_new.tail_.begin(); titer != titer_end; ++ titer)
	    *titer = position_node[*titer];
	  
	  sorted.connect_edge(edge_new.id_, position_node[edge_old.head_]);
	}
      
      if (sorted.edges_.empty() || sorted.nodes_.empty()) {
	sorted.clear();
	return;
      }
      
      sorted.goal_ = sorted.nodes_.size() - 1;
      
      // find empty nodes...
      for (size_type i = 0; i != sorted.nodes_.size(); ++ i)
	if (sorted.nodes_[i].edges_.empty())
	  nodes_empty.insert(i);
      
      if (nodes_empty.empty() && edges_cycle.empty()) return;
      
      forest_type sorted_new;
      filter_edge filter_new(sorted.edges_.size());
      
      id_unique_type::const_iterator eiter_end = edges_cycle.end();
      for (id_unique_type::const_iterator eiter = edges_cycle.begin(); eiter != eiter_end; ++ eiter)
	filter_new.removed_[position_edge[*eiter]] = position_edge[*eiter] >= 0;
      
      if (! nodes_empty.empty()) {
	forest_type::edge_set_type::const_iterator eiter_end = sorted.edges_.end();
	for (forest_type::edge_set_type::const_iterator eiter = sorted.edges_.begin(); eiter != eiter_end; ++ eiter) {
	  const edge_type& edge = *eiter;
	  
	  edge_type::node_set_type::const_iterator titer_end = edge.tail_.end();
	  for (edge_type::node_set_type::const_iterator titer = edge.tail_.begin(); titer != titer_end; ++ titer)
	    if (nodes_empty.find(*titer) != nodes_empty.end()) {
	      filter_new.removed_[edge.id_] = true;
	      break;
	    }
	}
      }
      
      operator()(sorted, sorted_new, filter_new);
      
      sorted.swap(sorted_new);
    }
  };
  
  template <typename Filter>
  inline
  void topologically_sort(const Forest& source, Forest& target, Filter filter)
  {
    SortTopologically sorter;
    sorter(source, target, filter);
  }

  inline
  void topologically_sort(const Forest& source, Forest& target)
  {
    SortTopologically sorter;
    sorter(source, target);
  }

  inline
  void topologically_sort(Forest& source)
  {
    Forest target;
    topologically_sort(source, target);
    source.swap(target);
  }

};

#endif
