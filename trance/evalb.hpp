// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__EVALB__HPP__
#define __TRANCE__EVALB__HPP__ 1

#include <cstddef>

#include <iostream>
#include <vector>

#include <trance/symbol.hpp>
#include <trance/span.hpp>
#include <trance/state.hpp>
#include <trance/tree.hpp>

namespace trance
{
  struct Evalb
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef int32_t count_type;

    Evalb() : match_(0), gold_(0), test_(0) {}

    Evalb(const count_type& match,
	  const count_type& gold,
	  const count_type& test)
      : match_(match), gold_(gold), test_(test) {}

  public:
    void clear()
    {
      match_ = 0;
      gold_  = 0;
      test_  = 0;
    }

  public:
    Evalb& operator+=(const Evalb& x)
    {
      match_ += x.match_;
      gold_  += x.gold_;
      test_  += x.test_;
      return *this;
    }

    Evalb& operator-=(const Evalb& x)
    {
      match_ -= x.match_;
      gold_  -= x.gold_;
      test_  -= x.test_;
      return *this;
    }

  public:
    friend
    std::ostream& operator<<(std::ostream& os, const Evalb& evalb);
    friend
    std::istream& operator>>(std::istream& is, Evalb& evalb);
    
  public:
    double recall() const { return (! gold_ ? 0.0 : double(match_) / gold_); }
    double precision() const { return (! test_ ? 0.0 : double(match_) / test_); }
    
    double f() const
    {
      if (! match_ || ! gold_ || ! test_)
	return 0.0;
      else {
	const double p = precision();
	const double r = recall();
	
	return 2 * p * r / (p + r);
      }
    }
    
    double operator()() const { return f(); }
    
    count_type match_;
    count_type gold_;
    count_type test_;
  };

  struct EvalbScorer
  {
    typedef Evalb evalb_type;

    typedef evalb_type::size_type       size_type;
    typedef evalb_type::difference_type difference_type;
    typedef evalb_type::count_type      count_type;

    typedef Symbol symbol_type;
    typedef Span   span_type;
    typedef State  state_type;
    typedef Tree   tree_type;

    typedef Operation operation_type;
    
    typedef std::pair<span_type, symbol_type> stat_type;
    typedef std::vector<stat_type, std::allocator<stat_type> > stat_set_type;
    
  public:
    EvalbScorer() {}
    EvalbScorer(const state_type& state) { assign(state); }
    EvalbScorer(const tree_type& tree) { assign(tree); }
    
    void assign(const state_type& state)
    {
      collect(state, gold_);
    }
    
    void assign(const tree_type& tree)
    {
      collect(tree, gold_);
    }

    evalb_type operator()(state_type state) const
    {
      collect(state, const_cast<stat_set_type&>(test_));
      
      
      return evalb_type(matched(), gold_.size(), test_.size());
    }

    evalb_type operator()(const tree_type& tree) const
    {
      collect(tree, const_cast<stat_set_type&>(test_));
      
      return evalb_type(matched(), gold_.size(), test_.size());
    }

  private:

    count_type matched() const
    {
      stat_set_type::const_iterator giter     = gold_.begin();
      stat_set_type::const_iterator giter_end = gold_.end();
      
      stat_set_type::const_iterator titer     = test_.begin();
      stat_set_type::const_iterator titer_end = test_.end();

      count_type match = 0;
      
      while (giter != giter_end && titer != titer_end) {
	if (*giter < *titer)
	  ++ giter;
	else if (*titer < *giter)
	  ++ titer;
	else {
	  ++ match;
	  ++ giter;
	  ++ titer;
	}
      }
      
      return match;
    }

    void collect(state_type state, stat_set_type& stats) const
    {
      stats.clear();
      
      while (state) {
	switch (state.operation().operation()) {
	case operation_type::REDUCE:
	case operation_type::REDUCE_LEFT:
	case operation_type::REDUCE_RIGHT:
	case operation_type::UNARY:
	  if (! state.label().binarized())
	    stats.push_back(stat_type(state.span(), state.label()));
	  break;
	default:
	  break;
	}
	state = state.derivation();
      }

      std::sort(stats.begin(), stats.end(), std::less<stat_type>());
    }
    
    void collect(const tree_type& tree, stat_set_type& stats) const
    {
      stats.clear();
      
      span_type span(0, 0);
      collect(tree, span, stats);

      std::sort(stats.begin(), stats.end(), std::less<stat_type>());
    }
    
    void collect(const tree_type& tree, span_type& span, stat_set_type& stats) const
    {
      bool pos = false;

      tree_type::const_iterator titer_end = tree.end();
      for (tree_type::const_iterator titer = tree.begin(); titer != titer_end; ++ titer) {
	const tree_type& antecedent = *titer;
	
	if (antecedent.antecedent_.empty()) {
	  ++ span.last_;
	  pos = true;
	} else {
	  span_type span_ant(span.last_, span.last_);
	  collect(antecedent, span_ant, stats);
	  
	  span.last_ = span_ant.last_;
	}
      }

      // post-traversal
      if (! pos && tree.label_.non_terminal() && ! tree.label_.binarized())
	stats.push_back(stat_type(span, tree.label_));
    }
    
  private:
    stat_set_type gold_;
    stat_set_type test_;
  };

  inline
  Evalb operator+(const Evalb& x, const Evalb& y)
  {
    Evalb ret = x;
    ret += y;
    return ret;
  }

  inline
  Evalb operator-(const Evalb& x, const Evalb& y)
  {
    Evalb ret = x;
    ret -= y;
    return ret;
  }

};

#endif
