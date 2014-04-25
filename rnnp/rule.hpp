// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__RULE__HPP__
#define __RNNP__RULE__HPP__ 1

#include <iostream>
#include <string>

#include <rnnp/symbol.hpp>

#include <utils/hashmurmur3.hpp>
#include <utils/piece.hpp>
#include <utils/small_vector.hpp>

namespace rnnp
{
  class Rule
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef rnnp::Symbol symbol_type;
    
    typedef symbol_type lhs_type;
    typedef utils::small_vector<symbol_type, std::allocator<symbol_type> > rhs_type;

  public:
    Rule() {}
    Rule(const lhs_type& lhs) : lhs_(lhs), rhs_() {}
    Rule(const lhs_type& lhs, const rhs_type& rhs) : lhs_(lhs), rhs_(rhs) {}
    Rule(const lhs_type& lhs, const size_type& size) : lhs_(lhs), rhs_(size) {}
    template <typename Iterator>
    Rule(const lhs_type& lhs, Iterator first, Iterator last) : lhs_(lhs), rhs_(first, last) {}
    Rule(const utils::piece& x) { assign(x); }
    
  public:
    void assign(const utils::piece& x);
    bool assign(std::string::const_iterator& iter, std::string::const_iterator end);
    bool assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end);
    
    void clear()
    {
      lhs_ = symbol_type();
      rhs_.clear();
    }

    void swap(Rule& x)
    {
      lhs_.swap(x.lhs_);
      rhs_.swap(x.rhs_);
    }
    
    std::string string() const;

  public:
    bool goal() const { return rhs_.empty(); }
    bool unary() const { return rhs_.size() == 1 && rhs_.front().non_terminal(); }
    bool binary() const { return rhs_.size() == 2 && rhs_.front().non_terminal() && rhs_.back().non_terminal(); }
    bool preterminal() const { return rhs_.size() == 1 && rhs_.front().terminal(); }
    
  public:
    friend
    std::ostream& operator<<(std::ostream& os, const Rule& rule);
    friend
    std::istream& operator>>(std::istream& is, Rule& rule);

  public:
    lhs_type lhs_;
    rhs_type rhs_;
  };
  
  inline
  size_t hash_value(Rule const& x)
  {
    return utils::hashmurmur3<size_t>()(x.rhs_.begin(), x.rhs_.end(), x.lhs_.id());
  }

  inline
  bool operator==(const Rule& x, const Rule& y)
  {
    return x.lhs_ == y.lhs_ && x.rhs_ == y.rhs_;
  }

  inline
  bool operator!=(const Rule& x, const Rule& y)
  {
    return x.lhs_ != y.lhs_ || x.rhs_ != y.rhs_;
  }
  
  inline
  bool operator<(const Rule& x, const Rule& y)
  {
    return (x.lhs_ < y.lhs_ || (!(y.lhs_ < x.lhs_) && x.rhs_ < y.rhs_));
  }

  inline
  bool operator>(const Rule& x, const Rule& y)
  {
    return y < x;
  }

  inline
  bool operator<=(const Rule& x, const Rule& y)
  {
    return ! (y < x);
  }
  
  inline
  bool operator>=(const Rule& x, const Rule& y)
  {
    return ! (x < y);
  }

};

namespace std
{
  inline
  void swap(rnnp::Rule& x, rnnp::Rule& y)
  {
    x.swap(y);
  }
};

#endif
