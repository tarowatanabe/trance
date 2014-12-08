// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__SPAN__HPP__
#define __TRANCE__SPAN__HPP__ 1

#include <stdint.h>

#include <utility>
#include <iostream>

#include <utils/piece.hpp>
#include <utils/hashmurmur3.hpp>

namespace trance
{
  struct Span
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef int32_t   index_type;
    
    Span() : first_(index_type(-1)), last_(index_type(-1)) {}
    Span(const index_type& first, const index_type& last) : first_(first), last_(last) {}
    Span(const utils::piece& x) { assign(x); }
    template <typename Tp>
    Span(const std::pair<Tp, Tp>& x) : first_(x.first), last_(x.second) {}

    void assign(const utils::piece& x);
    
  public:
    bool empty() const { return first_ == last_; }
    difference_type size() const { return last_ - first_; }

    void swap(Span& x)
    {
      std::swap(first_, x.first_);
      std::swap(last_,  x.last_);
    }
    
  public:
    friend
    std::istream& operator>>(std::istream& is, Span& x);
    
    friend
    std::ostream& operator<<(std::ostream& os, const Span& x);
  public:
    index_type first_;
    index_type last_;
  };
  
  inline
  size_t hash_value(Span const& x)
  {
    return utils::hashmurmur3<size_t>()(x.first_, x.last_);
  }
  
  inline
  bool operator==(const Span&x, const Span& y)
  {
    return x.first_ == y.first_ && x.last_ == y.last_;
  }

  inline
  bool operator!=(const Span&x, const Span& y)
  {
    return x.first_ != y.first_ || x.last_ != y.last_;
  }

  inline
  bool operator<(const Span& x, const Span& y)
  {
    return x.first_ < y.first_ || (! (y.first_ < x.first_) && x.last_ < y.last_);
  }
  
  inline
  bool operator>(const Span& x, const Span& y)
  {
    return y < x;
  }
  
  inline
  bool operator<=(const Span& x, const Span& y)
  {
    return ! (y < x);
  }
  
  inline
  bool operator>=(const Span& x, const Span& y)
  {
    return ! (x < y);
  }
};

namespace std
{
  inline
  void swap(trance::Span& x, trance::Span& y)
  {
    x.swap(y);
  }
};

#endif
