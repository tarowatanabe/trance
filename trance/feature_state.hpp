// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__FEATURE_STATE__HPP__
#define __TRANCE__FEATURE_STATE__HPP__ 1

#include <algorithm>

#include <boost/functional/hash/hash.hpp>

namespace trance
{
  template <typename Tp>
  class Allocator;

  class FeatureSet;
  
  struct FeatureState
  {
    friend class Allocator<FeatureState>;
    friend class FeatureSet;
    
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;
    
    typedef char* pointer;

    typedef FeatureState state_type;
    
  public:
    FeatureState() : buffer_(0) {}
    
  private:
    FeatureState(pointer buffer) : buffer_(buffer) {}
    
  public:
    bool empty() const { return ! buffer_; }
    
    operator bool() const { return ! empty(); }
    
  public:
    
    friend
    bool operator==(const state_type& x, const state_type& y)
    {
      return x.buffer_ == y.buffer_;
    }
    
    friend
    bool operator!=(const state_type& x, const state_type& y)
    {
      return x.buffer_ != y.buffer_;
    }
    
    friend
    size_t  hash_value(state_type const& x)
    {
      return boost::hash<pointer>()(x.buffer_);
    }
    
  private:
    pointer buffer_;
  };
};

#endif

// include allocator
#include <trance/allocator.hpp>
