// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__NODE__HPP__
#define __RNNP__NODE__HPP__ 1

#include <algorithm>

#include <rnnp/model.hpp>
#include <rnnp/state.hpp>

#include <boost/functional/hash/hash.hpp>

namespace rnnp
{
  template <typename Tp>
  class Allocator;

  struct Node
  {
    friend class Allocator<Node>;
    
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;
    
    typedef State state_type;
    typedef Node  node_type;
    
    typedef char* pointer;

    typedef Model model_type;
    
    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
  public:
    Node() : buffer_(0) {}
    
  private:
    Node(pointer buffer) : buffer_(buffer) {}
    
  public:
    bool empty() const { return ! buffer_; }
    
    operator bool() const { return ! empty(); }
    
    
  private:
    // state information
    // state_type state_;
    
    // stack and derivation
    // node_type stack_;
    // node_type derivation_;
    // node_type reduced_;
    
  public:
    static const size_type offset_state      = 0;
    static const size_type offset_stack      = offset_state + sizeof(state_type);
    static const size_type offset_derivation = offset_stack + sizeof(pointer);
    static const size_type offset_reduced    = offset_derivation + sizeof(pointer);
    
  public:
    static size_type size(const size_type rows=0)
    {
      return offset_reduced + sizeof(pointer);
    }
    
    inline const state_type& state() const { return *reinterpret_cast<const state_type*>(buffer_ + offset_state); }
    inline       state_type& state()       { return *reinterpret_cast<state_type*>(buffer_ + offset_state); }
    
    inline const node_type& stack() const { return *reinterpret_cast<const node_type*>(buffer_ + offset_stack); }
    inline       node_type& stack()       { return *reinterpret_cast<node_type*>(buffer_ + offset_stack); }
    
    inline const node_type& derivation() const { return *reinterpret_cast<const node_type*>(buffer_ + offset_derivation); }
    inline       node_type& derivation()       { return *reinterpret_cast<node_type*>(buffer_ + offset_derivation); }
    
    inline const node_type& reduced() const { return *reinterpret_cast<const node_type*>(buffer_ + offset_reduced); }
    inline       node_type& reduced()       { return *reinterpret_cast<node_type*>(buffer_ + offset_reduced); }
    
  public:
    
    friend
    bool operator==(const node_type& x, const node_type& y)
    {
      return x.buffer_ == y.buffer_;
    }

    friend
    bool operator!=(const node_type& x, const node_type& y)
    {
      return x.buffer_ != y.buffer_;
    }

    friend
    size_t  hash_value(node_type const& x)
    {
      return boost::hash<pointer>()(x.buffer_);
    }
    
  private:
    pointer buffer_;
  };
};

#endif

// include allocator
#include <rnnp/allocator.hpp>
