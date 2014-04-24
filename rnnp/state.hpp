// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__STATE__HPP__
#define __RNNP__STATE__HPP__ 1

#include <rnnp/operation.hpp>
#include <rnnp/model.hpp>

#include <boost/functional/hash/hash.hpp>

namespace rnnp
{
  template <typename Tp>
  class Allocator;
  
  struct State
  {
    friend class Allocator<State>;
    
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;

    typedef Model model_type;
    
    typedef model_type::symbol_type    symbol_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
    typedef State state_type;
    typedef char* pointer;
    
    typedef double score_type;

    typedef Operation  operation_type;
    
  public:
    State() : buffer_(0) {}
    
  private:
    State(pointer buffer) : buffer_(buffer) {}
    
  public:
    bool empty() const { return ! buffer_; }
    
    operator bool() const { return ! empty(); }
    
  private:
    // state information
    // index_type step_;
    // index_type next_;

    // label
    // symbol_type label_;
    
    // operation
    // operation_type operation_;
    
    // scoring
    // score_type score_;
    
    // neural network
    // tensor_type layer_;
    
  public:
    static const size_type offset_step      = 0;
    static const size_type offset_next      = offset_step + sizeof(index_type);
    static const size_type offset_label     = offset_next + sizeof(index_type);
    
    static const size_type offset_operation = offset_label + sizeof(symbol_type);
    
    static const size_type offset_score     = offset_operation + sizeof(operation_type);
    
    static const size_type offset_layer     = (offset_score + sizeof(score_type) + 15) & (~15);
    
  public:
    static size_type size(const size_type rows)
    {
      return offset_layer + rows * sizeof(parameter_type);
    }
    
  public:
    inline const index_type& step() const { return *reinterpret_cast<const index_type*>(buffer_ + offset_step); }
    inline       index_type& step()       { return *reinterpret_cast<index_type*>(buffer_ + offset_step); }

    inline const index_type& next() const { return *reinterpret_cast<const index_type*>(buffer_ + offset_next); }
    inline       index_type& next()       { return *reinterpret_cast<index_type*>(buffer_ + offset_next); }

    inline const symbol_type& label() const { return *reinterpret_cast<const symbol_type*>(buffer_ + offset_label); }
    inline       symbol_type& label()       { return *reinterpret_cast<symbol_type*>(buffer_ + offset_label); }
    
    inline const operation_type& operation() const { return *reinterpret_cast<const operation_type*>(buffer_ + offset_operation); }
    inline       operation_type& operation()       { return *reinterpret_cast<operation_type*>(buffer_ + offset_operation); }
    
    inline const score_type& score() const { return *reinterpret_cast<const score_type*>(buffer_ + offset_score); }
    inline       score_type& score()       { return *reinterpret_cast<score_type*>(buffer_ + offset_score); }
    
    inline const matrix_type layer(const size_type rows) const
    {
      return matrix_type(const_cast<parameter_type*>(reinterpret_cast<const parameter_type*>(buffer_ + offset_layer)), rows, 1);
    }

    inline       matrix_type layer(const size_type rows)
    {
      return matrix_type(reinterpret_cast<parameter_type*>(buffer_ + offset_layer), rows, 1);
    }
    
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
#include <rnnp/allocator.hpp>
