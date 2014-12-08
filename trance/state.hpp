// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__STATE__HPP__
#define __TRANCE__STATE__HPP__ 1

#include <trance/operation.hpp>
#include <trance/model.hpp>
#include <trance/span.hpp>
#include <trance/feature_state.hpp>
#include <trance/feature_vector.hpp>

#include <boost/functional/hash/hash.hpp>

namespace trance
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

    typedef Span span_type;

    typedef Model model_type;
    
    typedef model_type::symbol_type    symbol_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::adapted_type   adapted_type;

    typedef FeatureState                                                   feature_state_type;
    typedef FeatureVector<parameter_type, std::allocator<parameter_type> > feature_vector_type;
    typedef feature_vector_type*                                           feature_vector_ptr_type;
    
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
    // index_type unary_;

    // operation_type operation_;
    // symbol_type label_;
    // symbol_tyep head_;
    // span_type   span_;
    
    // stack and derivation
    // state_type stack_;
    // state_type derivation_;
    // state_type reduced_;

    // feature state
    // feature_state_type  featur_state_;
    // feature_vector_type* featur_vector_;
    
    // scoring
    // score_type score_;
    
    // neural network
    // tensor_type layer_;
    
  public:
    static const size_type offset_step       = 0;
    static const size_type offset_next       = offset_step + sizeof(index_type);
    static const size_type offset_unary      = offset_next + sizeof(index_type);

    static const size_type offset_operation  = offset_unary + sizeof(index_type);

    static const size_type offset_label      = offset_operation + sizeof(operation_type);
    static const size_type offset_head       = offset_label + sizeof(symbol_type);
    static const size_type offset_span       = offset_head + sizeof(symbol_type);
    
    static const size_type offset_stack      = (offset_span + sizeof(span_type) + 15) & (~15);
    static const size_type offset_derivation = offset_stack + sizeof(pointer);
    static const size_type offset_reduced    = offset_derivation + sizeof(pointer);
    
    static const size_type offset_feature_state  = (offset_reduced + sizeof(pointer) + 15) & (~15);
    static const size_type offset_feature_vector = offset_feature_state + sizeof(feature_state_type);
    
    static const size_type offset_score      = offset_feature_vector + sizeof(feature_vector_type*);
    
    static const size_type offset_layer      = (offset_score + sizeof(score_type) + 15) & (~15);
    
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

    inline const index_type& unary() const { return *reinterpret_cast<const index_type*>(buffer_ + offset_unary); }
    inline       index_type& unary()       { return *reinterpret_cast<index_type*>(buffer_ + offset_unary); }
    
    inline const operation_type& operation() const { return *reinterpret_cast<const operation_type*>(buffer_ + offset_operation); }
    inline       operation_type& operation()       { return *reinterpret_cast<operation_type*>(buffer_ + offset_operation); }

    inline const symbol_type& label() const { return *reinterpret_cast<const symbol_type*>(buffer_ + offset_label); }
    inline       symbol_type& label()       { return *reinterpret_cast<symbol_type*>(buffer_ + offset_label); }

    inline const symbol_type& head() const { return *reinterpret_cast<const symbol_type*>(buffer_ + offset_head); }
    inline       symbol_type& head()       { return *reinterpret_cast<symbol_type*>(buffer_ + offset_head); }
    
    inline const span_type& span() const { return *reinterpret_cast<const span_type*>(buffer_ + offset_span); }
    inline       span_type& span()       { return *reinterpret_cast<span_type*>(buffer_ + offset_span); }

    inline const state_type& stack() const { return *reinterpret_cast<const state_type*>(buffer_ + offset_stack); }
    inline       state_type& stack()       { return *reinterpret_cast<state_type*>(buffer_ + offset_stack); }

    inline const state_type& derivation() const { return *reinterpret_cast<const state_type*>(buffer_ + offset_derivation); }
    inline       state_type& derivation()       { return *reinterpret_cast<state_type*>(buffer_ + offset_derivation); }

    inline const state_type& reduced() const { return *reinterpret_cast<const state_type*>(buffer_ + offset_reduced); }
    inline       state_type& reduced()       { return *reinterpret_cast<state_type*>(buffer_ + offset_reduced); }

    inline const state_type& left() const { return reduced(); }
    inline       state_type& left()       { return reduced(); }
    
    inline const state_type& right() const { return derivation(); }
    inline       state_type& right()       { return derivation(); }
    
    inline const feature_state_type& feature_state() const { return *reinterpret_cast<const feature_state_type*>(buffer_ + offset_feature_state); } 
    inline       feature_state_type& feature_state()       { return *reinterpret_cast<feature_state_type*>(buffer_ + offset_feature_state); } 

    inline const feature_vector_ptr_type& feature_vector() const { return *reinterpret_cast<const feature_vector_ptr_type*>(buffer_ + offset_feature_vector); }
    inline       feature_vector_ptr_type& feature_vector()       { return *reinterpret_cast<feature_vector_ptr_type*>(buffer_ + offset_feature_vector); }

    inline const score_type& score() const { return *reinterpret_cast<const score_type*>(buffer_ + offset_score); }
    inline       score_type& score()       { return *reinterpret_cast<score_type*>(buffer_ + offset_score); }
    
    inline const adapted_type layer(const size_type rows) const
    {
      return adapted_type(const_cast<parameter_type*>(reinterpret_cast<const parameter_type*>(buffer_ + offset_layer)), rows, 1);
    }

    inline       adapted_type layer(const size_type rows)
    {
      return adapted_type(reinterpret_cast<parameter_type*>(buffer_ + offset_layer), rows, 1);
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
#include <trance/allocator.hpp>
