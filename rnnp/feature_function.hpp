// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__FEATURE_FUNCTION__HPP__
#define __RNNP__FEATURE_FUNCTION__HPP__ 1

#include <string>

#include <rnnp/operation.hpp>
#include <rnnp/model.hpp>
#include <rnnp/feature.hpp>
#include <rnnp/feature_vector.hpp>

#include <boost/shared_ptr.hpp>

namespace rnnp
{
  class FeatureFunction
  {
    friend class FeatureSet;
    
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

  public:
    typedef Model model_type;

    typedef model_type::symbol_type    symbol_type;
    typedef model_type::word_type      word_type;    
    typedef model_type::parameter_type parameter_type;
    
    typedef Operation  operation_type;
    
    typedef Feature feature_type;
    typedef FeatureVector<parameter_type, std::allocator<parameter_type> > feature_vector_type;
    
  public:
    typedef FeatureFunction feature_function_type;
    
    typedef boost::shared_ptr<feature_function_type> feature_function_ptr_type;
    
    typedef void* state_type;
    
  public:
    FeatureFunction(const std::string& name, const size_type& size)
      : name_(name), size_(size) {}
    virtual ~FeatureFunction() {}
    
  public:
    // cloning
    virtual feature_function_ptr_type clone() const = 0;
    
    // feature application for axiom
    virtual void apply(const operation_type& operation,
		       state_type state,
		       feature_vector_type& features) const = 0;
    
    // feature application for shift
    virtual void apply(const operation_type& operation,
		       const symbol_type& label,
		       const word_type& head,
		       state_type state,
		       feature_vector_type& features) const = 0;
    
    // feature application for reduce
    virtual void apply(const operation_type& operation,
		       const symbol_type& label,
		       const state_type state_top,
		       const state_type state_next,
		       state_type state,
		       feature_vector_type& features) const = 0;
    
    // feature application for unary
    virtual void apply(const operation_type& operation,
		       const symbol_type& label,
		       const state_type state_top,
		       state_type state,
		       feature_vector_type& features) const = 0;
    
    // feature application for final/idle
    virtual void apply(const operation_type& operation,
		       const state_type state_top,
		       state_type state,
		       feature_vector_type& features) const = 0;
    
  protected:
    feature_type name_;
    size_type    size_;
  };
  
  struct FeatureFunctionFactory
  {
    typedef FeatureFunction feature_function_type;
    
    typedef feature_function_type::feature_function_ptr_type feature_function_ptr_type;

    virtual feature_function_ptr_type create(const std::string& param) const = 0;

    virtual bool supported(const std::string& param) const = 0;
    
    virtual std::string usage() const = 0;
  };
};

#endif
