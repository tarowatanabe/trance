// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__FEATURE_SET__HPP__
#define __RNNP__FEATURE_SET__HPP__ 1

#include <string>
#include <vector>

#include <rnnp/model.hpp>
#include <rnnp/allocator.hpp>
#include <rnnp/feature_function.hpp>

#include <utils/piece.hpp>

namespace rnnp
{

  struct FeatureState;

  class FeatureSet
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef Model model_type;
    
    typedef model_type::parameter_type parameter_type;
    
    typedef FeatureFunction feature_function_type;

    typedef feature_function_type::feature_function_ptr_type feature_function_ptr_type;
    typedef feature_function_type::feature_type              feature_type;
    typedef feature_function_type::feature_vector_type       feature_vector_type;
    
    typedef feature_function_type::operation_type operation_type;
    typedef feature_function_type::symbol_type    symbol_type;
    typedef feature_function_type::word_type      word_type;

    typedef FeatureState state_type;

  private:
    typedef std::vector<feature_function_ptr_type, std::allocator<feature_function_ptr_type> > impl_type;
    typedef std::vector<size_type, std::allocator<size_type> > offset_set_type;
    typedef Allocator<state_type>  state_allocator_type;
    
  public:
    typedef impl_type::const_iterator  const_iterator;
    typedef impl_type::iterator        iterator;
    typedef impl_type::const_reference const_reference;
    typedef impl_type::reference       reference;

  public:
    FeatureSet() {}
    template <typename Iterator>
    FeatureSet(Iterator first, Iterator last)
    {
      for (/**/; first != last; ++ first)
	push_back(*first);
    }

  public:
    inline const_iterator begin() const { return impl_.begin(); }
    inline       iterator begin()       { return impl_.begin(); }
    inline const_iterator end() const { return impl_.end(); }
    inline       iterator end()       { return impl_.end(); }

    inline const_reference operator[](size_type x) const { return impl_[x]; }
    inline       reference operator[](size_type x)       { return impl_[x]; }

    void push_back(const feature_function_ptr_type& x) { impl_.push_back(x); }
    void push_back(const utils::piece& param) { impl_.push_back(create(param)); }
    
    bool empty() const { return impl_.empty(); }
    size_type size() const { return impl_.size(); }

  public:
    state_type apply(const operation_type& operation,
		     feature_vector_type& features) const;

    state_type apply(const operation_type& operation,
		     const symbol_type& label,
		     const word_type& head,
		     feature_vector_type& features) const;
    
    state_type apply(const operation_type& operation,
		     const symbol_type& label,
		     const state_type& state_top,
		     const state_type& state_next,
		     feature_vector_type& features) const;
    
    state_type apply(const operation_type& operation,
		     const symbol_type& label,
		     const state_type& state_top,
		     feature_vector_type& features) const;

    state_type apply(const operation_type& operation,
		     const state_type& state_top,
		     feature_vector_type& features) const;
    
  public:
    void deallocate(const state_type& state)
    {
      allocator_.deallocate(state);
    }
    
  public:
    static feature_function_ptr_type create(const utils::piece& param);
    static std::string usage();
    
  public:
    void initialize();

    void swap(FeatureSet& x)
    {
      impl_.swap(x.impl_);
      allocator_.swap(x.allocator_);
      
      offsets_.swap(x.offsets_);
      std::swap(size_, x.size_);
    }
    
    FeatureSet clone() const
    {
      FeatureSet feats;
      
      const_iterator fiter_end = impl_.end();
      for (const_iterator fiter = impl_.begin(); fiter != fiter_end; ++ fiter)
	feats.push_back((*fiter)->clone());
      
      feats.allocator_ = allocator_;
      feats.offsets_   = offsets_;
      feats.size_      = size_;
      
      return feats;
    }
    
  private:
    impl_type            impl_;
    state_allocator_type allocator_;
    
    // offset to compute states and the size
    offset_set_type  offsets_;
    size_type        size_;
  };
};

namespace std
{
  inline
  void swap(rnnp::FeatureSet& x, rnnp::FeatureSet& y)
  {
    x.swap(y);
  }
};

#endif

#include <rnnp/feature_state.hpp>
#include <rnnp/feature_function.hpp>
