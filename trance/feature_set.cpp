//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <stdexcept>

#include "trance/option.hpp"
#include "trance/feature_set.hpp"

#include "trance/feature/grammar.hpp"
#include "trance/feature/penalty.hpp"

#include "utils/unordered_set.hpp"

namespace trance
{
  FeatureSet::state_type FeatureSet::apply(const operation_type& operation,
					   feature_vector_type& features) const
  {
    typedef feature_function_type::state_type feature_state_type;
    
    if (! operation.axiom())
      throw std::runtime_error("invalid operation");
    
    state_type state = const_cast<state_allocator_type&>(allocator_).allocate();
    
    for (size_t i = 0; i != impl_.size(); ++ i)
      impl_[i]->apply(operation,
		      reinterpret_cast<feature_state_type>(state.buffer_ + offsets_[i]),
		      features);
    
    return state;
  }
  
  FeatureSet::state_type FeatureSet::apply(const operation_type& operation,
					   const symbol_type& label,
					   const word_type& head,
					   feature_vector_type& features) const
  {
    typedef feature_function_type::state_type feature_state_type;
    
    if (! operation.shift())
      throw std::runtime_error("invalid operation");
    
    state_type state = const_cast<state_allocator_type&>(allocator_).allocate();

    for (size_t i = 0; i != impl_.size(); ++ i)
      impl_[i]->apply(operation,
		      label,
		      head,
		      reinterpret_cast<feature_state_type>(state.buffer_ + offsets_[i]),
		      features);

    return state;
  }
  
  FeatureSet::state_type FeatureSet::apply(const operation_type& operation,
					   const symbol_type& label,
					   const state_type& state_top,
					   const state_type& state_next,
					   feature_vector_type& features) const
  {
    typedef feature_function_type::state_type feature_state_type;
    
    if (! operation.reduce())
      throw std::runtime_error("invalid operation");
    
    state_type state = const_cast<state_allocator_type&>(allocator_).allocate();
    
    for (size_t i = 0; i != impl_.size(); ++ i)
      impl_[i]->apply(operation,
		      label,
		      reinterpret_cast<const feature_state_type>(state_top.buffer_ + offsets_[i]),
		      reinterpret_cast<const feature_state_type>(state_next.buffer_ + offsets_[i]),
		      reinterpret_cast<feature_state_type>(state.buffer_ + offsets_[i]),
		      features);
    
    return state;
  }

  FeatureSet::state_type FeatureSet::apply(const operation_type& operation,
					   const symbol_type& label,
					   const state_type& state_top,
					   feature_vector_type& features) const
  {
    typedef feature_function_type::state_type feature_state_type;
    
    if (! operation.unary())
      throw std::runtime_error("invalid operation");
    
    state_type state = const_cast<state_allocator_type&>(allocator_).allocate();
    
    for (size_t i = 0; i != impl_.size(); ++ i)
      impl_[i]->apply(operation,
		      label,
		      reinterpret_cast<const feature_state_type>(state_top.buffer_ + offsets_[i]),
		      reinterpret_cast<feature_state_type>(state.buffer_ + offsets_[i]),
		      features);
    
    return state;
  }

  FeatureSet::state_type FeatureSet::apply(const operation_type& operation,
					   const state_type& state_top,
					   feature_vector_type& features) const
  {
    typedef feature_function_type::state_type feature_state_type;
    
    if (! operation.finished())
      throw std::runtime_error("invalid operation");
    
    state_type state = const_cast<state_allocator_type&>(allocator_).allocate();
    
    for (size_t i = 0; i != impl_.size(); ++ i)
      impl_[i]->apply(operation,
		      reinterpret_cast<const feature_state_type>(state_top.buffer_ + offsets_[i]),
		      reinterpret_cast<feature_state_type>(state.buffer_ + offsets_[i]),
		      features);
    
    return state;
  }

  void FeatureSet::initialize()
  {
    typedef utils::unordered_set<feature_type, boost::hash<feature_type>, std::equal_to<std::string>,
				 std::allocator<feature_type> >::type feature_unique_type;    
    
    feature_unique_type features;
    
    offsets_.clear();
    offsets_.reserve(impl_.size());
    offsets_.resize(impl_.size());
    size_ = 0;
    
    const size_type alignment_size = utils::bithack::max(sizeof(void*), size_type(16));
    const size_type alignment_mask = ~(alignment_size - 1);
    
    for (size_t i = 0; i != impl_.size(); ++ i) {
      offsets_[i] = size_;
      
      if (impl_[i]->size_)
	size_ += (impl_[i]->size_ + alignment_size - 1) & alignment_mask;

      if (! features.insert(impl_[i]->name_).second)
	throw std::runtime_error("you have already registered feature: " + static_cast<const std::string&>(impl_[i]->name_) + " try use different feature name by name=<feature name> option");
    }
    
    allocator_.clear();
    allocator_.assign(size_);
  }


  FeatureSet::feature_function_ptr_type FeatureSet::create(const utils::piece& param)
  {
    if (feature::FactoryGrammar().supported(param))
      return feature::FactoryGrammar().create(param);
    else if (feature::FactoryPenalty().supported(param))
      return feature::FactoryPenalty().create(param);
    else
      throw std::runtime_error("unsupported parameter: " + param);
  }
  
  std::string FeatureSet::usage()
  {
    std::string desc;

    desc += feature::FactoryGrammar().usage();
    desc += feature::FactoryPenalty().usage();
    
    return desc;
  }
};
