// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__FEATURE__PENALTY__HPP__
#define __RNNP__FEATURE__PENALTY__HPP__ 1

#include <rnnp/feature_function.hpp>

namespace rnnp
{
  namespace feature
  {
    class OperationPenalty : public FeatureFunction
    {
    public:
      OperationPenalty()
	: FeatureFunction("operation-penalty", 0),
	  name_shift_("operation-penalty:shift"),
	  name_reduce_("operation-penalty:reduce"),
	  name_unary_("operation-penalty:unary"),
	  name_final_("operation-penalty:final"),
	  name_idle_("operation-penalty:idle")
      { }
      
      // cloning
      virtual feature_function_ptr_type clone() const { return feature_function_ptr_type(new OperationPenalty(*this)); }
      
      // feature application for axiom
      virtual void apply(const operation_type& operation,
			 state_type state,
			 feature_vector_type& features) const { }
      
      // feature application for shift
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const word_type& head,
			 state_type state,
			 feature_vector_type& features) const
      {
	features[name_shift_] = -1;
      }
      
      // feature application for reduce
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const state_type state_top,
			 const state_type state_next,
			 state_type state,
			 feature_vector_type& features) const
      {
	features[name_reduce_] = -1;
      }
      
      // feature application for unary
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const state_type state_top,
			 state_type state,
			 feature_vector_type& features) const
      {
	features[name_unary_] = -1;
      }
      
      // feature application for final/idle
      virtual void apply(const operation_type& operation,
			 const state_type state_top,
			 state_type state,
			 feature_vector_type& features) const
      {
	if (operation.final())
	  features[name_final_] = -1;
	else if (operation.idle())
	  features[name_idle_] = -1;
      }
      
    private:
      feature_type name_shift_;
      feature_type name_reduce_;
      feature_type name_unary_;
      feature_type name_final_;
      feature_type name_idle_;
    };

    struct FactoryPenalty : public rnnp::FeatureFunctionFactory
    {
      feature_function_ptr_type create(const std::string& param) const
      {
	typedef rnnp::Option option_type;
	typedef boost::filesystem::path path_type;
	
	const option_type option(param);
	
	if (utils::ipiece(option.name()) == "operation-penalty")
	  return feature_function_ptr_type(new OperationPenalty());
	else
	  throw std::runtime_error("unsupported penalty feature" + option.name());
      }
      
      bool supported(const std::string& param) const
      {
	typedef rnnp::Option option_type;
	
	const option_type option(param);
	
	utils::ipiece name = option.name();
	
	return (name == "operation-penalty");
      }

      std::string usage() const
      {
	static const char* desc = "\
operation-penalty: penalty by the # of operations\n\
";
	return desc;
      }
    };

  }
}

#endif
