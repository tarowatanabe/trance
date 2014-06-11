// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__HPP__
#define __RNNP__OBJECTIVE__HPP__ 1

#include <vector>

#include <rnnp/derivation.hpp>
#include <rnnp/parser.hpp>
#include <rnnp/parser_oracle.hpp>
#include <rnnp/tree.hpp>
#include <rnnp/model.hpp>
#include <rnnp/gradient.hpp>
#include <rnnp/learn_option.hpp>
#include <rnnp/loss.hpp>

#include <utils/unordered_map.hpp>
#include <utils/compact_set.hpp>

namespace rnnp
{
  struct Objective
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef Model    model_type;
    typedef Gradient gradient_type;
    
    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;

    typedef LearnOption option_type;

    typedef Parser       parser_type;
    typedef ParserOracle parser_oracle_type;
    typedef Derivation   derivation_type;    

    typedef parser_type::sentence_type  sentence_type;
    typedef parser_type::operation_type operation_type;
    typedef parser_type::state_type     state_type;
    typedef parser_type::heap_type      heap_type;
    typedef parser_type::agenda_type    agenda_type;

    typedef parser_type::feature_vector_type feature_vector_type;

    typedef Loss loss_type;
    
    struct backward_type
    {
      double      loss_;
      tensor_type delta_;
      
      backward_type() : loss_(0.0), delta_() {}
    };
    
    typedef utils::unordered_map<state_type, backward_type,
				 boost::hash<state_type>, std::equal_to<state_type>,
				 std::allocator<std::pair<const state_type, backward_type> > >::type backward_set_type;
    
    typedef utils::compact_set<state_type, 
			       utils::unassigned<state_type>, utils::unassigned<state_type>,
			       boost::hash<state_type>, std::equal_to<state_type>,
			       std::allocator<state_type> > state_set_type;
    typedef std::vector<state_set_type, std::allocator<state_set_type> > state_map_type;
    
    void initialize(const parser_type& candidates,
		    const parser_oracle_type& oracles)
    {
      backward_.clear();
      
      states_.clear();
      states_.resize(std::max(candidates.agenda_.size(), oracles.agenda_.size()));
    }
    
    derivation_type   derivation_;
    backward_set_type backward_;
    state_map_type    states_;
    
    tensor_type queue_;
  };
  
};

#endif
