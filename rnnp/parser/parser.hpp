// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__PARSER__MODEL__HPP__
#define __RNNP__PARSER__MODEL__HPP__ 1

#include <rnnp/operation.hpp>
#include <rnnp/sentence.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/signature.hpp>
#include <rnnp/model.hpp>
#include <rnnp/span.hpp>
#include <rnnp/state.hpp>
#include <rnnp/feature_set.hpp>
#include <rnnp/dot_product.hpp>

namespace rnnp
{
  namespace parser
  {
    class Parser
    {
    public:
      typedef size_t    size_type;
      typedef ptrdiff_t difference_type;

      typedef Span span_type;
      
      typedef Sentence  sentence_type;
      
      typedef Grammar   grammar_type;
      typedef Signature signature_type;

      typedef rnnp::Model model_type;
      
      typedef model_type::symbol_type    symbol_type;
      typedef model_type::word_type      word_type;
      typedef model_type::parameter_type parameter_type;
      typedef model_type::tensor_type    tensor_type;
      typedef model_type::matrix_type    matrix_type;

      typedef rnnp::FeatureSet feature_set_type;
      
      typedef Operation operation_type;
      
      typedef State state_type;

      typedef state_type::feature_state_type  feature_state_type;
      typedef state_type::feature_vector_type feature_vector_type;
    };
  };
};

#endif
