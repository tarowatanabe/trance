// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__PARSER__MODEL__HPP__
#define __TRANCE__PARSER__MODEL__HPP__ 1

#include <trance/operation.hpp>
#include <trance/sentence.hpp>
#include <trance/grammar.hpp>
#include <trance/signature.hpp>
#include <trance/model.hpp>
#include <trance/span.hpp>
#include <trance/state.hpp>
#include <trance/feature_set.hpp>
#include <trance/dot_product.hpp>

namespace trance
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

      typedef trance::Model model_type;
      
      typedef model_type::symbol_type    symbol_type;
      typedef model_type::word_type      word_type;
      typedef model_type::parameter_type parameter_type;
      typedef model_type::tensor_type    tensor_type;
      typedef model_type::adapted_type   adapted_type;

      typedef trance::FeatureSet feature_set_type;
      
      typedef Operation operation_type;
      
      typedef State state_type;

      typedef state_type::feature_state_type  feature_state_type;
      typedef state_type::feature_vector_type feature_vector_type;
    };
  };
};

#endif
