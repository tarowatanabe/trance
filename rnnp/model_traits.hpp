// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__MODEL_TRAITS__HPP__
#define __RNNP__MODEL_TRAITS__HPP__ 1

#include <rnnp/gradient/model1.hpp>
#include <rnnp/gradient/model2.hpp>
#include <rnnp/gradient/model3.hpp>

#include <rnnp/model/model1.hpp>
#include <rnnp/model/model2.hpp>
#include <rnnp/model/model3.hpp>

#include <rnnp/parser/model1.hpp>
#include <rnnp/parser/model2.hpp>
#include <rnnp/parser/model3.hpp>

namespace rnnp
{
  template <typename M>
  struct model_traits { };
    
  template <>
  struct model_traits<model::Model1>
  {
    typedef model::Model1    model_type;
    typedef gradient::Model1 gradient_type;
    typedef parser::Model1   parser_type;
  };
    
  template <>
  struct model_traits<model::Model2>
  {
    typedef model::Model2    model_type;
    typedef gradient::Model2 gradient_type;
    typedef parser::Model2   parser_type;
  };
    
  template <>
  struct model_traits<model::Model3>
  {
    typedef model::Model3    model_type;
    typedef gradient::Model3 gradient_type;
    typedef parser::Model3   parser_type;
  };

  template <>
  struct model_traits<gradient::Model1>
  {
    typedef model::Model1    model_type;
    typedef gradient::Model1 gradient_type;
    typedef parser::Model1   parser_type;
  };
    
  template <>
  struct model_traits<gradient::Model2>
  {
    typedef model::Model2    model_type;
    typedef gradient::Model2 gradient_type;
    typedef parser::Model2   parser_type;
  };
    
  template <>
  struct model_traits<gradient::Model3>
  {
    typedef model::Model3    model_type;
    typedef gradient::Model3 gradient_type;
    typedef parser::Model3   parser_type;
  };

  template <>
  struct model_traits<parser::Model1>
  {
    typedef model::Model1    model_type;
    typedef gradient::Model1 gradient_type;
    typedef parser::Model1   parser_type;
  };
    
  template <>
  struct model_traits<parser::Model2>
  {
    typedef model::Model2    model_type;
    typedef gradient::Model2 gradient_type;
    typedef parser::Model2   parser_type;
  };
    
  template <>
  struct model_traits<parser::Model3>
  {
    typedef model::Model3    model_type;
    typedef gradient::Model3 gradient_type;
    typedef parser::Model3   parser_type;
  };
};


#endif
