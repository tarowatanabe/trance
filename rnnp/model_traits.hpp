// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__MODEL_TRAITS__HPP__
#define __RNNP__MODEL_TRAITS__HPP__ 1

#include <rnnp/gradient/model1.hpp>
#include <rnnp/gradient/model2.hpp>
#include <rnnp/gradient/model3.hpp>
#include <rnnp/gradient/model4.hpp>
#include <rnnp/gradient/model5.hpp>
#include <rnnp/gradient/model6.hpp>
#include <rnnp/gradient/model7.hpp>
#include <rnnp/gradient/model8.hpp>

#include <rnnp/model/model1.hpp>
#include <rnnp/model/model2.hpp>
#include <rnnp/model/model3.hpp>
#include <rnnp/model/model4.hpp>
#include <rnnp/model/model5.hpp>
#include <rnnp/model/model6.hpp>
#include <rnnp/model/model7.hpp>
#include <rnnp/model/model8.hpp>

#include <rnnp/parser/model1.hpp>
#include <rnnp/parser/model2.hpp>
#include <rnnp/parser/model3.hpp>
#include <rnnp/parser/model4.hpp>
#include <rnnp/parser/model5.hpp>
#include <rnnp/parser/model6.hpp>
#include <rnnp/parser/model7.hpp>
#include <rnnp/parser/model8.hpp>

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
  struct model_traits<model::Model4>
  {
    typedef model::Model4    model_type;
    typedef gradient::Model4 gradient_type;
    typedef parser::Model4   parser_type;
  };

  template <>
  struct model_traits<model::Model5>
  {
    typedef model::Model5    model_type;
    typedef gradient::Model5 gradient_type;
    typedef parser::Model5   parser_type;
  };

  template <>
  struct model_traits<model::Model6>
  {
    typedef model::Model6    model_type;
    typedef gradient::Model6 gradient_type;
    typedef parser::Model6   parser_type;
  };

  template <>
  struct model_traits<model::Model7>
  {
    typedef model::Model7    model_type;
    typedef gradient::Model7 gradient_type;
    typedef parser::Model7   parser_type;
  };

  template <>
  struct model_traits<model::Model8>
  {
    typedef model::Model8    model_type;
    typedef gradient::Model8 gradient_type;
    typedef parser::Model8   parser_type;
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
  struct model_traits<gradient::Model4>
  {
    typedef model::Model4    model_type;
    typedef gradient::Model4 gradient_type;
    typedef parser::Model4   parser_type;
  };

  template <>
  struct model_traits<gradient::Model5>
  {
    typedef model::Model5    model_type;
    typedef gradient::Model5 gradient_type;
    typedef parser::Model5   parser_type;
  };

  template <>
  struct model_traits<gradient::Model6>
  {
    typedef model::Model6    model_type;
    typedef gradient::Model6 gradient_type;
    typedef parser::Model6   parser_type;
  };

  template <>
  struct model_traits<gradient::Model7>
  {
    typedef model::Model7    model_type;
    typedef gradient::Model7 gradient_type;
    typedef parser::Model7   parser_type;
  };

  template <>
  struct model_traits<gradient::Model8>
  {
    typedef model::Model8    model_type;
    typedef gradient::Model8 gradient_type;
    typedef parser::Model8   parser_type;
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

  template <>
  struct model_traits<parser::Model4>
  {
    typedef model::Model4    model_type;
    typedef gradient::Model4 gradient_type;
    typedef parser::Model4   parser_type;
  };

  template <>
  struct model_traits<parser::Model5>
  {
    typedef model::Model5    model_type;
    typedef gradient::Model5 gradient_type;
    typedef parser::Model5   parser_type;
  };

  template <>
  struct model_traits<parser::Model6>
  {
    typedef model::Model6    model_type;
    typedef gradient::Model6 gradient_type;
    typedef parser::Model6   parser_type;
  };

  template <>
  struct model_traits<parser::Model7>
  {
    typedef model::Model7    model_type;
    typedef gradient::Model7 gradient_type;
    typedef parser::Model7   parser_type;
  };

  template <>
  struct model_traits<parser::Model8>
  {
    typedef model::Model8    model_type;
    typedef gradient::Model8 gradient_type;
    typedef parser::Model8   parser_type;
  };
};


#endif
