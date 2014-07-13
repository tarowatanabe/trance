// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPTIMIZE__HPP__
#define __RNNP__OPTIMIZE__HPP__ 1

#include <rnnp/model.hpp>
#include <rnnp/gradient.hpp>
#include <rnnp/learn_option.hpp>

namespace rnnp
{
  struct Optimize
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Model    model_type;
    typedef Gradient gradient_type;
    
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::adapted_type   adapted_type;

    typedef gradient_type::matrix_embedding_type matrix_embedding_type;
    typedef gradient_type::matrix_category_type  matrix_category_type;

    typedef LearnOption option_type;
  };
};

#endif
