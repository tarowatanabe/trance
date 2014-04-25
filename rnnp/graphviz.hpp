// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRAPHVIZ__HPP__
#define __RNNP__GRAPHVIZ__HPP__ 1

#include <iostream>

#include <rnnp/tree.hpp>

namespace rnnp
{
  struct Graphviz
  {
    typedef Tree tree_type;
    
    std::ostream& operator()(std::ostream& os, const tree_type& tree);
  };
  
  std::ostream& graphviz(std::ostream& os, const Tree& tree)
  {
    Graphviz output;
    return output(os, tree);
  }
};

#endif
