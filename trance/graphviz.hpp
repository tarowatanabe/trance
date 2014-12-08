// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__GRAPHVIZ__HPP__
#define __TRANCE__GRAPHVIZ__HPP__ 1

#include <iostream>

#include <trance/tree.hpp>

namespace trance
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
