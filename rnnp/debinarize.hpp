// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__DEBINARIZE__HPP__
#define __RNNP__DEBINARIZE__HPP__ 1

// debinarization

#include <algorithm>
#include <vector>
#include <iterator>

#include <rnnp/tree.hpp>

namespace rnnp
{
  struct Debinarize
  {
    typedef Tree tree_type;

    typedef std::vector<tree_type, std::allocator<tree_type> > tree_set_type;
    
    void operator()(const tree_type& source, tree_type& target)
    {
      tree_set_type antecedent;
      
      debinarize(source.antecedent_.begin(), source.antecedent_.end(), std::back_inserter(antecedent));
      
      target.label_ = source.label_;
      target.antecedent_.resize(antecedent.size());
      
      std::swap_ranges(antecedent.begin(), antecedent.end(), target.antecedent_.begin());
    }
    
    template <typename Iterator, typename Output>
    void debinarize(Iterator first, Iterator last, Output output)
    {
      for (/**/; first != last; ++ first)
	if (first->label_.binarized())
	  debinarize(first->antecedent_.begin(), first->antecedent_.end(), output);
	else {
	  tree_set_type antecedent;
	  
	  debinarize(first->antecedent_.begin(), first->antecedent_.end(), std::back_inserter(antecedent));
	  
	  *output = tree_type(first->label_, antecedent.begin(), antecedent.end());
	  ++ output;
	}
    }
  };

  inline
  void debinarize(const Tree& source, Tree& target)
  {
    Debinarize debinarize;
    debinarize(source, target);
  }
  
  inline
  void debinarize(Tree& tree)
  {
    Tree debinarized;
    debinarize(tree, debinarized);
    debinarized.swap(tree);
  }
};

#endif
