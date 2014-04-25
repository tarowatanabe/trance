// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__BINARIZE_LEFT__HPP__
#define __RNNP__BINARIZE_LEFT__HPP__ 1

// left-recursive binarization (or left heavy binarization)

#include <rnnp/tree.hpp>

namespace rnnp
{
  struct BinarizeLeft
  {
    typedef Tree tree_type;

    typedef tree_type::symbol_type symbol_type;
    
    void operator()(const tree_type& source, tree_type& target)
    {
      binarize(source, target);
    }
    
    void binarize(const tree_type& source, tree_type& target)
    {
      if (source.antecedent_.empty())
	target = source;
      else if (source.antecedent_.size() <= 2) {
	target.label_ = source.label_;
	target.antecedent_.resize(source.antecedent_.size());
	
	for (size_t i = 0; i != source.antecedent_.size(); ++ i)
	  binarize(source.antecedent_[i], target.antecedent_[i]);
      } else {
	target.label_ = source.label_;
	target.antecedent_.resize(2);
	
	// left-heavy binarization
	binarize("[" + source.label_.strip() + "^]",
		 source.antecedent_.begin(), source.antecedent_.end() - 1,
		 target.antecedent_.front());
	
	binarize(source.antecedent_.back(), target.antecedent_.back());
      }
    }

    template <typename Iterator>
    void binarize(const symbol_type& label, Iterator first, Iterator last, tree_type& target)
    {
      target.label_ = label;
      target.antecedent_.resize(2);

      if (std::distance(first, last) == 2) {
	binarize(*first,       target.antecedent_.front());
	binarize(*(first + 1), target.antecedent_.back());
      } else {
	binarize(label, first, last - 1, target.antecedent_.front());
	binarize(*(last - 1),            target.antecedent_.back());
      }
    }
  };

  inline
  void binarize_left(const Tree& source, Tree& target)
  {
    BinarizeLeft binarize;
    binarize(source, target);
  }
  
  inline
  void binarize_left(Tree& tree)
  {
    Tree binarized;
    binarize_left(tree, binarized);
    binarized.swap(tree);
  }
};

#endif
