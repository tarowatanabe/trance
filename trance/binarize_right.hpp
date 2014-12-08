// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__BINARIZE_RIGHT__HPP__
#define __TRANCE__BINARIZE_RIGHT__HPP__ 1

// right-recursive binarization (or right heavy binarization)

#include <trance/tree.hpp>

namespace trance
{
  struct BinarizeRight
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
	
	// right-heavy binarization
	binarize(source.antecedent_.front(), target.antecedent_.front());
	
	binarize("[" + source.label_.strip() + "^]",
		 source.antecedent_.begin() + 1, source.antecedent_.end(),
		 target.antecedent_.back());
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
	binarize(*first,       target.antecedent_.front());
	binarize(label, first + 1, last, target.antecedent_.back());
      }
    }
  };

  inline
  void binarize_right(const Tree& source, Tree& target)
  {
    BinarizeRight binarize;
    binarize(source, target);
  }
  
  inline
  void binarize_right(Tree& tree)
  {
    Tree binarized;
    binarize_right(tree, binarized);
    binarized.swap(tree);
  }
};

#endif
