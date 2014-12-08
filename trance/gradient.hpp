// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__GRADIENT__HPP__
#define __TRANCE__GRADIENT__HPP__ 1

#include <cmath>
#include <iostream>

#include <trance/symbol.hpp>
#include <trance/model.hpp>
#include <trance/feature.hpp>
#include <trance/feature_vector.hpp>


#include <utils/atomicop.hpp>
#include <utils/bithack.hpp>
#include <utils/unordered_map.hpp>

#include <Eigen/Core>

namespace trance
{
  class Gradient
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Symbol symbol_type;
    typedef Symbol word_type;
    
    typedef Feature feature_type;
    
    typedef Model model_type;

    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::adapted_type   adapted_type;
    
    typedef utils::unordered_map<word_type, tensor_type,
				 boost::hash<word_type>, std::equal_to<word_type>,
				 std::allocator<std::pair<const word_type, tensor_type> > >::type matrix_embedding_type;

    typedef FeatureVector<parameter_type, std::allocator<parameter_type> > weights_type;
    
    typedef symbol_type category_type;
    
    struct category_hash
    {
      size_t operator()(const category_type& x) const
      {
	return x.non_terminal_id();
      }
    };
    
    typedef utils::unordered_map<category_type, tensor_type,
				 category_hash, std::equal_to<category_type>,
				 std::allocator<std::pair<const category_type, tensor_type> > >::type matrix_category_type;
    
  public:
    Gradient() : hidden_(0), embedding_(0), count_(0), shared_(0) {}
    Gradient(const model_type& model)
      : count_(0), shared_(0) { initialize(model.hidden_, model.embedding_); }
    Gradient(const size_type& hidden,
	     const size_type& embedding)
      : count_(0), shared_(0){ initialize(hidden, embedding); }
    
    void initialize(const size_type& hidden,
		    const size_type& embedding);
    
    void write_matrix(std::ostream& os, const matrix_embedding_type& matrix) const;
    void write_matrix(std::ostream& os, const matrix_category_type& matrix) const;
    void write_matrix(std::ostream& os, const weights_type& matrix) const;
    void write_matrix(std::ostream& os, const tensor_type& matrix) const;
    
    void read_matrix(std::istream& is, matrix_embedding_type& matrix);
    void read_matrix(std::istream& is, matrix_category_type& matrix);
    void read_matrix(std::istream& is, weights_type& matrix);
    void read_matrix(std::istream& is, tensor_type& matrix);

  public:
    matrix_embedding_type& plus_equal(matrix_embedding_type& x, const matrix_embedding_type& y);
    matrix_embedding_type& minus_equal(matrix_embedding_type& x, const matrix_embedding_type& y);

    matrix_category_type& plus_equal(matrix_category_type& x, const matrix_category_type& y);
    matrix_category_type& minus_equal(matrix_category_type& x, const matrix_category_type& y);

    weights_type& plus_equal(weights_type& x, const weights_type& y)
    {
      x += y;
      return x;
    }
    weights_type& minus_equal(weights_type& x, const weights_type& y)
    {
      x -= y;
      return x;
    }
    
    tensor_type& plus_equal(tensor_type& x, const tensor_type& y);
    tensor_type& minus_equal(tensor_type& x, const tensor_type& y);
    
  public:
    void swap(Gradient& x)
    {
      std::swap(hidden_,    x.hidden_);
      std::swap(embedding_, x.embedding_);
      std::swap(count_,     x.count_);
      std::swap(shared_,    x.shared_);
    }

    void clear()
    {
      count_ = 0;
      shared_ = 0;
    }
    
  public:
    void increment()
    {
      utils::atomicop::add_and_fetch(shared_, size_type(1));
    }
    
    size_type shared() const
    {
      const size_type ret = shared_;
      utils::atomicop::memory_barrier();
      return ret;
    }
    
  public:
    // parameters
    size_type hidden_;
    size_type embedding_;
    size_type count_;
    size_type shared_;
  };
};

namespace std
{
  inline
  void swap(trance::Gradient& x, trance::Gradient& y)
  {
    x.swap(y);
  }
};

#endif
