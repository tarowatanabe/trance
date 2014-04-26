// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRADIENT__HPP__
#define __RNNP__GRADIENT__HPP__ 1

#include <cmath>
#include <vector>

#include <rnnp/symbol.hpp>
#include <rnnp/model.hpp>

#include <utils/atomicop.hpp>
#include <utils/bithack.hpp>
#include <utils/unordered_map.hpp>

#include <Eigen/Core>

namespace rnnp
{
  class Gradient
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Symbol symbol_type;
    typedef Symbol word_type;
    
    typedef Model model_type;

    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;
    
    typedef utils::unordered_map<word_type, tensor_type,
				 boost::hash<word_type>, std::equal_to<word_type>,
				 std::allocator<std::pair<const word_type, tensor_type> > >::type embedding_type;
    typedef embedding_type label_matrix_type;
    
  public:
    Gradient() : hidden_(0), embedding_(0), count_(0), shared_(0) {}
    Gradient(const model_type& model)
      : count_(0), shared_(0) { initialize(model.hidden_, model.embedding_); }
    Gradient(const size_type& hidden,
	     const size_type& embedding)
      : count_(0), shared_(0){ initialize(hidden, embedding); }
    
    void initialize(const size_type& hidden,
		    const size_type& embedding);
    
    friend
    std::ostream& operator<<(std::ostream& os, const Gradient& x);
    
    friend
    std::istream& operator>>(std::istream& is, Gradient& x);
    
    Gradient& operator+=(const Gradient& x);
    Gradient& operator-=(const Gradient& x);

  public:
    void swap(Gradient& x)
    {
      std::swap(hidden_,    x.hidden_);
      std::swap(embedding_, x.embedding_);
      std::swap(count_,     x.count_);
      std::swap(shared_,    x.shared_);
      
      terminal_.swap(x.terminal_);
      
      Wc_.swap(x.Wc_);
      
      Wsh_.swap(x.Wsh_);
      Bsh_.swap(x.Bsh_);
      
      Wre_.swap(x.Wre_);
      Bre_.swap(x.Bre_);

      Wu_.swap(x.Wu_);
      Bu_.swap(x.Bu_);

      Wf_.swap(x.Wf_);
      Bf_.swap(x.Bf_);

      Wi_.swap(x.Wi_);
      Bi_.swap(x.Bi_);
      
      Ba_.swap(x.Ba_);
    }

    void clear()
    {
      count_ = 0;
      shared_ = 0;
      
      terminal_.clear();

      Wc_.clear();
      
      Wsh_.clear();
      Bsh_.clear();

      Wre_.clear();
      Bre_.clear();

      Wu_.clear();
      Bu_.clear();

      Wf_.setZero();
      Bf_.setZero();

      Wi_.setZero();
      Bi_.setZero();
      
      Ba_.setZero();
    }
    
    tensor_type& terminal(const word_type& word)
    {
      tensor_type& tensor = terminal_[word];
      if (! tensor.rows())
	tensor = tensor_type::Zero(embedding_, 1);
      return tensor;
    }
    
    tensor_type& Wc(const word_type& label)
    {
      tensor_type& tensor = Wc_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(1, hidden_);
      return tensor;
    }

    tensor_type& Wsh(const word_type& label)
    {
      tensor_type& tensor = Wsh_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, hidden_ + embedding_);
      return tensor;
    }

    tensor_type& Bsh(const word_type& label)
    {
      tensor_type& tensor = Bsh_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, 1);
      return tensor;
    }

    tensor_type& Wre(const word_type& label)
    {
      tensor_type& tensor = Wre_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, hidden_ + hidden_);
      return tensor;
    }

    tensor_type& Bre(const word_type& label)
    {
      tensor_type& tensor = Bre_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, 1);
      return tensor;
    }

    tensor_type& Wu(const word_type& label)
    {
      tensor_type& tensor = Wu_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, hidden_);
      return tensor;
    }

    tensor_type& Bu(const word_type& label)
    {
      tensor_type& tensor = Bu_[label];
      if (! tensor.rows())
	tensor = tensor_type::Zero(hidden_, 1);
      return tensor;
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
    
    // source/target embedding
    embedding_type terminal_;

    // classification
    label_matrix_type Wc_;
    
    // shift
    label_matrix_type Wsh_;
    label_matrix_type Bsh_;
    
    // reduce
    label_matrix_type Wre_;
    label_matrix_type Bre_;
    
    // unary
    label_matrix_type Wu_;
    label_matrix_type Bu_;
    
    // final
    tensor_type Wf_;
    tensor_type Bf_;

    // idle
    tensor_type Wi_;
    tensor_type Bi_;
    
    // axiom
    tensor_type Ba_;
  };
};

namespace std
{
  inline
  void swap(rnnp::Gradient& x, rnnp::Gradient& y)
  {
    x.swap(y);
  }
};

#endif
