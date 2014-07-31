// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRADIENT__MODEL5__HPP__
#define __RNNP__GRADIENT__MODEL5__HPP__ 1

#include <rnnp/gradient.hpp>

namespace rnnp
{
  namespace gradient
  {
    class Model5 : public Gradient
    {
    public:
    
    public:
      Model5() {}
      Model5(const model_type& model) { initialize(model.hidden_, model.embedding_); }
      Model5(const size_type& hidden,
	     const size_type& embedding) { initialize(hidden, embedding); }
    
      void initialize(const size_type& hidden,
		      const size_type& embedding);
    
      friend
      std::ostream& operator<<(std::ostream& os, const Model5& x);
    
      friend
      std::istream& operator>>(std::istream& is, Model5& x);
    
      Model5& operator+=(const Model5& x);
      Model5& operator-=(const Model5& x);

    public:
      void swap(Model5& x)
      {
	Gradient::swap(static_cast<Gradient&>(x));
	
	terminal_.swap(x.terminal_);
      
	Wc_.swap(x.Wc_);
	Bc_.swap(x.Bc_);
	Wfe_.swap(x.Wfe_);
      
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
      
	Wqu_.swap(x.Wqu_);
	Bqu_.swap(x.Bqu_);
	Bqe_.swap(x.Bqe_);

	Ba_.swap(x.Ba_);
      }

      void clear()
      {
	Gradient::clear();

	terminal_.clear();

	Wc_.clear();
	Bc_.clear();
	Wfe_.clear();
      
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

	Wqu_.setZero();
	Bqu_.setZero();
	Bqe_.setZero();
      
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
	  tensor = tensor_type::Zero(1, hidden_ * 3);
	return tensor;
      }

      tensor_type& Bc(const word_type& label)
      {
	tensor_type& tensor = Bc_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(1, 3);
	return tensor;
      }

      tensor_type& Wsh(const word_type& label)
      {
	tensor_type& tensor = Wsh_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + embedding_ + hidden_);
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
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_ + hidden_);
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
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_);
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
      // embedding
      matrix_embedding_type terminal_;
    
      // classification
      matrix_category_type Wc_;
      matrix_category_type Bc_;

      // features
      weights_type Wfe_;
    
      // shift
      matrix_category_type Wsh_;
      matrix_category_type Bsh_;
    
      // reduce
      matrix_category_type Wre_;
      matrix_category_type Bre_;
      
      // category
      matrix_category_type Wu_;
      matrix_category_type Bu_;
    
      // final
      tensor_type Wf_;
      tensor_type Bf_;

      // idle
      tensor_type Wi_;
      tensor_type Bi_;

      // queue
      tensor_type Wqu_;
      tensor_type Bqu_;
      tensor_type Bqe_;
    
      // axiom
      tensor_type Ba_;
    };
  };
};

namespace std
{
  inline
  void swap(rnnp::gradient::Model5& x, rnnp::gradient::Model5& y)
  {
    x.swap(y);
  }
};

#endif
