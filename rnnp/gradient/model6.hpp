// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRADIENT__MODEL6__HPP__
#define __RNNP__GRADIENT__MODEL6__HPP__ 1

#include <rnnp/gradient.hpp>

namespace rnnp
{
  namespace gradient
  {
    class Model6 : public Gradient
    {
    public:
    
    public:
      Model6() {}
      Model6(const model_type& model) { initialize(model.hidden_, model.embedding_); }
      Model6(const size_type& hidden,
	     const size_type& embedding) { initialize(hidden, embedding); }
    
      void initialize(const size_type& hidden,
		      const size_type& embedding);
    
      friend
      std::ostream& operator<<(std::ostream& os, const Model6& x);
    
      friend
      std::istream& operator>>(std::istream& is, Model6& x);
    
      Model6& operator+=(const Model6& x);
      Model6& operator-=(const Model6& x);

    public:
      void swap(Model6& x)
      {
	Gradient::swap(static_cast<Gradient&>(x));
	
	terminal_.swap(x.terminal_);
	category_.swap(x.category_);
      
	Wc_.swap(x.Wc_);
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
	category_.clear();
	
	Wc_.clear();
	Wfe_.clear();
      
	Wsh_.setZero();
	Bsh_.setZero();

	Wre_.setZero();
	Bre_.setZero();

	Wu_.setZero();
	Bu_.setZero();

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

      tensor_type& category(const word_type& label)
      {
	tensor_type& tensor = category_[label];
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
    
    public:
      // embedding
      matrix_embedding_type terminal_;
      matrix_category_type  category_;
    
      // classification
      matrix_category_type Wc_;

      // features
      weights_type Wfe_;
    
      // shift
      tensor_type Wsh_;
      tensor_type Bsh_;
    
      // reduce
      tensor_type Wre_;
      tensor_type Bre_;
      
      // category
      tensor_type Wu_;
      tensor_type Bu_;
    
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
  void swap(rnnp::gradient::Model6& x, rnnp::gradient::Model6& y)
  {
    x.swap(y);
  }
};

#endif
