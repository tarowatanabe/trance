// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRADIENT__MODEL1__HPP__
#define __RNNP__GRADIENT__MODEL1__HPP__ 1

#include <rnnp/gradient.hpp>

namespace rnnp
{
  namespace gradient
  {
    class Model1 : public Gradient
    {
    public:
    
    public:
      Model1() {}
      Model1(const model_type& model) { initialize(model.hidden_, model.embedding_); }
      Model1(const size_type& hidden,
	     const size_type& embedding) { initialize(hidden, embedding); }
    
      void initialize(const size_type& hidden,
		      const size_type& embedding);
    
      friend
      std::ostream& operator<<(std::ostream& os, const Model1& x);
    
      friend
      std::istream& operator>>(std::istream& is, Model1& x);
    
      Model1& operator+=(const Model1& x);
      Model1& operator-=(const Model1& x);

    public:
      void swap(Model1& x)
      {
	Gradient::swap(static_cast<Gradient&>(x));
	
	terminal_.swap(x.terminal_);
      
	Wc_.swap(x.Wc_);
      
	Wsh_.swap(x.Wsh_);
	Bsh_.swap(x.Bsh_);
      
	Wrel_.swap(x.Wrel_);
	Brel_.swap(x.Brel_);

	Wrer_.swap(x.Wrer_);
	Brer_.swap(x.Brer_);
	
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
	Gradient::clear();

	terminal_.clear();

	Wc_.clear();
      
	Wsh_.clear();
	Bsh_.clear();

	Wrel_.clear();
	Brel_.clear();

	Wrer_.clear();
	Brer_.clear();

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
	  tensor = tensor_type::Zero(hidden_, embedding_);
	return tensor;
      }
    
      tensor_type& Bsh(const word_type& label)
      {
	tensor_type& tensor = Bsh_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, 1);
	return tensor;
      }
    
      tensor_type& Wrel(const word_type& label)
      {
	tensor_type& tensor = Wrel_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_);
	return tensor;
      }
    
      tensor_type& Brel(const word_type& label)
      {
	tensor_type& tensor = Brel_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, 1);
	return tensor;
      }

      tensor_type& Wrer(const word_type& label)
      {
	tensor_type& tensor = Wrer_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_);
	return tensor;
      }
    
      tensor_type& Brer(const word_type& label)
      {
	tensor_type& tensor = Brer_[label];
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
      // source/target embedding
      matrix_embedding_type terminal_;
    
      // classification
      matrix_category_type Wc_;
    
      // shift
      matrix_category_type Wsh_;
      matrix_category_type Bsh_;
    
      // reduce left
      matrix_category_type Wrel_;
      matrix_category_type Brel_;

      // reduce right
      matrix_category_type Wrer_;
      matrix_category_type Brer_;
    
      // category
      matrix_category_type Wu_;
      matrix_category_type Bu_;
    
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
};

namespace std
{
  inline
  void swap(rnnp::gradient::Model1& x, rnnp::gradient::Model1& y)
  {
    x.swap(y);
  }
};

#endif
