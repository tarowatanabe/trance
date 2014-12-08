// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__GRADIENT__MODEL7__HPP__
#define __TRANCE__GRADIENT__MODEL7__HPP__ 1

#include <trance/gradient.hpp>

namespace trance
{
  namespace gradient
  {
    class Model7 : public Gradient
    {
    public:
    
    public:
      Model7() {}
      Model7(const model_type& model) { initialize(model.hidden_, model.embedding_); }
      Model7(const size_type& hidden,
	     const size_type& embedding) { initialize(hidden, embedding); }
    
      void initialize(const size_type& hidden,
		      const size_type& embedding);
    
      friend
      std::ostream& operator<<(std::ostream& os, const Model7& x);
    
      friend
      std::istream& operator>>(std::istream& is, Model7& x);
    
      Model7& operator+=(const Model7& x);
      Model7& operator-=(const Model7& x);

    public:
      void swap(Model7& x)
      {
	Gradient::swap(static_cast<Gradient&>(x));
	
	terminal_.swap(x.terminal_);
      
	Wc_.swap(x.Wc_);
	Bc_.swap(x.Bc_);
	Wfe_.swap(x.Wfe_);
      
	Wsh_.swap(x.Wsh_);
	Bsh_.swap(x.Bsh_);

	Wshr_.swap(x.Wshr_);
	Bshr_.swap(x.Bshr_);
	
	Wshz_.swap(x.Wshz_);
	Bshz_.swap(x.Bshz_);

	Wre_.swap(x.Wre_);
	Bre_.swap(x.Bre_);

	Wrer_.swap(x.Wrer_);
	Brer_.swap(x.Brer_);

	Wrez_.swap(x.Wrez_);
	Brez_.swap(x.Brez_);

	Wu_.swap(x.Wu_);
	Bu_.swap(x.Bu_);

	Wur_.swap(x.Wur_);
	Bur_.swap(x.Bur_);

	Wuz_.swap(x.Wuz_);
	Buz_.swap(x.Buz_);
	
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

	Wshr_.clear();
	Bshr_.clear();

	Wshz_.clear();
	Bshz_.clear();
	
	Wre_.clear();
	Bre_.clear();

	Wrer_.clear();
	Brer_.clear();

	Wrez_.clear();
	Brez_.clear();
	
	Wu_.clear();
	Bu_.clear();

	Wur_.clear();
	Bur_.clear();

	Wuz_.clear();
	Buz_.clear();
	
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

      tensor_type& Wshr(const word_type& label)
      {
	tensor_type& tensor = Wshr_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + embedding_ + hidden_);
	return tensor;
      }
    
      tensor_type& Bshr(const word_type& label)
      {
	tensor_type& tensor = Bshr_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, 1);
	return tensor;
      }
    
      tensor_type& Wshz(const word_type& label)
      {
	tensor_type& tensor = Wshz_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + embedding_ + hidden_);
	return tensor;
      }
    
      tensor_type& Bshz(const word_type& label)
      {
	tensor_type& tensor = Bshz_[label];
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

      tensor_type& Wrer(const word_type& label)
      {
	tensor_type& tensor = Wrer_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_ + hidden_);
	return tensor;
      }
    
      tensor_type& Brer(const word_type& label)
      {
	tensor_type& tensor = Brer_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, 1);
	return tensor;
      }

      tensor_type& Wrez(const word_type& label)
      {
	tensor_type& tensor = Wrez_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_ + hidden_);
	return tensor;
      }
    
      tensor_type& Brez(const word_type& label)
      {
	tensor_type& tensor = Brez_[label];
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

      tensor_type& Wur(const word_type& label)
      {
	tensor_type& tensor = Wur_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_);
	return tensor;
      }

      tensor_type& Bur(const word_type& label)
      {
	tensor_type& tensor = Bur_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, 1);
	return tensor;
      }

      tensor_type& Wuz(const word_type& label)
      {
	tensor_type& tensor = Wuz_[label];
	if (! tensor.rows())
	  tensor = tensor_type::Zero(hidden_, hidden_ + hidden_ + hidden_);
	return tensor;
      }

      tensor_type& Buz(const word_type& label)
      {
	tensor_type& tensor = Buz_[label];
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

      // shift reset
      matrix_category_type Wshr_;
      matrix_category_type Bshr_;

      // shift update
      matrix_category_type Wshz_;
      matrix_category_type Bshz_;
    
      // reduce
      matrix_category_type Wre_;
      matrix_category_type Bre_;

      // reduce reset
      matrix_category_type Wrer_;
      matrix_category_type Brer_;

      // reduce update
      matrix_category_type Wrez_;
      matrix_category_type Brez_;
      
      // unary
      matrix_category_type Wu_;
      matrix_category_type Bu_;

      // unary reset
      matrix_category_type Wur_;
      matrix_category_type Bur_;

      // unary update
      matrix_category_type Wuz_;
      matrix_category_type Buz_;
      
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
  void swap(trance::gradient::Model7& x, trance::gradient::Model7& y)
  {
    x.swap(y);
  }
};

#endif
