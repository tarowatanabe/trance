// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__MODEL__MODEL6__HPP__
#define __RNNP__MODEL__MODEL6__HPP__ 1

#include <rnnp/model.hpp>

#include <boost/random/uniform_real_distribution.hpp>

namespace rnnp
{
  namespace model
  {
    class Model6 : public Model
    {
    public:
      
    public:
      Model6() : Model() {}
      Model6(const path_type& path) { read(path); }
      Model6(const size_type& hidden,
	     const size_type& embedding,
	     const grammar_type& grammar)
      { initialize(hidden, embedding, grammar); }
      
      void initialize(const size_type& hidden,
		      const size_type& embedding,
		      const grammar_type& grammar);
      
      // IO
      void write(const path_type& path) const;
      void read(const path_type& path);
      void embedding(const path_type& path);
    
      friend
      std::ostream& operator<<(std::ostream& os, const Model6& x);
      friend
      std::istream& operator>>(std::istream& is, Model6& x);
    
      Model6& operator+=(const Model6& x);
      Model6& operator-=(const Model6& x);
      Model6& operator*=(const double& x);
      Model6& operator/=(const double& x);
    
    private:
      template <typename Gen>
      struct __randomize
      {
	__randomize(Gen& gen, const double range=0.01) : gen_(gen), range_(range) {}
      
	template <typename Tp>
	Tp operator()(const Tp& x) const
	{
	  return boost::random::uniform_real_distribution<Tp>(-range_, range_)(const_cast<Gen&>(gen_));
	}
      
	Gen& gen_;
	double range_;
      };

    public:
      template <typename Gen>
      void random(Gen& gen)
      {
	const double range_embed = std::sqrt(6.0 / (embedding_ + 1));
	const double range_c  = std::sqrt(6.0 / (hidden_ + 1));
	const double range_sh = std::sqrt(6.0 / (hidden_ + hidden_ + embedding_));
	const double range_re = std::sqrt(6.0 / (hidden_ + hidden_ + hidden_ + embedding_));
	const double range_u  = std::sqrt(6.0 / (hidden_ + hidden_ + embedding_));
	const double range_f  = std::sqrt(6.0 / (hidden_ + hidden_));
	const double range_i  = std::sqrt(6.0 / (hidden_ + hidden_));
	
	terminal_ = terminal_.array().unaryExpr(__randomize<Gen>(gen, range_embed));
      
	Wc_ = Wc_.array().unaryExpr(__randomize<Gen>(gen, range_c));
      
	Wsh_  = Wsh_.array().unaryExpr(__randomize<Gen>(gen, range_sh));
	Wrel_ = Wrel_.array().unaryExpr(__randomize<Gen>(gen, range_re));
	Wrer_ = Wrer_.array().unaryExpr(__randomize<Gen>(gen, range_re));
      
	Wu_ = Wu_.array().unaryExpr(__randomize<Gen>(gen, range_u));
      
	Wf_ = Wf_.array().unaryExpr(__randomize<Gen>(gen, range_f));
	Wi_ = Wi_.array().unaryExpr(__randomize<Gen>(gen, range_i));
      }
    
    
      void swap(Model6& x)
      {
	Model::swap(static_cast<Model&>(x));
	
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
	Model::clear();

	terminal_.setZero();
      
	Wc_.setZero();
      
	Wsh_.setZero();
	Bsh_.setZero();

	Wrel_.setZero();
	Brel_.setZero();

	Wrer_.setZero();
	Brer_.setZero();
      
	Wu_.setZero();
	Bu_.setZero();
      
	Wf_.setZero();
	Bf_.setZero();
      
	Wi_.setZero();
	Bi_.setZero();

	Ba_.setZero();
      }
    
    public:
      double l1() const
      {
	double norm = 0.0;
      
	norm += Wc_.lpNorm<1>();
	
	norm += Wsh_.lpNorm<1>();
	norm += Wrel_.lpNorm<1>();
	norm += Wrer_.lpNorm<1>();
      
	norm += Wu_.lpNorm<1>();
	norm += Wf_.lpNorm<1>();
	norm += Wi_.lpNorm<1>();
            
	return norm;
      }
    
      double l2() const
      {
	double norm = 0.0;
      
	norm += Wc_.squaredNorm();
      
	norm += Wsh_.squaredNorm();
	norm += Wrel_.squaredNorm();
	norm += Wrer_.squaredNorm();
      
	norm += Wu_.squaredNorm();
	norm += Wf_.squaredNorm();
	norm += Wi_.squaredNorm();
      
	return std::sqrt(norm);
      }
      
    public:
      // terminal embedding
      tensor_type terminal_;
    
      // classification
      tensor_type Wc_;
    
      // shift
      tensor_type Wsh_;
      tensor_type Bsh_;
    
      // reduce left
      tensor_type Wrel_;
      tensor_type Brel_;

      // reduce right
      tensor_type Wrer_;
      tensor_type Brer_;
    
      // unary
      tensor_type Wu_;
      tensor_type Bu_;

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
  void swap(rnnp::model::Model6& x, rnnp::model::Model6& y)
  {
    x.swap(y);
  }
};

#endif
