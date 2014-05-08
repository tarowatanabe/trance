// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__MODEL__HPP__
#define __RNNP__MODEL__HPP__ 1

#include <cmath>
#include <vector>

#include <rnnp/symbol.hpp>
#include <rnnp/grammar.hpp>
#include <rnnp/signature.hpp>

#include <utils/bithack.hpp>

#include <Eigen/Core>

#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace rnnp
{
  class Model
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef float     parameter_type;
    
    typedef Symbol symbol_type;
    typedef Symbol word_type;

    typedef Grammar   grammar_type;
    typedef Signature signature_type;

    typedef std::vector<word_type, std::allocator<word_type> > word_set_type;
    typedef std::vector<bool, std::allocator<bool> >           word_map_type;
    
    typedef boost::filesystem::path path_type;
    
    typedef Eigen::Matrix<parameter_type, Eigen::Dynamic, Eigen::Dynamic> tensor_type;
    typedef Eigen::Map<tensor_type>                                       matrix_type;
    
    struct activation
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	//return std::min(std::max(x, Tp(0)), Tp(63));
	return std::min(std::max(x, Tp(-1)), Tp(1));
      }
    };
    
    struct dactivation
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	//return Tp(0) < x && x < Tp(63);
	return Tp(- 1) < x && x < Tp(1);
      }
    };

  public:
    Model() : hidden_(0), embedding_(0) {}
    Model(const path_type& path) { read(path); }
    Model(const size_type& hidden,
	  const size_type& embedding,
	  const grammar_type& grammar) { initialize(hidden, embedding, grammar); }
    
    void initialize(const size_type& hidden,
		    const size_type& embedding,
		    const grammar_type& grammar);
    
    // IO
    void write(const path_type& path) const;
    void read(const path_type& path);
    void read_embedding(const path_type& path);
    
    friend
    std::ostream& operator<<(std::ostream& os, const Model& x);
    friend
    std::istream& operator>>(std::istream& is, Model& x);
    
    Model& operator+=(const Model& x);
    Model& operator-=(const Model& x);
    Model& operator*=(const double& x);
    Model& operator/=(const double& x);
    
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
      const double range_re = std::sqrt(6.0 / (hidden_ + hidden_ + hidden_));
      const double range_u  = std::sqrt(6.0 / (hidden_ + hidden_));
      const double range_f  = std::sqrt(6.0 / (hidden_ + hidden_));
      const double range_i  = std::sqrt(6.0 / (hidden_ + hidden_));
      
      terminal_ = terminal_.array().unaryExpr(__randomize<Gen>(gen, range_embed));
      
      Wc_ = Wc_.array().unaryExpr(__randomize<Gen>(gen, range_c));
      
      Wsh_ = Wsh_.array().unaryExpr(__randomize<Gen>(gen, range_sh));
      Wre_ = Wre_.array().unaryExpr(__randomize<Gen>(gen, range_re));
      
      Wu_ = Wu_.array().unaryExpr(__randomize<Gen>(gen, range_u));
      Wf_ = Wf_.array().unaryExpr(__randomize<Gen>(gen, range_f));
      Wi_ = Wi_.array().unaryExpr(__randomize<Gen>(gen, range_i));
    }
    
    void swap(Model& x)
    {
      std::swap(hidden_,    x.hidden_);
      std::swap(embedding_, x.embedding_);
      
      vocab_terminal_.swap(x.vocab_terminal_);
      vocab_non_terminal_.swap(x.vocab_non_terminal_);
      
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
      terminal_.setZero();
      
      Wc_.setZero();
      
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

      Ba_.setZero();
    }

  public:
    double l1() const
    {
      double norm = 0.0;
      
      norm += Wc_.lpNorm<1>();
      
      norm += Wsh_.lpNorm<1>();
      norm += Wre_.lpNorm<1>();
      
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
      norm += Wre_.squaredNorm();
      
      norm += Wu_.squaredNorm();
      norm += Wf_.squaredNorm();
      norm += Wi_.squaredNorm();
      
      return std::sqrt(norm);
    }
    
  public:
    word_type::id_type terminal(const word_type& x) const
    {
      return utils::bithack::branch(x.id() >= vocab_terminal_.size() || ! vocab_terminal_[x.id()],
				    symbol_type::UNK.id(),
				    x.id());
    }
    
    word_type::id_type terminal(const signature_type& signature, const word_type& x) const
    {
      if (x.id() < vocab_terminal_.size() && vocab_terminal_[x.id()])
	return x.id();
      else {
	const word_type sig = signature(x);
	
	return utils::bithack::branch(sig.id() >= vocab_terminal_.size() || ! vocab_terminal_[sig.id()],
				      symbol_type::UNK.id(),
				      sig.id());
      }
    }
    
    size_type offset_classification(const symbol_type& x) const
    {
      return x.non_terminal_id();
    }

    size_type offset_grammar(const symbol_type& x) const
    {
      return x.non_terminal_id() * hidden_;
    }
    
  public:
    // parameters
    size_type hidden_;
    size_type embedding_;
    
    // word set
    word_map_type vocab_terminal_;
    word_set_type vocab_non_terminal_;
    
    // terminal embedding
    tensor_type terminal_;
    
    // classification
    tensor_type Wc_;
    
    // shift
    tensor_type Wsh_;
    tensor_type Bsh_;
    
    // reduce
    tensor_type Wre_;
    tensor_type Bre_;
    
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

namespace std
{
  inline
  void swap(rnnp::Model& x, rnnp::Model& y)
  {
    x.swap(y);
  }
};

#endif
