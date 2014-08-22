//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include "gradient/model8.hpp"

namespace rnnp
{
  namespace gradient
  {
    
    void Model8::initialize(const size_type& hidden,
			    const size_type& embedding)
    {
      Gradient::initialize(hidden, embedding);

      const size_type reduced = utils::bithack::max(hidden_ >> 3, size_type(8));
    
      terminal_.clear();
    
      // initialize matrix    
      Wc_.clear();
      Bc_.clear();
      Wfe_.clear();
    
      Psh_ = tensor_type::Zero(hidden_ * (hidden_ + embedding_ + hidden_), reduced);
      Qsh_ = tensor_type::Zero(hidden_ * reduced,                          hidden_ + embedding_ + hidden_);
      Wsh_.clear();
      Bsh_.clear();
    
      Pre_ = tensor_type::Zero(hidden_ * hidden_ * 4, reduced);
      Qre_ = tensor_type::Zero(hidden_ * reduced,     hidden_ * 4);
      Wre_.clear();
      Bre_.clear();

      Pu_  = tensor_type::Zero(hidden_ * hidden_ * 3, reduced);
      Qu_  = tensor_type::Zero(hidden_ * reduced,     hidden_ * 3);
      Wu_.clear();
      Bu_.clear();
      
      Wf_ = tensor_type::Zero(hidden_, hidden_);
      Bf_ = tensor_type::Zero(hidden_, 1);
    
      Wi_ = tensor_type::Zero(hidden_, hidden_);
      Bi_ = tensor_type::Zero(hidden_, 1);

      Wqu_ = tensor_type::Zero(hidden_, hidden_ + embedding_);
      Bqu_ = tensor_type::Zero(hidden_, 1);
      Bqe_ = tensor_type::Zero(hidden_, 1);
      
      Ba_ = tensor_type::Zero(hidden_, 1);
    }

#define GRADIENT_STREAM_OPERATOR(Theta, Op, Stream)	\
    Theta.Op(Stream, Theta.terminal_);			\
							\
    Theta.Op(Stream, Theta.Wc_);			\
    Theta.Op(Stream, Theta.Bc_);			\
    Theta.Op(Stream, Theta.Wfe_);			\
							\
    Theta.Op(Stream, Theta.Psh_);			\
    Theta.Op(Stream, Theta.Qsh_);			\
    Theta.Op(Stream, Theta.Wsh_);			\
    Theta.Op(Stream, Theta.Bsh_);			\
							\
    Theta.Op(Stream, Theta.Pre_);			\
    Theta.Op(Stream, Theta.Qre_);			\
    Theta.Op(Stream, Theta.Wre_);			\
    Theta.Op(Stream, Theta.Bre_);			\
							\
    Theta.Op(Stream, Theta.Pu_);			\
    Theta.Op(Stream, Theta.Qu_);			\
    Theta.Op(Stream, Theta.Wu_);			\
    Theta.Op(Stream, Theta.Bu_);			\
							\
    Theta.Op(Stream, Theta.Wf_);			\
    Theta.Op(Stream, Theta.Bf_);			\
							\
    Theta.Op(Stream, Theta.Wi_);			\
    Theta.Op(Stream, Theta.Bi_);			\
							\
    Theta.Op(Stream, Theta.Wqu_);			\
    Theta.Op(Stream, Theta.Bqu_);			\
    Theta.Op(Stream, Theta.Bqe_);			\
							\
    Theta.Op(Stream, Theta.Bi_);

    std::ostream& operator<<(std::ostream& os, const Model8& theta)
    {
      os.write((char*) &theta.hidden_,    sizeof(theta.hidden_));
      os.write((char*) &theta.embedding_, sizeof(theta.embedding_));
      os.write((char*) &theta.count_,     sizeof(theta.count_));

      GRADIENT_STREAM_OPERATOR(theta, write_matrix, os);
    
      return os;
    }
  
    std::istream& operator>>(std::istream& is, Model8& theta)
    {
      is.read((char*) &theta.hidden_,    sizeof(theta.hidden_));
      is.read((char*) &theta.embedding_, sizeof(theta.embedding_));
      is.read((char*) &theta.count_,     sizeof(theta.count_));

      GRADIENT_STREAM_OPERATOR(theta, read_matrix, is);
    
      return is;
    }

#undef GRADIENT_STREAM_OPERATOR

#define GRADIENT_BINARY_OPERATOR(Op)	\
    Op(terminal_, x.terminal_);			\
						\
    Op(Wc_,  x.Wc_);				\
    Op(Bc_,  x.Bc_);				\
    Op(Wfe_, x.Wfe_);				\
						\
    Op(Psh_, x.Psh_);				\
    Op(Qsh_, x.Qsh_);				\
    Op(Wsh_, x.Wsh_);				\
    Op(Bsh_, x.Bsh_);				\
						\
    Op(Pre_, x.Pre_);				\
    Op(Qre_, x.Qre_);				\
    Op(Wre_, x.Wre_);				\
    Op(Bre_, x.Bre_);				\
						\
    Op(Pu_, x.Pu_);				\
    Op(Qu_, x.Qu_);				\
    Op(Wu_, x.Wu_);				\
    Op(Bu_, x.Bu_);				\
						\
    Op(Wf_, x.Wf_);				\
    Op(Bf_, x.Bf_);				\
						\
    Op(Wi_, x.Wi_);				\
    Op(Bi_, x.Bi_);				\
						\
    Op(Wqu_, x.Wqu_);				\
    Op(Bqu_, x.Bqu_);				\
    Op(Bqe_, x.Bqe_);				\
						\
    Op(Ba_, x.Ba_);

  
    Model8& Model8::operator+=(const Model8& x)
    {
      GRADIENT_BINARY_OPERATOR(plus_equal);

      return *this;
    }
  
    Model8& Model8::operator-=(const Model8& x)
    {
      GRADIENT_BINARY_OPERATOR(minus_equal);

      return *this;
    }

#undef GRADIENT_BINARY_OPERATOR

  };
};
