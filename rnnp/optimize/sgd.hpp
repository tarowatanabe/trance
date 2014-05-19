// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPTIMIZE_SGD__HPP__
#define __RNNP__OPTIMIZE_SGD__HPP__ 1

#include <rnnp/optimize.hpp>

namespace rnnp
{
  namespace optimize
  {
    struct SGD : public Optimize
    {
      SGD(const model_type& theta,
	  const double& lambda,
	  const double& eta0)
	: lambda_(lambda), eta0_(eta0) { }
      
      void operator()(model_type& theta,
		      const gradient_type& gradient,
		      const option_type& option) const
      {
	if (! gradient.count_) return;
	
	const double scale = 1.0 / gradient.count_;
	
	if (option.learn_embedding())
	  update(theta, theta.terminal_, gradient.terminal_, scale, false);
	
	if (option.learn_classification())
	  update(theta, theta.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	
	if (option.learn_hidden()) {
	  update(theta, theta.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta, theta.Bsh_, gradient.Bsh_, scale, false);
	  
	  update(theta, theta.Wre_, gradient.Wre_, scale, lambda_ != 0.0);
	  update(theta, theta.Bre_, gradient.Bre_, scale, false);
	  
	  update(theta, theta.Wu_, gradient.Wu_, scale, lambda_ != 0.0);
	  update(theta, theta.Bu_, gradient.Bu_, scale, false);
	  
	  update(theta, theta.Wf_, gradient.Wf_, scale, lambda_ != 0.0);
	  update(theta, theta.Bf_, gradient.Bf_, scale, false);
	  
	  update(theta, theta.Wi_, gradient.Wi_, scale, lambda_ != 0.0);
	  update(theta, theta.Bi_, gradient.Bi_, scale, false);
	  
	  update(theta, theta.Ba_, gradient.Ba_, scale, false);
	}
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  const gradient_type::embedding_type& embedding,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::embedding_type embedding_type;

	embedding_type::const_iterator eiter_end = embedding.end();
	for (embedding_type::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter)
	  theta.col(eiter->first.id()) -= (eta0_ * scale) * eiter->second;
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  const gradient_type::matrix_unary_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::matrix_unary_type matrix_unary_type;
	
	if (regularize)
	  theta *= 1.0 - eta0_ * lambda_;
	
	matrix_unary_type::const_iterator giter_end = grad.end();
	for (matrix_unary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter)
	  theta.block(giter->second.rows() * giter->first.non_terminal_id(), 0, giter->second.rows(), giter->second.cols())
	    -= (eta0_ * scale) * giter->second;
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  const gradient_type::matrix_binary_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::matrix_binary_type matrix_binary_type;
	
	if (regularize)
	  theta *= 1.0 - eta0_ * lambda_;
	
	matrix_binary_type::const_iterator giter_end = grad.end();
	for (matrix_binary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	  const size_type offset = model.offset_binary(giter->first.first, giter->first.second);
	  
	  theta.block(offset, 0, giter->second.rows(), giter->second.cols())
	    -= (eta0_ * scale) * giter->second;
	}
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  const tensor_type& g,
		  const double scale,
		  const bool regularize) const
      {
	if (regularize)
	  theta *= 1.0 - eta0_ * lambda_;
      
	theta -= eta0_ * scale * g;
      }
    
    private:
      double lambda_;
      double eta0_;
    };
  };
};

#endif
