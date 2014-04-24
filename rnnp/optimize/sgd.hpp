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
	
	if (option.learn_embedding()) {
	  update_embedding(theta.source_, gradient.source_, scale);
	  update_embedding(theta.target_, gradient.target_, scale);
	  
	  update_embedding(theta.head_source_, gradient.head_source_, scale);
	  update_embedding(theta.head_target_, gradient.head_target_, scale);
	}
	
	if (option.learn_classification()) {
	  update_weights(theta.Wf_, gradient.Wf_, scale);
	  
	  update(theta.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	}
	  
	if (option.learn_hidden()) {
	  update(theta.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta.Bsh_, gradient.Bsh_, scale, false);
	
	  update(theta.Wrs_, gradient.Wrs_, scale, lambda_ != 0.0);
	  update(theta.Brs_, gradient.Brs_, scale, false);
	
	  update(theta.Wri_, gradient.Wri_, scale, lambda_ != 0.0);
	  update(theta.Bri_, gradient.Bri_, scale, false);
	
	  update(theta.Bi_, gradient.Bi_, scale, false);
	}
      }


      template <typename Theta, typename Embedding>
      void update_embedding(Eigen::MatrixBase<Theta>& theta,
			    const Embedding& embedding,
			    const double scale) const
      {
	typename Embedding::const_iterator eiter_end = embedding.end();
	for (typename Embedding::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter)
	  theta.col(eiter->first.id()) -= (eta0_ * scale) * eiter->second;
      }
    
      template <typename Vector, typename Grads>
      void update_weights(Vector& theta,
			  const Grads& grads,
			  const double scale) const
      {
	typename Grads::const_iterator fiter_end = grads.end();
	for (typename Grads::const_iterator fiter = grads.begin(); fiter != fiter_end; ++ fiter) 
	  if (fiter->second != 0)
	    theta[fiter->first] -= eta0_ * scale * fiter->second;
      }

    
      template <typename Theta, typename Grad>
      void update(Eigen::MatrixBase<Theta>& theta,
		  const Eigen::MatrixBase<Grad>& g,
		  const double scale,
		  const bool regularize=true) const
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
