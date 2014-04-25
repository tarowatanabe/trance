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
	  update_embedding(theta.terminal_, gradient.terminal_, scale);
	
	if (option.learn_classification())
	  update(theta.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	
	if (option.learn_hidden()) {
	  update(theta.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta.Bsh_, gradient.Bsh_, scale, false);
	  
	  update(theta.Wre_, gradient.Wre_, scale, lambda_ != 0.0);
	  update(theta.Bre_, gradient.Bre_, scale, false);
	  
	  update(theta.Wu_, gradient.Wu_, scale, lambda_ != 0.0);
	  update(theta.Bu_, gradient.Bu_, scale, false);
	  
	  update(theta.Wf_, gradient.Wf_, scale, lambda_ != 0.0);
	  update(theta.Bf_, gradient.Bf_, scale, false);
	  
	  update(theta.Wi_, gradient.Wi_, scale, lambda_ != 0.0);
	  update(theta.Bi_, gradient.Bi_, scale, false);
	  
	  update(theta.Ba_, gradient.Ba_, scale, false);
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
      
      template <typename Theta, typename Grad>
      void update(Eigen::MatrixBase<Theta>& theta,
		  const Grad& grad,
		  const double scale,
		  const bool regularize=true) const
      {
	if (regularize)
	  theta *= 1.0 - eta0_ * lambda_;
	
	typename Grad::const_iterator giter_end = grad.end();
	for (typename Grad::const_iterator giter = grad.begin(); giter != giter_end; ++ giter)
	  theta.block(giter->second.rows() * giter->first.non_terminal_id(), 0, giter->second.rows(), giter->second.cols())
	    -= (eta0_ * scale) * giter->second;
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
