// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPTIMIZE_ADADEC__HPP__
#define __RNNP__OPTIMIZE_ADADEC__HPP__ 1

// AN EMPIRICAL STUDY OF LEARNING RATES IN DEEP NEURAL NETWORKS FOR SPEECH RECOGNITION
//
// ICASSP 2013
//

#include <rnnp/optimize.hpp>

#include <utils/mathop.hpp>

namespace rnnp
{
  namespace optimize
  {
    struct AdaDec : public Optimize
    {
      AdaDec(const model_type& theta,
	     const double& lambda,
	     const double& eta0)
	: G_(theta), lambda_(lambda), eta0_(eta0) { G_.clear(); }
    
      void operator()(model_type& theta,
		      const gradient_type& gradient,
		      const option_type& option) const
      {
	if (! gradient.count_) return;

	const double scale = 1.0 / gradient.count_;
      
	model_type& G = const_cast<model_type&>(G_);
      
	if (option.learn_embedding()) {
	  update_embedding(theta.source_, G.source_, gradient.source_, scale);
	  update_embedding(theta.target_, G.target_, gradient.target_, scale);
	  
	  update_embedding(theta.head_source_, G.head_source_, gradient.head_source_, scale);
	  update_embedding(theta.head_target_, G.head_target_, gradient.head_target_, scale);
	}

	if (option.learn_classification()) {
	  update_weights(theta.Wf_, G.Wf_, gradient.Wf_, scale);
	  
	  update(theta.Wc_, G.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	}
	
	if (option.learn_hidden()) {
	  update(theta.Wsh_, G.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta.Bsh_, G.Bsh_, gradient.Bsh_, scale, false);
	
	  update(theta.Wrs_, G.Wrs_, gradient.Wrs_, scale, lambda_ != 0.0);
	  update(theta.Brs_, G.Brs_, gradient.Brs_, scale, false);
	
	  update(theta.Wri_, G.Wri_, gradient.Wri_, scale, lambda_ != 0.0);
	  update(theta.Bri_, G.Bri_, gradient.Bri_, scale, false);
	
	  update(theta.Bi_, G.Bi_, gradient.Bi_, scale, false);
	}
      }

      template <typename Theta, typename GVar, typename Grad>
      struct update_visitor_regularize
      {
	update_visitor_regularize(Eigen::MatrixBase<Theta>& theta,
				  Eigen::MatrixBase<GVar>& G,
				  const Eigen::MatrixBase<Grad>& g,
				  const double& scale,
				  const double& lambda,
				  const double& eta0)
	  : theta_(theta), G_(G), g_(g), scale_(scale), lambda_(lambda), eta0_(eta0) {}
      
	void init(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  operator()(value, i, j);
	}
      
	void operator()(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  if (g_(i, j) == 0) return;
	
	  G_(i, j) = G_(i, j) * 0.95 + g_(i, j) * g_(i, j) * scale_ * scale_;
	
	  const double rate = eta0_ / std::sqrt(double(1.0) + G_(i, j));
	  const double x1 = theta_(i, j) - rate * scale_ * g_(i, j);
	
	  theta_(i, j) = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	}
      
	Eigen::MatrixBase<Theta>&      theta_;
	Eigen::MatrixBase<GVar>&    G_;
	const Eigen::MatrixBase<Grad>& g_;
      
	const double scale_;
	const double lambda_;
	const double eta0_;
      };
    
      struct learning_rate
      {
	learning_rate(const double& eta0) : eta0_(eta0) {}
      
	template <typename Tp>
	Tp operator()(const Tp& x) const
	{
	  return (x == 0.0 ? 0.0 : eta0_ / std::sqrt(double(1.0) + x));
	}
      
	const double& eta0_;
      };

      template <typename Vector, typename GVector, typename Grads>
      void update_weights(Vector& theta,
			  GVector& Gs,
			  const Grads& grads,
			  const double scale) const
      {
	typename Grads::const_iterator fiter_end = grads.end();
	for (typename Grads::const_iterator fiter = grads.begin(); fiter != fiter_end; ++ fiter) 
	  if (fiter->second != 0) {
	    typename Vector::value_type& x = theta[fiter->first];
	    typename GVector::value_type& G = Gs[fiter->first];
	    const typename Grads::mapped_type& g = fiter->second;
	  
	    G = G * 0.95 + g * g * scale * scale;
	  
	    const double rate = eta0_ / std::sqrt(double(1.0) + G);
	    const double x1 = x - rate * scale * g;
	  
	    x = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	  }
      }

      template <typename Theta, typename GVar, typename Embedding>
      void update_embedding(Eigen::MatrixBase<Theta>& theta,
			    Eigen::MatrixBase<GVar>& G,
			    const Embedding& embedding,
			    const double scale) const
      {
	if (lambda_ != 0.0) {
	  typename Embedding::const_iterator eiter_end = embedding.end();
	  for (typename Embedding::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	    const tensor_type& g = eiter->second;
	  
	    for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	      if (g(row, 0) != 0.0) {
		G(row, col) = G(row, col) * 0.95 + g(row, 0) * g(row, 0) * scale * scale;
	      
		const double rate = eta0_ / std::sqrt(double(1.0) + G(row, col));
		const double x1 = theta(row, col) - rate * scale * g(row, 0);
	      
		theta(row, col) = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	      }
	  }
	} else {
	  typename Embedding::const_iterator eiter_end = embedding.end();
	  for (typename Embedding::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	  
	    G.col(col) = G.col(col).array() * 0.95 + eiter->second.array().square() * scale * scale;
	    theta.col(col).array() -= scale * eiter->second.array() * G.col(col).array().unaryExpr(learning_rate(eta0_));
	  }
	}
      }
    
      template <typename Theta, typename GVar, typename Grad>
      void update(Eigen::MatrixBase<Theta>& theta,
		  Eigen::MatrixBase<GVar>& G,
		  const Eigen::MatrixBase<Grad>& g,
		  const double scale,
		  const bool regularize=true) const
      {
	if (regularize) {
	  update_visitor_regularize<Theta, GVar, Grad> visitor(theta, G, g, scale, lambda_, eta0_);
	
	  theta.visit(visitor);
	} else {
	  G = G.array() * 0.95 + g.array().square() * scale * scale;
	  theta.array() -= scale * g.array() * G.array().unaryExpr(learning_rate(eta0_));
	}
      }
    
    private:
      model_type G_;
    
      double lambda_;
      double eta0_;
    };
  };
};

#endif
