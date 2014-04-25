// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPTIMIZE_ADADELTA__HPP__
#define __RNNP__OPTIMIZE_ADADELTA__HPP__ 1

//
// ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
//

#include <rnnp/optimize.hpp>

#include <utils/mathop.hpp>

namespace rnnp
{
  namespace optimize
  {
    struct AdaDelta : public Optimize
    {
      AdaDelta(const model_type& theta,
	       const double& lambda,
	       const double& eta0)
	: G_(theta), X_(theta), lambda_(lambda), eta0_(eta0) { G_.clear(); X_.clear(); }
    
      void operator()(model_type& theta,
		      const gradient_type& gradient,
		      const option_type& option) const
      {
	if (! gradient.count_) return;

	const double scale = 1.0 / gradient.count_;
      
	model_type& G = const_cast<model_type&>(G_);
	model_type& X = const_cast<model_type&>(X_);
      
	if (option.learn_embedding())
	  update_embedding(theta.terminal_, G.terminal_, X.terminal_, gradient.terminal_, scale);
	
	if (option.learn_classification())
	  update(theta.Wc_, G.Wc_, X.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	
	if (option.learn_hidden()) {
	  update(theta.Wsh_, G.Wsh_, X.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta.Bsh_, G.Bsh_, X.Bsh_, gradient.Bsh_, scale, false);
	
	  update(theta.Wre_, G.Wre_, X.Wre_, gradient.Wre_, scale, lambda_ != 0.0);
	  update(theta.Bre_, G.Bre_, X.Bre_, gradient.Bre_, scale, false);
	
	  update(theta.Wu_, G.Wu_, X.Wu_, gradient.Wu_, scale, lambda_ != 0.0);
	  update(theta.Bu_, G.Bu_, X.Bu_, gradient.Bu_, scale, false);

	  update(theta.Wf_, G.Wf_, X.Wf_, gradient.Wf_, scale, lambda_ != 0.0);
	  update(theta.Bf_, G.Bf_, X.Bf_, gradient.Bf_, scale, false);
	  
	  update(theta.Wi_, G.Wi_, X.Wi_, gradient.Wi_, scale, lambda_ != 0.0);
	  update(theta.Bi_, G.Bi_, X.Bi_, gradient.Bi_, scale, false);
	
	  update(theta.Ba_, G.Ba_, X.Ba_, gradient.Ba_, scale, false);
	}
      }

      template <typename Theta, typename GVar, typename XVar, typename Grad>
      struct update_visitor_regularize
      {
	update_visitor_regularize(Eigen::MatrixBase<Theta>& theta,
				  Eigen::MatrixBase<GVar>& G,
				  Eigen::MatrixBase<XVar>& X,
				  const Eigen::MatrixBase<Grad>& g,
				  const double& scale,
				  const double& lambda,
				  const double& eta0)
	  : theta_(theta), G_(G), X_(X), g_(g), scale_(scale), lambda_(lambda), eta0_(eta0) {}
      
	void init(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  operator()(value, i, j);
	}
      
	void operator()(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  if (g_(i, j) == 0) return;
	
	  G_(i, j) = G_(i, j) * 0.95 + g_(i, j) * g_(i, j) * scale_ * scale_;
	
	  const double rate = std::sqrt(eta0_ + X_(i, j)) / std::sqrt(eta0_ + G_(i, j));
	  const double x1 = theta_(i, j) - rate * scale_ * g_(i, j);
	  const double x2 = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	
	  X_(i, j) = X_(i, j) * 0.95 + (x2 - theta_(i, j)) * (x2 - theta_(i, j));
	
	  theta_(i, j) = x2;
	}
      
	Eigen::MatrixBase<Theta>&      theta_;
	Eigen::MatrixBase<GVar>&       G_;
	Eigen::MatrixBase<XVar>&       X_;
	const Eigen::MatrixBase<Grad>& g_;
      
	const double scale_;
	const double lambda_;
	const double eta0_;
      };

      template <typename Theta, typename GVar, typename XVar, typename Embedding>
      void update_embedding(Eigen::MatrixBase<Theta>& theta,
			    Eigen::MatrixBase<GVar>& G,
			    Eigen::MatrixBase<XVar>& X,
			    const Embedding& embedding,
			    const double scale) const
      {
	typename Embedding::const_iterator eiter_end = embedding.end();
	for (typename Embedding::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
	  const size_type col = eiter->first.id();
	  const tensor_type& g = eiter->second;
	
	  for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	    if (g(row, 0) != 0.0) {
	      G(row, col) = G(row, col) * 0.95 + g(row, 0) * g(row, 0) * scale * scale;
	    
	      const double rate = std::sqrt(eta0_ + X(row, col)) / std::sqrt(eta0_ + G(row, col));
	      const double x1 = theta(row, col) - rate * scale * g(row, 0);
	      const double x2 = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	    
	      X(row, col) = X(row, col) * 0.95 + (x2 - theta(row, col)) * (x2 - theta(row, col));
	    
	      theta(row, col) = x2;
	    }
	}
      }
      
      template <typename Theta, typename GVar, typename XVar, typename Grad>
      void update(Eigen::MatrixBase<Theta>& theta,
		  Eigen::MatrixBase<GVar>& G,
		  Eigen::MatrixBase<XVar>& X,
		  const Grad& grad,
		  const double scale,
		  const bool regularize=true) const
      {
	if (regularize) {
	  typename Grad::const_iterator giter_end = grad.end();
	  for (typename Grad::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = grad->second.rows();
	    const size_type cols = grad->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;

	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + g(row, col) * g(row, col) * scale * scale;

		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = (std::sqrt(eta0_ + X.block(offset, 0, rows, cols)(row, col))
				       / std::sqrt(eta0_ + G.block(offset, 0, rows, cols)(row, col)));
		  
		  const double x1 = x - rate * scale * g(row, col);
		  const double x2 = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
		  
		  X.block(offset, 0, rows, cols)(row, col)
		    = X.block(offset, 0, rows, cols)(row, col) * 0.95 + (x2 - x) * (x2 - x);
		  
		  x = x2;
		}
	  }
	} else {
	  typename Grad::const_iterator giter_end = grad.end();
	  for (typename Grad::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = grad->second.rows();
	    const size_type cols = grad->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;

	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + g(row, col) * g(row, col) * scale * scale;

		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = (std::sqrt(eta0_ + X.block(offset, 0, rows, cols)(row, col))
				       / std::sqrt(eta0_ + G.block(offset, 0, rows, cols)(row, col)));
		  const double x1 = x - rate * scale * g(row, col);
		  
		  X.block(offset, 0, rows, cols)(row, col)
		    = X.block(offset, 0, rows, cols)(row, col) * 0.95 + (x1 - x) * (x1 - x);
		  
		  x = x1;
		}
	  }
	}
      }
      
      template <typename Theta, typename GVar, typename XVar, typename Grad>
      void update(Eigen::MatrixBase<Theta>& theta,
		  Eigen::MatrixBase<GVar>& G,
		  Eigen::MatrixBase<XVar>& X,
		  const Eigen::MatrixBase<Grad>& g,
		  const double scale,
		  const bool regularize=true) const
      {
	update_visitor_regularize<Theta, GVar, XVar, Grad> visitor(theta, G, X, g, scale, regularize ? lambda_ : 0.0, eta0_);
      
	theta.visit(visitor);
      }
    
    private:
      model_type G_;
      model_type X_;
    
      double lambda_;
      double eta0_;
    };
  };
};

#endif
