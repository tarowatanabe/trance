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
      
	if (option.learn_embedding())
	  update(theta, theta.terminal_, G.terminal_, gradient.terminal_, scale, false);
	
	if (option.learn_classification())
	  update(theta, theta.Wc_, G.Wc_, gradient.Wc_, scale, lambda_ != 0.0);
	
	if (option.learn_hidden()) {
	  update(theta, theta.Wsh_, G.Wsh_, gradient.Wsh_, scale, lambda_ != 0.0);
	  update(theta, theta.Bsh_, G.Bsh_, gradient.Bsh_, scale, false);
	
	  update(theta, theta.Wre_, G.Wre_, gradient.Wre_, scale, lambda_ != 0.0);
	  update(theta, theta.Bre_, G.Bre_, gradient.Bre_, scale, false);
	
	  update(theta, theta.Wu_, G.Wu_, gradient.Wu_, scale, lambda_ != 0.0);
	  update(theta, theta.Bu_, G.Bu_, gradient.Bu_, scale, false);

	  update(theta, theta.Wf_, G.Wf_, gradient.Wf_, scale, lambda_ != 0.0);
	  update(theta, theta.Bf_, G.Bf_, gradient.Bf_, scale, false);

	  update(theta, theta.Wi_, G.Wi_, gradient.Wi_, scale, lambda_ != 0.0);
	  update(theta, theta.Bi_, G.Bi_, gradient.Bi_, scale, false);
	
	  update(theta, theta.Ba_, G.Ba_, gradient.Ba_, scale, false);
	}
      }

      static inline
      double learning_rate(const double& eta0, const double& g)
      {
	return std::min(1.0, eta0 / std::sqrt(1.0 + g));
      }
      
      struct update_visitor_regularize
      {
	update_visitor_regularize(tensor_type& theta,
				  tensor_type& G,
				  const tensor_type&  g,
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
	
	  G_(i, j) = G_(i, j) * 0.95 + (g_(i, j) * scale_) * (g_(i, j) * scale_);
	
	  const double rate = learning_rate(eta0_, G_(i, j));
	  const double x1 = theta_(i, j) - rate * scale_ * g_(i, j);
	
	  theta_(i, j) = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	}
      
	tensor_type&       theta_;
	tensor_type&       G_;
	const tensor_type& g_;
      
	const double scale_;
	const double lambda_;
	const double eta0_;
      };

      struct update_visitor
      {
	update_visitor(tensor_type& theta,
		       tensor_type& G,
		       const tensor_type&  g,
		       const double& scale,
		       const double& eta0)
	  : theta_(theta), G_(G), g_(g), scale_(scale), eta0_(eta0) {}
      
	void init(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  operator()(value, i, j);
	}
      
	void operator()(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  if (g_(i, j) == 0) return;
	
	  G_(i, j) = G_(i, j) * 0.95 + (g_(i, j) * scale_) * (g_(i, j) * scale_);
	  
	  theta_(i, j) -= learning_rate(eta0_, G_(i, j)) * scale_ * g_(i, j);
	}
      
	tensor_type&       theta_;
	tensor_type&       G_;
	const tensor_type& g_;
      
	const double scale_;
	const double eta0_;
      };
      
      void update(model_type& model,
		  tensor_type& theta,
		  tensor_type& G,
		  const gradient_type::embedding_type& embedding,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::embedding_type embedding_type;
	
	if (lambda_ != 0.0) {
	  embedding_type::const_iterator eiter_end = embedding.end();
	  for (embedding_type::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	    const tensor_type& g = eiter->second;
	  
	    for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	      if (g(row, 0) != 0.0) {
		G(row, col) = G(row, col) * 0.95 + (g(row, 0) * scale) * (g(row, 0) * scale);
		
		const double rate = learning_rate(eta0_, G(row, col));
		const double x1 = theta(row, col) - rate * scale * g(row, 0);
	      
		theta(row, col) = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
	      }
	  }
	} else {
	  embedding_type::const_iterator eiter_end = embedding.end();
	  for (embedding_type::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	    const tensor_type& g = eiter->second;
	  
	    for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	      if (g(row, 0) != 0.0) {
		G(row, col) = G(row, col) * 0.95 + (g(row, 0) * scale) * (g(row, 0) * scale);
		
		theta(row, col) -= learning_rate(eta0_, G(row, col)) * scale * g(row, 0);
	      }
	  }
	}
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  tensor_type& G,
		  const gradient_type::matrix_unary_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::matrix_unary_type matrix_unary_type;
	
	if (regularize) {
	  matrix_unary_type::const_iterator giter_end = grad.end();
	  for (matrix_unary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;
	    
	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + (g(row, col) * scale) * (g(row, col) * scale);
		  
		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = learning_rate(eta0_, G.block(offset, 0, rows, cols)(row, col));
		  const double x1 = x - rate * scale * g(row, col);
		  
		  x = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
		}
	  }
	} else {
	  matrix_unary_type::const_iterator giter_end = grad.end();
	  for (matrix_unary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;
	    
	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + (g(row, col) * scale) * (g(row, col) * scale);
		  
		  const double rate = learning_rate(eta0_, G.block(offset, 0, rows, cols)(row, col));
		  
		  theta.block(offset, 0, rows, cols)(row, col) -= rate * scale * g(row, col);
		}
	  }
	}
      }
      
      void update(model_type& model,
		  tensor_type& theta,
		  tensor_type& G,
		  const gradient_type::matrix_binary_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	typedef gradient_type::matrix_binary_type matrix_binary_type;
	
	if (regularize) {
	  matrix_binary_type::const_iterator giter_end = grad.end();
	  for (matrix_binary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = model.offset_binary(giter->first.first, giter->first.second);
	    
	    const tensor_type& g = giter->second;
	    
	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + (g(row, col) * scale) * (g(row, col) * scale);
		  
		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = learning_rate(eta0_, G.block(offset, 0, rows, cols)(row, col));
		  const double x1 = x - rate * scale * g(row, col);
		  
		  x = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - rate * lambda_);
		}
	  }
	} else {
	  matrix_binary_type::const_iterator giter_end = grad.end();
	  for (matrix_binary_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = model.offset_binary(giter->first.first, giter->first.second);
	    
	    const tensor_type& g = giter->second;
	    
	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) =
		    G.block(offset, 0, rows, cols)(row, col) * 0.95 + (g(row, col) * scale) * (g(row, col) * scale);
		  
		  const double rate = learning_rate(eta0_, G.block(offset, 0, rows, cols)(row, col));
		  
		  theta.block(offset, 0, rows, cols)(row, col) -= rate * scale * g(row, col);
		}
	  }
	}
      }
    
      void update(model_type& model,
		  tensor_type& theta,
		  tensor_type& G,
		  const tensor_type& g,
		  const double scale,
		  const bool regularize) const
      {
	if (regularize) {
	  update_visitor_regularize visitor(theta, G, g, scale, lambda_, eta0_);
	
	  theta.visit(visitor);
	} else {
	  update_visitor visitor(theta, G, g, scale, eta0_);
	  
	  theta.visit(visitor);
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
