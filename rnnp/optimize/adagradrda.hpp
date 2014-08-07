// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPTIMIZE_ADAGRADRDA__HPP__
#define __RNNP__OPTIMIZE_ADAGRADRDA__HPP__ 1

//
// ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
//

#include <rnnp/optimize.hpp>
#include <rnnp/model_traits.hpp>

#include <utils/mathop.hpp>

namespace rnnp
{
  namespace optimize
  {

    template <typename Theta>
    struct AdaGradRDA : public Optimize
    {
      typedef typename model_traits<Theta>::model_type    model_impl_type;
      typedef typename model_traits<Theta>::gradient_type gradient_impl_type;
      
      AdaGradRDA(const Theta& theta,
		 const double& lambda,
		 const double& eta0,
		 const double& epsilon)
	: X_(theta), G_(theta), A_(theta), lambda_(lambda), eta0_(eta0), epsilon_(epsilon), t_(0) { G_.clear(); A_.clear(); }
      
      double decay()
      {
	eta0_ *= 0.5;
	return eta0_;
      }
      
      void operator()(model_impl_type& theta,
		      const gradient_impl_type& gradient,
		      const option_type& option) const;
      
      static inline
      double learning_rate(const double& eta0, const double& epsilon, const double& g)
      {
	return eta0 / std::sqrt(epsilon + g);
      }
      
      struct update_visitor_regularize
      {
	update_visitor_regularize(tensor_type& theta,
				  const tensor_type& X,
				  tensor_type& G,
				  tensor_type& A,
				  const tensor_type& g,
				  const double& scale,
				  const double& lambda,
				  const double& eta0,
				  const double& epsilon,
				  const size_type& t)
	  : theta_(theta), X_(X), G_(G), A_(A), g_(g), scale_(scale), lambda_(lambda), eta0_(eta0), epsilon_(epsilon), t_(t) {}
      
	void init(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  operator()(value, i, j);
	}
      
	void operator()(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  if (g_(i, j) == 0) return;
	  
	  G_(i, j) += (g_(i, j) * scale_) * (g_(i, j) * scale_);
	  A_(i, j) += g_(i, j) * scale_;
	  
	  const double x = X_(i, j) - learning_rate(eta0_, epsilon_, G_(i, j)) * A_(i, j);
	  
	  theta_(i, j) = utils::mathop::sgn(x) * std::max(0.0, std::fabs(x) - t_ * lambda_);
	}
      
	tensor_type&       theta_;
	const tensor_type& X_;
	tensor_type&       G_;
	tensor_type&       A_;
	const tensor_type& g_;
      
	const double scale_;
	const double lambda_;
	const double eta0_;
	const double epsilon_;
	const size_type t_;
      };

      struct update_visitor
      {
	update_visitor(tensor_type& theta,
		       const tensor_type& X,
		       tensor_type& G,
		       tensor_type& A,
		       const tensor_type& g,
		       const double& scale,
		       const double& eta0,
		       const double& epsilon,
		       const size_type& t)
	  : theta_(theta), X_(X), G_(G), A_(A), g_(g), scale_(scale), eta0_(eta0), epsilon_(epsilon), t_(t) {}
      
	void init(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  operator()(value, i, j);
	}
      
	void operator()(const tensor_type::Scalar& value, tensor_type::Index i, tensor_type::Index j)
	{
	  if (g_(i, j) == 0) return;
	
	  G_(i, j) += (g_(i, j) * scale_) * (g_(i, j) * scale_);
	  A_(i, j) += g_(i, j) * scale_;
	  
	  theta_(i, j) = X_(i, j) - learning_rate(eta0_, epsilon_, G_(i, j)) * A_(i, j);
	}
      
	tensor_type&       theta_;
	const tensor_type& X_;
	tensor_type&       G_;
	tensor_type&       A_;
	const tensor_type& g_;
      
	const double scale_;
	const double eta0_;
	const double epsilon_;
	const size_type t_;
      };

      void update(tensor_type& theta,
		  const tensor_type& X,
		  tensor_type& G,
		  tensor_type& A,
		  const matrix_embedding_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	if (lambda_ != 0.0) {
	  matrix_embedding_type::const_iterator eiter_end = grad.end();
	  for (matrix_embedding_type::const_iterator eiter = grad.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	    const tensor_type& g = eiter->second;
	    
	    for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	      if (g(row, 0) != 0.0) {
		G(row, col) += (g(row, 0) * scale) * (g(row, 0) * scale);
		A(row, col) += g(row, 0) * scale;
		
		const double x = X(row, col) - learning_rate(eta0_, epsilon_, G(row, col)) * A(row, col);
		
		theta(row, col) = utils::mathop::sgn(x) * std::max(0.0, std::fabs(x) - t_ * lambda_);
	      }
	  }
	} else {
	  matrix_embedding_type::const_iterator eiter_end = grad.end();
	  for (matrix_embedding_type::const_iterator eiter = grad.begin(); eiter != eiter_end; ++ eiter) {
	    const size_type col = eiter->first.id();
	    const tensor_type& g = eiter->second;
	    
	    for (tensor_type::Index row = 0; row != eiter->second.rows(); ++ row) 
	      if (g(row, 0) != 0.0) {
		G(row, col) += (g(row, 0) * scale) * (g(row, 0) * scale);
		A(row, col) += g(row, 0) * scale;
		
		theta(row, col) = X(row, col) - learning_rate(eta0_, epsilon_, G(row, col)) * A(row, col);
	      }
	  }
	}
      }

      
      void update(model_type::weights_type& theta,
		  const model_type::weights_type& Xs,
		  model_type::weights_type& Gs,
		  model_type::weights_type& As,
		  const gradient_type::weights_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	if (lambda_ != 0.0) {
	  gradient_type::weights_type::const_iterator fiter_end = grad.end();
	  for (gradient_type::weights_type::const_iterator fiter = grad.begin(); fiter != fiter_end; ++ fiter) 
	    if (fiter->second != 0) {
	      model_type::weights_type::value_type& x = theta[fiter->first];
	      const model_type::weights_type::value_type& X = Xs[fiter->first];
	      model_type::weights_type::value_type& G = Gs[fiter->first];
	      model_type::weights_type::value_type& A = As[fiter->first];
	      const gradient_type::weights_type::mapped_type& g = fiter->second;
	      
	      G += (g * scale) * (g * scale);
	      A += g * scale;
	      
	      const double x1 = X - learning_rate(eta0_, epsilon_, G) * A;
	      
	      x = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - t_ * lambda_);
	    }
	} else {
	  gradient_type::weights_type::const_iterator fiter_end = grad.end();
	  for (gradient_type::weights_type::const_iterator fiter = grad.begin(); fiter != fiter_end; ++ fiter) 
	    if (fiter->second != 0) {
	      model_type::weights_type::value_type& x = theta[fiter->first];
	      const model_type::weights_type::value_type& X = Xs[fiter->first];
	      model_type::weights_type::value_type& G = Gs[fiter->first];
	      model_type::weights_type::value_type& A = As[fiter->first];
	      const gradient_type::weights_type::mapped_type& g = fiter->second;
	      
	      G += (g * scale) * (g * scale);
	      A += g * scale;
	      
	      x = X - learning_rate(eta0_, epsilon_, G) * A;
	    }
	}
      }
      
      void update(tensor_type& theta,
		  const tensor_type& X,
		  tensor_type& G,
		  tensor_type& A,
		  const matrix_category_type& grad,
		  const double scale,
		  const bool regularize) const
      {
	if (regularize && lambda_ != 0.0) {
	  matrix_category_type::const_iterator giter_end = grad.end();
	  for (matrix_category_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;

	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) += (g(row, col) * scale) * (g(row, col) * scale);
		  A.block(offset, 0, rows, cols)(row, col) += g(row, col) * scale;
		  
		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = learning_rate(eta0_, epsilon_, G.block(offset, 0, rows, cols)(row, col));
		  const double x1 = X.block(offset, 0, rows, cols)(row, col) - rate * A.block(offset, 0, rows, cols)(row, col);
		  
		  x = utils::mathop::sgn(x1) * std::max(0.0, std::fabs(x1) - t_ * lambda_);
		}
	  }
	} else {
	  matrix_category_type::const_iterator giter_end = grad.end();
	  for (matrix_category_type::const_iterator giter = grad.begin(); giter != giter_end; ++ giter) {
	    const size_type rows = giter->second.rows();
	    const size_type cols = giter->second.cols();
	    const size_type offset = rows * giter->first.non_terminal_id();
	    
	    const tensor_type& g = giter->second;

	    for (tensor_type::Index col = 0; col != g.cols(); ++ col) 
	      for (tensor_type::Index row = 0; row != g.rows(); ++ row) 
		if (g(row, col) != 0) {
		  G.block(offset, 0, rows, cols)(row, col) += (g(row, col) * scale) * (g(row, col) * scale);
		  A.block(offset, 0, rows, cols)(row, col) += g(row, col) * scale;
		  
		  tensor_type::Scalar& x = theta.block(offset, 0, rows, cols)(row, col);
		  
		  const double rate = learning_rate(eta0_, epsilon_, G.block(offset, 0, rows, cols)(row, col));
		  
		  x = X.block(offset, 0, rows, cols)(row, col) - rate * A.block(offset, 0, rows, cols)(row, col);
		}
	  }
	}
      }
      
      void update(tensor_type& theta,
		  const tensor_type& X,
		  tensor_type& G,
		  tensor_type& A,
		  const tensor_type& g,
		  const double scale,
		  const bool regularize) const
      {
	if (regularize && lambda_ != 0.0) {
	  update_visitor_regularize visitor(theta, X, G, A, g, scale, lambda_, eta0_, epsilon_, t_);
	  
	  theta.visit(visitor);
	} else {
	  update_visitor visitor(theta, X, G, A, g, scale, eta0_, epsilon_, t_);
	  
	  theta.visit(visitor);
	}
      }
    
    private:
      Theta X_;
      Theta G_;
      Theta A_;
    
      double lambda_;
      double eta0_;
      double epsilon_;
      
      size_type t_;
    };

    template <>
    inline
    void AdaGradRDA<model::Model1>::operator()(model::Model1& theta,
					       const gradient::Model1& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;

      ++ const_cast<size_type&>(t_);

      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
	
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
      
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model2>::operator()(model::Model2& theta,
					       const gradient::Model2& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;
      
      ++ const_cast<size_type&>(t_);

      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model3>::operator()(model::Model3& theta,
					       const gradient::Model3& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;

      ++ const_cast<size_type&>(t_);
      
      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Wqu_, X_.Wqu_, G.Wqu_, A.Wqu_, gradient.Wqu_, scale, true);
	update(theta.Bqu_, X_.Bqu_, G.Bqu_, A.Bqu_, gradient.Bqu_, scale, false);
	update(theta.Bqe_, X_.Bqe_, G.Bqe_, A.Bqe_, gradient.Bqe_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model4>::operator()(model::Model4& theta,
					       const gradient::Model4& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;
      
      ++ const_cast<size_type&>(t_);
      
      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model5>::operator()(model::Model5& theta,
					       const gradient::Model5& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;
      
      ++ const_cast<size_type&>(t_);
      
      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Wqu_, X_.Wqu_, G.Wqu_, A.Wqu_, gradient.Wqu_, scale, true);
	update(theta.Bqu_, X_.Bqu_, G.Bqu_, A.Bqu_, gradient.Bqu_, scale, false);
	update(theta.Bqe_, X_.Bqe_, G.Bqe_, A.Bqe_, gradient.Bqe_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model6>::operator()(model::Model6& theta,
					       const gradient::Model6& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;
      
      ++ const_cast<size_type&>(t_);

      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Psh_, X_.Psh_, G.Psh_, A.Psh_, gradient.Psh_, scale, true);
	update(theta.Qsh_, X_.Qsh_, G.Qsh_, A.Qsh_, gradient.Qsh_, scale, true);
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Pre_, X_.Pre_, G.Pre_, A.Pre_, gradient.Pre_, scale, true);
	update(theta.Qre_, X_.Qre_, G.Qre_, A.Qre_, gradient.Qre_, scale, true);
	update(theta.Wre_, X_.Qre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Pu_, X_.Pu_, G.Pu_, A.Pu_, gradient.Pu_, scale, true);
	update(theta.Qu_, X_.Qu_, G.Qu_, A.Qu_, gradient.Qu_, scale, true);
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);
	
	update(theta.Wqu_, X_.Wqu_, G.Wqu_, A.Wqu_, gradient.Wqu_, scale, true);
	update(theta.Bqu_, X_.Bqu_, G.Bqu_, A.Bqu_, gradient.Bqu_, scale, false);
	update(theta.Bqe_, X_.Bqe_, G.Bqe_, A.Bqe_, gradient.Bqe_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }

    template <>
    inline
    void AdaGradRDA<model::Model7>::operator()(model::Model7& theta,
					       const gradient::Model7& gradient,
					       const option_type& option) const
    {
      if (! gradient.count_) return;
      
      ++ const_cast<size_type&>(t_);
      
      const double scale = 1.0 / gradient.count_;
      
      model_impl_type& G = const_cast<model_impl_type&>(G_);
      model_impl_type& A = const_cast<model_impl_type&>(A_);
      
      if (option.learn_embedding())
	update(theta.terminal_, X_.terminal_, G.terminal_, A.terminal_, gradient.terminal_, scale, false);
      
      if (option.learn_classification()) {
	update(theta.Wc_,  X_.Wc_,  G.Wc_,  A.Wc_,  gradient.Wc_, scale, true);
	update(theta.Bc_,  X_.Bc_,  G.Bc_,  A.Bc_,  gradient.Bc_, scale, false);
	update(theta.Wfe_, X_.Wfe_, G.Wfe_, A.Wfe_, gradient.Wfe_, scale, true);
      }
	
      if (option.learn_hidden()) {
	update(theta.Wsh_, X_.Wsh_, G.Wsh_, A.Wsh_, gradient.Wsh_, scale, true);
	update(theta.Bsh_, X_.Bsh_, G.Bsh_, A.Bsh_, gradient.Bsh_, scale, false);
	
	update(theta.Wre_, X_.Wre_, G.Wre_, A.Wre_, gradient.Wre_, scale, true);
	update(theta.Bre_, X_.Bre_, G.Bre_, A.Bre_, gradient.Bre_, scale, false);
	
	update(theta.Wu_, X_.Wu_, G.Wu_, A.Wu_, gradient.Wu_, scale, true);
	update(theta.Bu_, X_.Bu_, G.Bu_, A.Bu_, gradient.Bu_, scale, false);

	update(theta.Wf_, X_.Wf_, G.Wf_, A.Wf_, gradient.Wf_, scale, true);
	update(theta.Bf_, X_.Bf_, G.Bf_, A.Bf_, gradient.Bf_, scale, false);
	  
	update(theta.Wi_, X_.Wi_, G.Wi_, A.Wi_, gradient.Wi_, scale, true);
	update(theta.Bi_, X_.Bi_, G.Bi_, A.Bi_, gradient.Bi_, scale, false);

	update(theta.Wbu_, X_.Wbu_, G.Wbu_, A.Wbu_, gradient.Wbu_, scale, true);
	update(theta.Bbu_, X_.Bbu_, G.Bbu_, A.Bbu_, gradient.Bbu_, scale, false);
	update(theta.Bbs_, X_.Bbs_, G.Bbs_, A.Bbs_, gradient.Bbs_, scale, false);
	
	update(theta.Wqu_, X_.Wqu_, G.Wqu_, A.Wqu_, gradient.Wqu_, scale, true);
	update(theta.Bqu_, X_.Bqu_, G.Bqu_, A.Bqu_, gradient.Bqu_, scale, false);
	update(theta.Bqe_, X_.Bqe_, G.Bqe_, A.Bqe_, gradient.Bqe_, scale, false);
	
	update(theta.Ba_, X_.Ba_, G.Ba_, A.Ba_, gradient.Ba_, scale, false);
      }
    }
  };
};

#endif
