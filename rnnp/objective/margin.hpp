// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN__HPP__
#define __RNNP__OBJECTIVE__MARGIN__HPP__ 1

#include <rnnp/objective.hpp>
#include <rnnp/model_traits.hpp>

namespace rnnp
{
  namespace objective
  {
    struct Margin : public Objective
    {
    public:
      Margin() {}
      virtual ~Margin() {}
      
    public:
      template <typename Theta, typename Grad>
      loss_type operator()(const Theta& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   Grad& g)
      {
	initialize(candidates, oracles);
	
	const double loss = margin(candidates, oracles, option);

	if (! backward_.empty())
	  propagate(theta, candidates, oracles, option, g);
	
	return loss;
      }
      
      template <typename Grad>
      loss_type operator()(Grad& g) { return loss_type(); }
      
      virtual double margin(const parser_type& candidates,
			    const parser_oracle_type& oracles,
			    const option_type& option) = 0;
      
    private:
      template <typename Theta, typename Grad>
      void propagate(const Theta& theta,
		     const parser_type& candidates,
		     const parser_oracle_type& oracles,
		     const option_type& option,
		     Grad& g);
    };

    template <>
    inline
    void Margin::propagate(const model::Model1& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model1& g)
    {
      ++ g.count_;
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	  
	  backward_type& backward = backward_state(theta, state);
	  
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;
	  
	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	  
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wsh(state.label()) += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    g.Bsh(state.label()) += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_);
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, 0, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	      
	    g.Wu(state.label())
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bu(state.label())
	      += backward.delta_;
	      
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, 0, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }
    }

    template <>
    inline
    void Margin::propagate(const model::Model2& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model2& g)
    {
      ++ g.count_;
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	  
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;
	  
	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());
	    
	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Bsh += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	      
	    g.Wu(state.label())
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bu(state.label())
	      += backward.delta_;
	      
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, 0, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }
    }

    template <>
    inline
    void Margin::propagate(const model::Model3& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model3& g)
    {
      ++ g.count_;

      queue_.resize(candidates.queue_.rows(), candidates.queue_.cols());
      queue_.setZero();
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	  
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;

	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.embedding_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());
	    
	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bsh += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	      
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bu += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }

      const sentence_type& input = oracles.oracle_.sentence_;
      
      // finally, propagate queue contexts...
      for (size_type i = 0; i != input.size(); ++ i) {
	const size_type offset1 = 0;
	const size_type offset2 = theta.hidden_;
	
	const word_type::id_type word_id = theta.terminal(input[i]);
	
	g.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_)
	  += queue_.col(i) * candidates.queue_.col(i + 1).transpose();
	g.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_)
	  += queue_.col(i) * theta.terminal_.col(word_id).transpose();
	g.Bqu_ += queue_.col(i);
	
	queue_.col(i + 1).array()
	  += (candidates.queue_.col(i + 1).array().unaryExpr(model_type::dactivation())
	      * (theta.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_).transpose()
		 * queue_.col(i)).array());
	
	g.terminal(word_id) += (theta.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_).transpose()
				* queue_.col(i));
      }
      
      g.Bqe_ += queue_.col(input.size());
    }

    template <>
    inline
    void Margin::propagate(const model::Model4& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model4& g)
    {
      ++ g.count_;
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	    
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;

	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Bsh += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();	    
	    Bu += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }
    }

    template <>
    inline
    void Margin::propagate(const model::Model5& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model5& g)
    {
      ++ g.count_;

      queue_.resize(candidates.queue_.rows(), candidates.queue_.cols());
      queue_.setZero();
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	    
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;

	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.embedding_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bsh += backward.delta_;
	      
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    const size_type offset4 = theta.hidden_ + theta.hidden_ + theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	      
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();	    
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bu += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }

      const sentence_type& input = oracles.oracle_.sentence_;
      
      // finally, propagate queue contexts...
      for (size_type i = 0; i != input.size(); ++ i) {
	const size_type offset1 = 0;
	const size_type offset2 = theta.hidden_;
	
	const word_type::id_type word_id = theta.terminal(input[i]);
	
	g.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_)
	  += queue_.col(i) * candidates.queue_.col(i + 1).transpose();
	g.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_)
	  += queue_.col(i) * theta.terminal_.col(word_id).transpose();
	g.Bqu_ += queue_.col(i);
	
	queue_.col(i + 1).array()
	  += (candidates.queue_.col(i + 1).array().unaryExpr(model_type::dactivation())
	      * (theta.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_).transpose()
		 * queue_.col(i)).array());
	
	g.terminal(word_id) += (theta.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_).transpose()
				* queue_.col(i));
      }
      
      g.Bqe_ += queue_.col(input.size());
    }

    template <>
    inline
    void Margin::propagate(const model::Model6& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model6& g)
    {
      const size_type reduced = utils::bithack::max(theta.hidden_ >> 3, size_type(8));

      tensor_type L(theta.hidden_, reduced);
      tensor_type R(theta.hidden_, reduced);
      
      ++ g.count_;

      queue_.resize(candidates.queue_.rows(), candidates.queue_.cols());
      queue_.setZero();
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	    
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;

	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.embedding_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      const size_type full = theta.hidden_ + theta.embedding_ + theta.hidden_;
	      
	      L.block(row, 0, 1, reduced) = ((state.layer(theta.hidden_).transpose()
					      * theta.Psh_.block(full * row + offset1, 0, theta.hidden_, reduced))
					     + (theta.terminal_.col(head_id).transpose()
						* theta.Psh_.block(full * row + offset2, 0, theta.embedding_, reduced))
					     + (candidates.queue_.col(state.next()).transpose()
						* theta.Psh_.block(full * row + offset3, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qsh_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.layer(theta.hidden_))
					     + (theta.Qsh_.block(reduced * row, offset2, reduced, theta.embedding_)
						* theta.terminal_.col(head_id))
					     + (theta.Qsh_.block(reduced * row, offset3, reduced, theta.hidden_)
						* candidates.queue_.col(state.next()))).transpose();
	      
	      g.Psh_.block(full * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Psh_.block(full * row + offset2, 0, theta.embedding_, reduced).noalias()
		+= theta.terminal_.block(0, head_id, theta.embedding_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Psh_.block(full * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.derivation().next(), theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
	      g.Qsh_.block(reduced * row, offset1, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.derivation().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qsh_.block(reduced * row, offset2, reduced, theta.embedding_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * theta.terminal_.col(head_id).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qsh_.block(reduced * row, offset3, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * candidates.queue_.col(state.derivation().next()).transpose()
		    * backward.delta_.row(row)(0,0));
	    }

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_).noalias()
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bsh += backward.delta_;
	    
	    // propagate to ancedent
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    
	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      const size_type full = theta.hidden_ + theta.embedding_ + theta.hidden_;
	      
	      delta_derivation.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qsh_.block(reduced * row, offset1, reduced, theta.hidden_)).transpose()
			+ (theta.Psh_.block(full * row + offset1, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      
	      g.terminal(head_id).noalias()
		+= (((L.block(row, 0, 1, reduced)
		      * theta.Qsh_.block(reduced * row, offset2, reduced, theta.embedding_)).transpose()
		     + (theta.Psh_.block(full * row + offset2, 0, theta.embedding_, reduced)
			* R.block(row, 0, 1, reduced).transpose()))
		    * backward.delta_.row(row)(0, 0));
	    
	      queue_.col(state.derivation().next()).array()
		+= (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qsh_.block(reduced * row, offset3, reduced, theta.hidden_)).transpose()
			+ (theta.Psh_.block(full * row + offset3, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	    }
	    
	    delta_derivation.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id).noalias()
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    const size_type offset4 = theta.hidden_ + theta.hidden_ + theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());

	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      L.block(row, 0, 1, reduced) = ((state.derivation().layer(theta.hidden_).transpose()
					      * theta.Pre_.block(theta.hidden_ * 4 * row + offset1, 0, theta.hidden_, reduced))
					     + (state.reduced().layer(theta.hidden_).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset2, 0, theta.hidden_, reduced))
					     + (state.stack().layer(theta.hidden_).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset3, 0, theta.hidden_, reduced))
					     + (candidates.queue_.col(state.derivation().next()).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset4, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qre_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.derivation().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset2, reduced, theta.hidden_)
						* state.reduced().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset3, reduced, theta.hidden_)
						* state.stack().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset4, reduced, theta.hidden_)
						* candidates.queue_.col(state.derivation().next()))).transpose();
	      
	      g.Pre_.block(theta.hidden_ * 4 * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset2, 0, theta.hidden_, reduced).noalias()
		+= state.reduced().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= state.stack().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset4, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.derivation().next(), theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
	      g.Qre_.block(reduced * row, offset1, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.derivation().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qre_.block(reduced * row, offset2, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.reduced().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qre_.block(reduced * row, offset3, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.stack().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qre_.block(reduced * row, offset4, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * candidates.queue_.col(state.derivation().next()).transpose()
		    * backward.delta_.row(row)(0,0));
	    }
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    tensor_type& delta_reduced    = backward_state(theta, state.reduced()).delta_;
	    tensor_type& delta_stack      = backward_state(theta, state.stack()).delta_;
	    
	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      delta_derivation.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qre_.block(reduced * row, offset1, reduced, theta.hidden_)).transpose()
			+ (theta.Pre_.block(theta.hidden_ * 4 * row + offset1, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      
	      delta_reduced.array()
		+= (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qre_.block(reduced * row, offset2, reduced, theta.hidden_)).transpose()
			+ (theta.Pre_.block(theta.hidden_ * 4 * row + offset2, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      
	      delta_stack.array()
		+= (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qre_.block(reduced * row, offset3, reduced, theta.hidden_)).transpose()
			+ (theta.Pre_.block(theta.hidden_ * 4 * row + offset3, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      
	      queue_.col(state.derivation().next()).array()
		+= (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qre_.block(reduced * row, offset4, reduced, theta.hidden_)).transpose()
			+ (theta.Pre_.block(theta.hidden_ * 4 * row + offset4, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	    }
	    
	    delta_derivation.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    delta_reduced.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    delta_stack.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      L.block(row, 0, 1, reduced) = ((state.derivation().layer(theta.hidden_).transpose()
					      * theta.Pu_.block(theta.hidden_ * 3 * row + offset1, 0, theta.hidden_, reduced))
					     + (state.stack().layer(theta.hidden_).transpose()
						* theta.Pu_.block(theta.hidden_ * 3 * row + offset2, 0, theta.hidden_, reduced))
					     + (candidates.queue_.col(state.derivation().next()).transpose()
						* theta.Pu_.block(theta.hidden_ * 3 * row + offset3, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qu_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.derivation().layer(theta.hidden_))
					     + (theta.Qu_.block(reduced * row, offset2, reduced, theta.hidden_)
						* state.stack().layer(theta.hidden_))
					     + (theta.Qu_.block(reduced * row, offset3, reduced, theta.hidden_)
						* candidates.queue_.col(state.derivation().next()))).transpose();
	      
	      g.Pu_.block(theta.hidden_ * 3 * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pu_.block(theta.hidden_ * 3 * row + offset2, 0, theta.hidden_, reduced).noalias()
		+= state.stack().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pu_.block(theta.hidden_ * 3 * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.derivation().next(), theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
	      g.Qu_.block(reduced * row, offset1, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.derivation().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qu_.block(reduced * row, offset2, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * state.stack().layer(theta.hidden_).transpose()
		    * backward.delta_.row(row)(0,0));
	      g.Qu_.block(reduced * row, offset3, reduced, theta.hidden_).noalias()
		+= (L.block(row, 0, 1, reduced).transpose()
		    * candidates.queue_.col(state.derivation().next()).transpose()
		    * backward.delta_.row(row)(0,0));
	    }
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bu += backward.delta_;
	    
	    // propagate to ancedent
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    tensor_type& delta_stack      = backward_state(theta, state.stack()).delta_;
	    
	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      delta_derivation.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qu_.block(reduced * row, offset1, reduced, theta.hidden_)).transpose()
			+ (theta.Pu_.block(theta.hidden_ * 3 * row + offset1, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      	      
	      delta_stack.array()
		+= (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qu_.block(reduced * row, offset2, reduced, theta.hidden_)).transpose()
			+ (theta.Pu_.block(theta.hidden_ * 3 * row + offset2, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	      
	      queue_.col(state.derivation().next()).array()
		+= (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		    * (((L.block(row, 0, 1, reduced)
			 * theta.Qu_.block(reduced * row, offset3, reduced, theta.hidden_)).transpose()
			+ (theta.Pu_.block(theta.hidden_ * 3 * row + offset3, 0, theta.hidden_, reduced)
			   * R.block(row, 0, 1, reduced).transpose()))
		       * backward.delta_.row(row)(0, 0)).array());
	    }
	    
	    delta_derivation.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    delta_stack.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_.noalias() += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_.noalias() += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_.noalias() += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_.noalias() += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }

      const sentence_type& input = oracles.oracle_.sentence_;
      
      // finally, propagate queue contexts...
      for (size_type i = 0; i != input.size(); ++ i) {
	const size_type offset1 = 0;
	const size_type offset2 = theta.hidden_;
	
	const word_type::id_type word_id = theta.terminal(input[i]);
	
	g.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_)
	  += queue_.col(i) * candidates.queue_.col(i + 1).transpose();
	g.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_)
	  += queue_.col(i) * theta.terminal_.col(word_id).transpose();
	g.Bqu_ += queue_.col(i);
	
	queue_.col(i + 1).array()
	  += (candidates.queue_.col(i + 1).array().unaryExpr(model_type::dactivation())
	      * (theta.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_).transpose()
		 * queue_.col(i)).array());
	
	g.terminal(word_id) += (theta.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_).transpose()
				* queue_.col(i));
      }
      
      g.Bqe_ += queue_.col(input.size());
    }
    
    template <>
    inline
    void Margin::propagate(const model::Model7& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model7& g)
    {
      ++ g.count_;

      queue_.resize(candidates.queue_.rows(), candidates.queue_.cols());
      queue_.setZero();
      
      for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	state_set_type::iterator siter_end = states_[step].end();
	for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	  const state_type& state = *siter;
	    
	  backward_type& backward = backward_state(theta, state);
	    
	  //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;

	  // feature set
	  if (option.learn_classification()) {
	    const feature_vector_type& feats = *state.feature_vector();
	    
	    feature_vector_type::const_iterator fiter_end = feats.end();
	    for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
	      g.Wfe_[fiter->first] += backward.loss_ * fiter->second;
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.embedding_;
	
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bsh += backward.delta_;
	      
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    const size_type offset4 = theta.hidden_ + theta.hidden_ + theta.hidden_;
	      
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_category       = theta.offset_category(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();	    
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bre += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.reduced()).delta_.array()
	      += (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_classification = theta.offset_classification(state.label());
	    const size_type offset_closure        = utils::bithack::min(size_type(2), state.derivation().operation().closure()) * theta.hidden_;
	    const size_type offset_category       = theta.offset_category(state.label()) * 3 + offset_closure;
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());

	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    Wu.block(offset_closure, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(offset_closure, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(offset_closure, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.derivation().next()).transpose();
	    Bu.block(offset_closure, 0, theta.hidden_, 1)
	      += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    backward_state(theta, state.stack()).delta_.array()
	      += (state.stack().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.derivation().next()).array()
	      += (candidates.queue_.col(state.derivation().next()).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	      
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wf_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    Wc.block(0, 0, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Wc.block(0, theta.hidden_, 1, 1).array()   += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wi_.transpose() * backward.delta_).array());
	      
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  default:
	    throw std::runtime_error("invlaid operator");
	  }
	}
      }

      const sentence_type& input = oracles.oracle_.sentence_;
      
      // finally, propagate queue contexts...
      for (size_type i = 0; i != input.size(); ++ i) {
	const size_type offset1 = 0;
	const size_type offset2 = theta.hidden_;
	
	const word_type::id_type word_id = theta.terminal(input[i]);
	
	g.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_)
	  += queue_.col(i) * candidates.queue_.col(i + 1).transpose();
	g.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_)
	  += queue_.col(i) * theta.terminal_.col(word_id).transpose();
	g.Bqu_ += queue_.col(i);
	
	queue_.col(i + 1).array()
	  += (candidates.queue_.col(i + 1).array().unaryExpr(model_type::dactivation())
	      * (theta.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_).transpose()
		 * queue_.col(i)).array());
	
	g.terminal(word_id) += (theta.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_).transpose()
				* queue_.col(i));
      }
      
      g.Bqe_ += queue_.col(input.size());
    }
  };
};

#endif
