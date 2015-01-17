// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__OBJECTIVE__MARGIN__HPP__
#define __TRANCE__OBJECTIVE__MARGIN__HPP__ 1

#include <trance/objective.hpp>
#include <trance/model_traits.hpp>

namespace trance
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

	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	  }
	  
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
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
	      
	    const size_type offset_category = theta.offset_category(state.label());
	    
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
	    const size_type offset_category = theta.offset_category(state.label());
	    
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

	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
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
	      
	    const size_type offset_category = theta.offset_category(state.label());
	    
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
	    const size_type offset_category = theta.offset_category(state.label());
	    
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

	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());
	    
	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_ - 1).transpose();
	    Bsh += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.span().last_ - 1).array()
	      += (candidates.queue_.col(state.span().last_ - 1).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
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
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	      
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
	    Bu += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
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
	  
	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
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
	      
	    const size_type offset_category = theta.offset_category(state.label());
	    
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
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

	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_ - 1).transpose();
	    Bsh += backward.delta_;
	      
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.array()
	      += (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    g.terminal(head_id)
	      += (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose()
		  * backward.delta_);
	    
	    queue_.col(state.span().last_ - 1).array()
	      += (candidates.queue_.col(state.span().last_ - 1).array().unaryExpr(model_type::dactivation())
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();	    
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
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
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(0, offset3, theta.hidden_, theta.hidden_)
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
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
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
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
	  
	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
	  }
	    
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_ += backward.delta_;
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());
	    
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
	      
	    const size_type offset_category = theta.offset_category(state.label());
	    
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
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
    void Margin::propagate(const model::Model7& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model7& g)
    {
      ++ g.count_;

      queue_.resize(candidates.queue_.rows(), candidates.queue_.cols());
      queue_.setZero();

      tensor_type layer_update;
      tensor_type layer_reset;
      tensor_type layer_hidden;
      
      tensor_type delta_update;
      tensor_type delta_reset;
      tensor_type delta_hidden;
      
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

	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.noalias()
	      += theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose() * backward.loss_;
	  }
	  
	  switch (state.operation().operation()) {
	  case operation_type::AXIOM: {
	    // initial bias
	    g.Ba_.array() += state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation()) * backward.delta_.array();
	  } break;
	  case operation_type::SHIFT: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.embedding_;
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());

	    layer_update = (theta.Bshz_.block(offset_category, 0, theta.hidden_, 1)
			    + (theta.Wshz_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			       * state.derivation().layer(theta.hidden_))
			    + (theta.Wshz_.block(offset_category, offset2, theta.hidden_, theta.embedding_)
			       * theta.terminal_.col(head_id))
			    + (theta.Wshz_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_ - 1))
			    ).array().unaryExpr(model_type::sigmoid());
	    
	    layer_reset = (theta.Bshr_.block(offset_category, 0, theta.hidden_, 1)
			   + (theta.Wshr_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			      * state.derivation().layer(theta.hidden_))
			   + (theta.Wshr_.block(offset_category, offset2, theta.hidden_, theta.embedding_)
			      * theta.terminal_.col(head_id))
			   + (theta.Wshr_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			      * candidates.queue_.col(state.span().last_ - 1))
			   ).array().unaryExpr(model_type::sigmoid());
	    
	    layer_hidden = (theta.Bsh_.block(offset_category, 0, theta.hidden_, 1)
			    + (layer_reset.array() * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
						      * state.derivation().layer(theta.hidden_)).array()).matrix()
			    + (theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_)
			       * theta.terminal_.col(head_id))
			    + (theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_ - 1))
			    ).array().unaryExpr(model_type::activation());

	    delta_update
	      = (layer_update.array().unaryExpr(model_type::dsigmoid())
		 * (state.derivation().layer(theta.hidden_) - layer_hidden).array()
		 * backward.delta_.array());
	    
	    delta_hidden
	      = (layer_hidden.array().unaryExpr(model_type::dactivation())
		 * (1.0 - layer_update.array())
		 * backward.delta_.array());
	    
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    tensor_type& terminal = g.terminal(head_id);
	    
	    delta_derivation.array() += layer_update.array() * backward.delta_.array();

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());
	    
	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_)
	      += (delta_hidden.array() * layer_reset.array()).matrix() * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_)
	      += delta_hidden * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_hidden * candidates.queue_.col(state.span().last_ - 1).transpose();
	    Bsh += delta_hidden;
	    
	    // propagate to reset...
	    delta_reset
	      = (layer_reset.array().unaryExpr(model_type::dsigmoid())
		 * (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
		    * state.derivation().layer(theta.hidden_)).array()
		 * delta_hidden.array());
	    
	    // propagate to antecedent
	    delta_derivation.noalias()
	      += (theta.Wsh_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose()
		  * (delta_hidden.array() * layer_reset.array()).matrix());
	    
	    terminal.noalias()
	      += theta.Wsh_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose() * delta_hidden;
	    
	    queue_.col(state.span().last_ - 1).noalias()
	      += theta.Wsh_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_hidden;

	    // reset...
	    tensor_type& Wshr = g.Wshr(state.label());
	    tensor_type& Bshr = g.Bshr(state.label());
	    
	    Wshr.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_reset * state.derivation().layer(theta.hidden_).transpose();
	    Wshr.block(0, offset2, theta.hidden_, theta.embedding_)
	      += delta_reset * theta.terminal_.col(head_id).transpose();
	    Wshr.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_reset * candidates.queue_.col(state.span().last_ - 1).transpose();
	    Bshr += delta_reset;
	    
	    delta_derivation.noalias()
	      += theta.Wshr_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    terminal.noalias()
	      += theta.Wshr_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose() * delta_reset;
	    
	    queue_.col(state.span().last_ - 1).noalias()
	      += theta.Wshr_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_reset;

	    // update...
	    tensor_type& Wshz = g.Wshz(state.label());
	    tensor_type& Bshz = g.Bshz(state.label());
	    
	    Wshz.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_update * state.derivation().layer(theta.hidden_).transpose();
	    Wshz.block(0, offset2, theta.hidden_, theta.embedding_)
	      += delta_update * theta.terminal_.col(head_id).transpose();
	    Wshz.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_update * candidates.queue_.col(state.span().last_ - 1).transpose();
	    Bshz += delta_update;
	    
	    delta_derivation.noalias()
	      += theta.Wshz_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    terminal.noalias()
	      += theta.Wshz_.block(offset_category, offset2, theta.hidden_, theta.embedding_).transpose() * delta_update;
	    
	    queue_.col(state.span().last_ - 1).noalias()
	      += theta.Wshz_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::REDUCE: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    const size_type offset4 = theta.hidden_ + theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());

	    layer_update = (theta.Brez_.block(offset_category, 0, theta.hidden_, 1)
			    + (theta.Wrez_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			       * state.derivation().layer(theta.hidden_))
			    + (theta.Wrez_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
			       * state.reduced().layer(theta.hidden_))
			    + (theta.Wrez_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			       * state.stack().layer(theta.hidden_))
			    + (theta.Wrez_.block(offset_category, offset4, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_))
			    ).array().unaryExpr(model_type::sigmoid());
	    
	    layer_reset = (theta.Brer_.block(offset_category, 0, theta.hidden_, 1)
			   + (theta.Wrer_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			      * state.derivation().layer(theta.hidden_))
			   + (theta.Wrer_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
			      * state.reduced().layer(theta.hidden_))
			   + (theta.Wrer_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			      * state.stack().layer(theta.hidden_))
			   + (theta.Wrer_.block(offset_category, offset4, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_))
			   ).array().unaryExpr(model_type::sigmoid());
	    
	    layer_hidden = (theta.Bre_.block(offset_category, 0, theta.hidden_, 1)
			    + (theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			       * state.derivation().layer(theta.hidden_))
			    + (theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
			       * state.reduced().layer(theta.hidden_))
			    + (layer_reset.array() * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
						      * state.stack().layer(theta.hidden_)).array()).matrix()
			    + (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_))
			    ).array().unaryExpr(model_type::sigmoid());
	    
	    delta_update
	      = (layer_update.array().unaryExpr(model_type::dsigmoid())
		 * (state.stack().layer(theta.hidden_) - layer_hidden).array()
		 * backward.delta_.array());
	    
	    delta_hidden
	      = (layer_hidden.array().unaryExpr(model_type::dactivation())
		 * (1.0 - layer_update.array())
		 * backward.delta_.array());
	    
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    tensor_type& delta_reduced    = backward_state(theta, state.reduced()).delta_;
	    tensor_type& delta_stack      = backward_state(theta, state.stack()).delta_;
	    
	    delta_stack.array() += layer_update.array() * backward.delta_.array();
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_hidden * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_)
	      += delta_hidden * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_)
	      += (delta_hidden.array() * layer_reset.array()).matrix() * state.stack().layer(theta.hidden_).transpose();
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_)
	      += delta_hidden * candidates.queue_.col(state.span().last_).transpose();
	    Bre += delta_hidden;
	    
	    // propagate to reset...
	    delta_reset
	      = (layer_reset.array().unaryExpr(model_type::dsigmoid())
		 * (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
		    * state.stack().layer(theta.hidden_)).array()
		 * delta_hidden.array());
	    
	    // propagate to antecedent
	    delta_derivation.noalias()
	      += theta.Wre_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() *delta_hidden;
	    
	    delta_reduced.noalias()
	      += theta.Wre_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose() * delta_hidden;
	    
	    delta_stack.noalias()
	      += (theta.Wre_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		  * (delta_hidden.array() * layer_reset.array()).matrix());
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose() * delta_hidden;
	    
	    // reset...
	    tensor_type& Wrer = g.Wrer(state.label());
	    tensor_type& Brer = g.Brer(state.label());
	    
	    Wrer.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_reset * state.derivation().layer(theta.hidden_).transpose();
	    Wrer.block(0, offset2, theta.hidden_, theta.hidden_)
	      += delta_reset * state.reduced().layer(theta.hidden_).transpose();
	    Wrer.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_reset * state.stack().layer(theta.hidden_).transpose();	    
	    Wrer.block(0, offset4, theta.hidden_, theta.hidden_)
	      += delta_reset * candidates.queue_.col(state.span().last_).transpose();
	    Brer += delta_reset;
	    
	    // propagate to antecedent
	    delta_derivation.noalias()
	      += theta.Wrer_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    delta_reduced.noalias()
	      += theta.Wrer_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    delta_stack.noalias()
	      += theta.Wrer_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wrer_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    // update...
	    tensor_type& Wrez = g.Wrez(state.label());
	    tensor_type& Brez = g.Brez(state.label());
	    
	    Wrez.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_update * state.derivation().layer(theta.hidden_).transpose();
	    Wrez.block(0, offset2, theta.hidden_, theta.hidden_)
	      += delta_update * state.reduced().layer(theta.hidden_).transpose();
	    Wrez.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_update * state.stack().layer(theta.hidden_).transpose();	    
	    Wrez.block(0, offset4, theta.hidden_, theta.hidden_)
	      += delta_update * candidates.queue_.col(state.span().last_).transpose();
	    Brez += delta_update;
	    
	    // propagate to antecedent
	    delta_derivation.noalias()
	      += theta.Wrez_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    delta_reduced.noalias()
	      += theta.Wrez_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    delta_stack.noalias()
	      += theta.Wrez_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wrez_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_category = theta.offset_category(state.label());

	    layer_update = (theta.Buz_.block(offset_category, 0, theta.hidden_, 1)
			    + (theta.Wuz_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			       * state.derivation().layer(theta.hidden_))
			    + (theta.Wuz_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
			       * state.stack().layer(theta.hidden_))
			    + (theta.Wuz_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_))
			    ).array().unaryExpr(model_type::sigmoid());
	    
	    layer_reset = (theta.Bur_.block(offset_category, 0, theta.hidden_, 1)
			   + (theta.Wur_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			      * state.derivation().layer(theta.hidden_))
			   + (theta.Wur_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
			      * state.stack().layer(theta.hidden_))
			   + (theta.Wur_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			      * candidates.queue_.col(state.span().last_))
			   ).array().unaryExpr(model_type::sigmoid());

	    layer_hidden = (theta.Bu_.block(offset_category, 0, theta.hidden_, 1)
			    + (theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_)
			       * state.derivation().layer(theta.hidden_))
			    + (layer_reset.array() * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
						      * state.stack().layer(theta.hidden_)).array()).matrix()
			    + (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_)
			       * candidates.queue_.col(state.span().last_))
			    ).array().unaryExpr(model_type::sigmoid());
	    
	    delta_update
	      = (layer_update.array().unaryExpr(model_type::dsigmoid())
		 * (state.stack().layer(theta.hidden_) - layer_hidden).array()
		 * backward.delta_.array());
	    
	    delta_hidden
	      = (layer_hidden.array().unaryExpr(model_type::dactivation())
		 * (1.0 - layer_update.array())
		 * backward.delta_.array());
	    
	    tensor_type& delta_derivation = backward_state(theta, state.derivation(), backward.loss_).delta_;
	    tensor_type& delta_stack      = backward_state(theta, state.stack()).delta_;
	    
	    delta_stack.array() += layer_update.array() * backward.delta_.array();
	    
	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_hidden * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_)
	      += (delta_hidden.array() * layer_reset.array()).matrix() * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_hidden * candidates.queue_.col(state.span().last_).transpose();
	    Bu += delta_hidden;

	    // propagate to reset...
	    delta_reset
	      = (layer_reset.array().unaryExpr(model_type::dsigmoid())
		 * (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_)
		    * state.stack().layer(theta.hidden_)).array()
		 * delta_hidden.array());
	    
	    // propagate to ancedent
	    delta_derivation.noalias()
	      += theta.Wu_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_hidden;
	    
	    delta_stack.noalias()
	      += (theta.Wu_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose()
		  * (delta_hidden.array() * layer_reset.array()).matrix());
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_hidden;
	    
	    // reset...
	    tensor_type& Wur = g.Wur(state.label());
	    tensor_type& Bur = g.Bur(state.label());
	    
	    Wur.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_reset * state.derivation().layer(theta.hidden_).transpose();
	    Wur.block(0, offset2, theta.hidden_, theta.hidden_)
	      += delta_reset * state.stack().layer(theta.hidden_).transpose();
	    Wur.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_reset * candidates.queue_.col(state.span().last_).transpose();
	    Bur += delta_reset;
	    
	    // propagate to ancedent
	    delta_derivation.noalias()
	      += theta.Wur_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    delta_stack.noalias()
	      += theta.Wur_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wur_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_reset;
	    
	    // update...
	    tensor_type& Wuz = g.Wuz(state.label());
	    tensor_type& Buz = g.Buz(state.label());
	    
	    Wuz.block(0, offset1, theta.hidden_, theta.hidden_)
	      += delta_update * state.derivation().layer(theta.hidden_).transpose();
	    Wuz.block(0, offset2, theta.hidden_, theta.hidden_)
	      += delta_update * state.stack().layer(theta.hidden_).transpose();
	    Wuz.block(0, offset3, theta.hidden_, theta.hidden_)
	      += delta_update * candidates.queue_.col(state.span().last_).transpose();
	    Buz += delta_update;
	    
	    // propagate to ancedent
	    delta_derivation.noalias()
	      += theta.Wuz_.block(offset_category, offset1, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    delta_stack.noalias()
	      += theta.Wuz_.block(offset_category, offset2, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    queue_.col(state.span().last_).noalias()
	      += theta.Wuz_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose() * delta_update;
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
	    backward.delta_ = state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation()) * backward.delta_.array();

	    g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bf_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.noalias()
	      += theta.Wf_.transpose() * backward.delta_;
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::IDLE: {
	    backward.delta_ = state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation()) * backward.delta_.array();
	    
	    g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    g.Bi_ += backward.delta_;
	    
	    // propagate to ancedent
	    backward_state(theta, state.derivation(), backward.loss_).delta_.noalias()
	      += theta.Wi_.transpose() * backward.delta_;
	      
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

	queue_.col(i) = candidates.queue_.col(i).array().unaryExpr(model_type::dactivation()) * queue_.col(i).array();
	
	g.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_)
	  += queue_.col(i) * candidates.queue_.col(i + 1).transpose();
	g.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_)
	  += queue_.col(i) * theta.terminal_.col(word_id).transpose();
	g.Bqu_ += queue_.col(i);
	
	queue_.col(i + 1).noalias()
	  += theta.Wqu_.block(0, offset1, theta.hidden_, theta.hidden_).transpose() * queue_.col(i);
	
	g.terminal(word_id).noalias()
	  += theta.Wqu_.block(0, offset2, theta.hidden_, theta.embedding_).transpose() * queue_.col(i);
      }
      
      g.Bqe_ += queue_.col(input.size());
    }
    
    template <>
    inline
    void Margin::propagate(const model::Model8& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient::Model8& g)
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
	  
	  if (! state.operation().axiom()) {
	    const size_type index_operation       = theta.index_operation(state.operation());
	    const size_type offset_operation      = index_operation * theta.hidden_;
	    const size_type offset_classification = theta.offset_classification(state.label());
	    
	    // classification
	    tensor_type& Wc = g.Wc(state.label());
	    tensor_type& Bc = g.Bc(state.label());
	    
	    Wc.block(0, offset_operation, 1, theta.hidden_).noalias() += backward.loss_ * state.layer(theta.hidden_).transpose();
	    Bc.block(0, index_operation, 1, 1).array()                += backward.loss_;
	    
	    // propagate to delta
	    backward.delta_.array()
	      += (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wc_.block(offset_classification, offset_operation, 1, theta.hidden_).transpose()
		     * backward.loss_).array());
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
	    
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    const size_type head_id = theta.terminal(state.head());

	    tensor_type& Wsh = g.Wsh(state.label());
	    tensor_type& Bsh = g.Bsh(state.label());

	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      const size_type full = theta.hidden_ + theta.embedding_ + theta.hidden_;
	      
	      L.block(row, 0, 1, reduced) = ((state.layer(theta.hidden_).transpose()
					      * theta.Psh_.block(full * row + offset1, 0, theta.hidden_, reduced))
					     + (theta.terminal_.col(head_id).transpose()
						* theta.Psh_.block(full * row + offset2, 0, theta.embedding_, reduced))
					     + (candidates.queue_.col(state.span().last_ - 1).transpose()
						* theta.Psh_.block(full * row + offset3, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qsh_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.layer(theta.hidden_))
					     + (theta.Qsh_.block(reduced * row, offset2, reduced, theta.embedding_)
						* theta.terminal_.col(head_id))
					     + (theta.Qsh_.block(reduced * row, offset3, reduced, theta.hidden_)
						* candidates.queue_.col(state.span().last_ - 1))).transpose();
	      
	      g.Psh_.block(full * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Psh_.block(full * row + offset2, 0, theta.embedding_, reduced).noalias()
		+= theta.terminal_.block(0, head_id, theta.embedding_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Psh_.block(full * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.span().last_ - 1, theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
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
		    * candidates.queue_.col(state.span().last_ - 1).transpose()
		    * backward.delta_.row(row)(0,0));
	    }

	    Wsh.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wsh.block(0, offset2, theta.hidden_, theta.embedding_).noalias()
	      += backward.delta_ * theta.terminal_.col(head_id).transpose();
	    Wsh.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.span().last_ - 1).transpose();
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
	    
	      queue_.col(state.span().last_ - 1).array()
		+= (candidates.queue_.col(state.span().last_ - 1).array().unaryExpr(model_type::dactivation())
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
	    
	    queue_.col(state.span().last_ - 1).array()
	      += (candidates.queue_.col(state.span().last_ - 1).array().unaryExpr(model_type::dactivation())
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
	      
	    const size_type offset_category = theta.offset_category(state.label());
	    
	    tensor_type& Wre = g.Wre(state.label());
	    tensor_type& Bre = g.Bre(state.label());

	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      L.block(row, 0, 1, reduced) = ((state.derivation().layer(theta.hidden_).transpose()
					      * theta.Pre_.block(theta.hidden_ * 4 * row + offset1, 0, theta.hidden_, reduced))
					     + (state.reduced().layer(theta.hidden_).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset2, 0, theta.hidden_, reduced))
					     + (state.stack().layer(theta.hidden_).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset3, 0, theta.hidden_, reduced))
					     + (candidates.queue_.col(state.span().last_).transpose()
						* theta.Pre_.block(theta.hidden_ * 4 * row + offset4, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qre_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.derivation().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset2, reduced, theta.hidden_)
						* state.reduced().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset3, reduced, theta.hidden_)
						* state.stack().layer(theta.hidden_))
					     + (theta.Qre_.block(reduced * row, offset4, reduced, theta.hidden_)
						* candidates.queue_.col(state.span().last_))).transpose();
	      
	      g.Pre_.block(theta.hidden_ * 4 * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset2, 0, theta.hidden_, reduced).noalias()
		+= state.reduced().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= state.stack().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pre_.block(theta.hidden_ * 4 * row + offset4, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.span().last_, theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
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
		    * candidates.queue_.col(state.span().last_).transpose()
		    * backward.delta_.row(row)(0,0));
	    }
	    
	    Wre.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wre.block(0, offset2, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	    Wre.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wre.block(0, offset4, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
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
	      
	      queue_.col(state.span().last_).array()
		+= (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
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
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wre_.block(offset_category, offset4, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::UNARY: {
	    const size_type offset1 = 0;
	    const size_type offset2 = theta.hidden_;
	    const size_type offset3 = theta.hidden_ + theta.hidden_;
	    
	    const size_type offset_category       = theta.offset_category(state.label());
	    
	    tensor_type& Wu = g.Wu(state.label());
	    tensor_type& Bu = g.Bu(state.label());
	    
	    for (size_type row = 0; row != theta.hidden_; ++ row) {
	      L.block(row, 0, 1, reduced) = ((state.derivation().layer(theta.hidden_).transpose()
					      * theta.Pu_.block(theta.hidden_ * 3 * row + offset1, 0, theta.hidden_, reduced))
					     + (state.stack().layer(theta.hidden_).transpose()
						* theta.Pu_.block(theta.hidden_ * 3 * row + offset2, 0, theta.hidden_, reduced))
					     + (candidates.queue_.col(state.span().last_).transpose()
						* theta.Pu_.block(theta.hidden_ * 3 * row + offset3, 0, theta.hidden_, reduced)));
	      
	      R.block(row, 0, 1, reduced) = ((theta.Qu_.block(reduced * row, offset1, reduced, theta.hidden_)
					      * state.derivation().layer(theta.hidden_))
					     + (theta.Qu_.block(reduced * row, offset2, reduced, theta.hidden_)
						* state.stack().layer(theta.hidden_))
					     + (theta.Qu_.block(reduced * row, offset3, reduced, theta.hidden_)
						* candidates.queue_.col(state.span().last_))).transpose();
	      
	      g.Pu_.block(theta.hidden_ * 3 * row + offset1, 0, theta.hidden_, reduced).noalias()
		+= state.derivation().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pu_.block(theta.hidden_ * 3 * row + offset2, 0, theta.hidden_, reduced).noalias()
		+= state.stack().layer(theta.hidden_) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      g.Pu_.block(theta.hidden_ * 3 * row + offset3, 0, theta.hidden_, reduced).noalias()
		+= candidates.queue_.block(0, state.span().last_, theta.hidden_, 1) * R.block(row, 0, 1, reduced) * backward.delta_.row(row)(0,0);
	      
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
		    * candidates.queue_.col(state.span().last_).transpose()
		    * backward.delta_.row(row)(0,0));
	    }
	    
	    Wu.block(0, offset1, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	    Wu.block(0, offset2, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * state.stack().layer(theta.hidden_).transpose();
	    Wu.block(0, offset3, theta.hidden_, theta.hidden_).noalias()
	      += backward.delta_ * candidates.queue_.col(state.span().last_).transpose();
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
	      
	      queue_.col(state.span().last_).array()
		+= (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
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
	    
	    queue_.col(state.span().last_).array()
	      += (candidates.queue_.col(state.span().last_).array().unaryExpr(model_type::dactivation())
		  * (theta.Wu_.block(offset_category, offset3, theta.hidden_, theta.hidden_).transpose()
		     * backward.delta_).array());
	    
	    // register state
	    states_[state.derivation().step()].insert(state.derivation());
	  } break;
	  case operation_type::FINAL: {
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
    
  };
};

#endif