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
  };
};

#endif
