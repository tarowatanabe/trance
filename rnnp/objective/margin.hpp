// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__MARGIN__HPP__
#define __RNNP__OBJECTIVE__MARGIN__HPP__ 1

#include <rnnp/objective.hpp>

namespace rnnp
{
  namespace objective
  {    
    struct Margin : public Objective
    {
    public:
      
      loss_type operator()(const model_type& theta,
			   const parser_type& candidates,
			   const parser_oracle_type& oracles,
			   const option_type& option,
			   gradient_type& g)
      {
	initialize(candidates, oracles);
	
	const double loss = margin(theta, candidates, oracles, option, g);

	if (! backward_.empty())
	  propagate(theta, option, g);
	
	return loss;
      }
      
      loss_type operator()(gradient_type& g) { return loss_type(); }
      
      virtual double margin(const model_type& theta,
			    const parser_type& candidates,
			    const parser_oracle_type& oracles,
			    const option_type& option,
			    gradient_type& g) = 0;
      
    private:
      void propagate(const model_type& theta,
		     const option_type& option,
		     gradient_type& g)
      {
	++ g.count_;
	
	for (difference_type step = states_.size() - 1; step >= 0; -- step) {
	  state_set_type::iterator siter_end = states_[step].end();
	  for (state_set_type::iterator siter = states_[step].begin(); siter != siter_end; ++ siter) {
	    const state_type& state = *siter;
	    
	    backward_type& backward = backward_[state];
	    
	    if (! backward.delta_.rows())
	      backward.delta_ = tensor_type::Zero(theta.hidden_, 1);
	    
	    //std::cerr << "step: " << step << " loss: " << backward.loss_ << std::endl;
	    
	    switch (state.operation().operation()) {
	    case operation_type::AXIOM: {
	      // initial bias
	      g.Ba_ += backward.delta_;
	    } break;
	    case operation_type::SHIFT: {
	      const size_type offset1 = 0;
	      const size_type offset2 = theta.hidden_;
	      
	      const size_type offset_grammar        = theta.offset_grammar(state.label());
	      const size_type offset_classification = theta.offset_classification(state.label());
	      
	      const size_type head_id = theta.terminal(state.head());
	      
	      // classification
	      g.Wc(state.label()) += backward.loss_ * state.layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		       * backward.loss_).array());
	      
	      g.Wsh(state.label()).block(0, offset1, theta.hidden_, theta.hidden_)
		+= backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	      g.Wsh(state.label()).block(0, offset2, theta.hidden_, theta.embedding_)
		+= backward.delta_ * theta.terminal_.col(head_id).transpose();
	      g.Bsh(state.label())
		+= backward.delta_;
	      
	      // propagate to ancedent
	      backward_type& ant = backward_[state.derivation()];
	      
	      ant.loss_  += backward.loss_;
	      
	      if (! ant.delta_.rows())
		ant.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant.delta_.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wsh_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
	      
	      g.terminal(head_id)
		+= (theta.Wsh_.block(offset_grammar, offset2, theta.hidden_, theta.embedding_).transpose()
		    * backward.delta_);
	      
	      // register state
	      states_[state.derivation().step()].insert(state.derivation());
	    } break;
	    case operation_type::REDUCE: {
	      const size_type offset1 = 0;
	      const size_type offset2 = theta.hidden_;
	      
	      const size_type offset_grammar        = theta.offset_grammar(state.label());
	      const size_type offset_classification = theta.offset_classification(state.label());
	      
	      // classification
	      g.Wc(state.label()) += backward.loss_ * state.layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		       * backward.loss_).array());
	      
	      g.Wre(state.label()).block(0, offset1, theta.hidden_, theta.hidden_)
		+= backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	      g.Wre(state.label()).block(0, offset2, theta.hidden_, theta.hidden_)
		+= backward.delta_ * state.reduced().layer(theta.hidden_).transpose();
	      g.Bre(state.label())
		+= backward.delta_;
	      	      
	      // propagate to ancedent
	      backward_type& ant1 = backward_[state.derivation()];
	      backward_type& ant2 = backward_[state.reduced()];

	      ant1.loss_ += backward.loss_;
	      //ant2.loss_ += backward.loss_;
	      
	      if (! ant1.delta_.rows())
		ant1.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      if (! ant2.delta_.rows())
		ant2.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant1.delta_.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wre_.block(offset_grammar, offset1, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
#if 0
	      ant2.delta_.array()
		+= (state.reduced().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wre_.block(offset_grammar, offset2, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
#endif
	      
	      // register state
	      states_[state.derivation().step()].insert(state.derivation());
	    } break;
	    case operation_type::UNARY: {
	      const size_type offset_grammar        = theta.offset_grammar(state.label());
	      const size_type offset_classification = theta.offset_classification(state.label());
	      
	      // classification
	      g.Wc(state.label()) += backward.loss_ * state.layer(theta.hidden_).transpose();
	      
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
	      backward_type& ant = backward_[state.derivation()];
	      
	      ant.loss_ += backward.loss_;
	      
	      if (! ant.delta_.rows())
		ant.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant.delta_.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wu_.block(offset_grammar, 0, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
	      
	      // register state
	      states_[state.derivation().step()].insert(state.derivation());
	    } break;
	    case operation_type::FINAL: {
	      const size_type offset_classification = theta.offset_classification(state.label());
	      
	      // classification
	      g.Wc(state.label()) += backward.loss_ * state.layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		       * backward.loss_).array());
	      
	      g.Wf_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	      g.Bf_ += backward.delta_;
	      
	      // propagate to ancedent
	      backward_type& ant = backward_[state.derivation()];
	      
	      ant.loss_  += backward.loss_;
	      
	      if (! ant.delta_.rows())
		ant.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant.delta_.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wf_.transpose() * backward.delta_).array());
	      
	      // register state
	      states_[state.derivation().step()].insert(state.derivation());
	    } break;
	    case operation_type::IDLE: {
	      const size_type offset_classification = theta.offset_classification(state.label());
	      
	      // classification
	      g.Wc(state.label()) += backward.loss_ * state.layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (state.layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.block(offset_classification, 0, 1, theta.hidden_).transpose()
		       * backward.loss_).array());
	      
	      g.Wi_ += backward.delta_ * state.derivation().layer(theta.hidden_).transpose();
	      g.Bi_ += backward.delta_;
	      
	      // propagate to ancedent
	      backward_type& ant = backward_[state.derivation()];
	      
	      ant.loss_  += backward.loss_;
	      
	      if (! ant.delta_.rows())
		ant.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant.delta_.array()
		+= (state.derivation().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wi_.transpose() * backward.delta_).array());
	      
	      // register state
	      states_[state.derivation().step()].insert(state.derivation());
	    } break;
	    }
	  }
	}
      }
    };
  };
};

#endif
