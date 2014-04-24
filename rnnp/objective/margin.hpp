// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TMT__OBJECTIVE__MARGIN__HPP__
#define __TMT__OBJECTIVE__MARGIN__HPP__ 1

#include <tmt/derivation.hpp>
#include <tmt/decoder.hpp>
#include <tmt/feature_set.hpp>
#include <tmt/objective.hpp>

#include <utils/unordered_map.hpp>
#include <utils/compact_set.hpp>

namespace tmt
{
  namespace objective
  {    
    struct Margin : public Objective
    {
    public:
      typedef std::vector<sentence_type, std::allocator<sentence_type> > sentence_set_type;
      typedef std::vector<bool, std::allocator<bool> > error_type;
    
      template <typename Gen>
      loss_type operator()(const bitext_type& bitext,
			   const model_type& theta,
			   const decoder_type& candidates,
			   const decoder_type& oracles,
			   const option_type& option,
			   gradient_type& g,
			   Gen& gen)
      {
	if (bitext.source().empty() || bitext.targets().empty() || bitext.target().empty())
	  return 0.0;
	
	initialize(bitext);
	
	const size_type kbest_candidate_size = candidates.agenda_.back().size();
	const size_type kbest_oracle_size    = oracles.agenda_.back().size();
      
	error_.clear();
	error_.resize(kbest_candidate_size, false);

	translations_.clear();
	translations_.resize(kbest_candidate_size);
	
	for (size_type c = 0; c != kbest_candidate_size; ++ c) {
	  derivation_.assign(theta, candidates.agenda_.back()[c]);
	  
	  translations_[c] = derivation_.target_;
	  error_[c] = (bitext.target() != derivation_.target_);
	}
	
	const double loss = margin(bitext, theta, candidates, oracles, option, g);
		
	if (! backward_.empty())
	  propagate(bitext, theta, option, g);
	
	return loss;
      }
      
      loss_type operator()(gradient_type& g) { return loss_type(); }

      virtual double margin(const bitext_type& bitext,
			    const model_type& theta,
			    const decoder_type& candidates,
			    const decoder_type& oracles,
			    const option_type& option,
			    gradient_type& g) = 0;
      
    private:
      void propagate(const bitext_type& bitext,
		     const model_type& theta,
		     const option_type& option,
		     gradient_type& g)
      {
	++ g.count_;
	
	for (difference_type step = nodes_.size() - 1; step >= 0; -- step) {
	  node_set_type::iterator niter_end = nodes_[step].end();
	  for (node_set_type::iterator niter = nodes_[step].begin(); niter != niter_end; ++ niter) {
	    const node_type& node = *niter;

	    backward_type& backward = backward_[node.state()];
	    
	    if (! backward.delta_.rows())
	      backward.delta_ = tensor_type::Zero(theta.hidden_, 1);
	    
	    switch (node.state().operation().operation()) {
	    case operation_type::AXIOM: {
	      
	      // feature set
	      if (option.learn_classification()) {
		const feature_vector_type& feats = *node.state().feature_vector();
		
		feature_vector_type::const_iterator fiter_end = feats.end();
		for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
		  g.Wf_[fiter->first] += backward.loss_ * fiter->second;
	      }
	      
	      // initial bias
	      g.Bi_ += backward.delta_;
	    } break;
	    case operation_type::SHIFT: {
	      const size_type offset1 = 0;
	      const size_type offset2 = theta.hidden_;
	      const size_type offset3 = theta.hidden_ + theta.embedding_;
	      
	      // feature set
	      if (option.learn_classification()) {
		const feature_vector_type& feats = *node.state().feature_vector();
		
		feature_vector_type::const_iterator fiter_end = feats.end();
		for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
		  g.Wf_[fiter->first] += backward.loss_ * fiter->second;
	      }
	      
	      // classification
	      g.Wc_ += backward.loss_ * node.state().layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (node.state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.transpose() * backward.loss_).array());
	      
	      // NN
	      const size_type layer_size = utils::bithack::max(size_type(node.state().source().size()),
							       node.state().target().size());
	      
	      layers_.resize(theta.hidden_, layer_size + 1);
	      deltas_.resize(theta.hidden_, layer_size + 1);
	      
	      // forward pass...
	      layers_.col(0) = node.derivation().state().layer(theta.hidden_);
	      
	      for (size_type pos = 0; pos != layer_size; ++ pos) {
		const word_type::id_type source_id = theta.word_source(pos < node.state().source().size()
								       ? bitext.source()[node.state().source().first_ + pos]
								       : vocab_type::EPSILON);
		const word_type::id_type target_id = theta.word_target(pos < node.state().target().size()
								       ? node.state().target()[pos]
								       : vocab_type::EPSILON);
		layers_.col(pos + 1) = (theta.Bsh().block(0, 0, theta.hidden_, 1)
					+ (theta.Wsh().block(0, offset1, theta.hidden_, theta.hidden_)
					   * layers_.col(pos))
					+ (theta.Wsh().block(0, offset2, theta.hidden_, theta.embedding_)
					   * theta.source().col(source_id))
					+ (theta.Wsh().block(0, offset3, theta.hidden_, theta.embedding_)
					   * theta.target().col(target_id))
					).array().unaryExpr(model_type::activation());
	      }
	      
	      // backward pass...
	      deltas_.col(layer_size) = backward.delta_;
	      
	      for (size_type pos = layer_size; pos != 0; -- pos) {
		const word_type::id_type source_id = theta.word_source(pos - 1 < node.state().source().size()
								       ? bitext.source()[node.state().source().first_ + pos - 1]
								       : vocab_type::EPSILON);
		const word_type::id_type target_id = theta.word_target(pos - 1 < node.state().target().size()
								       ? node.state().target()[pos - 1]
								       : vocab_type::EPSILON);
		
		g.Wsh_.block(0, offset1, theta.hidden_, theta.hidden_)
		  += deltas_.col(pos) * layers_.col(pos - 1).transpose();
		g.Wsh_.block(0, offset2, theta.hidden_, theta.embedding_)
		  += deltas_.col(pos) * theta.source().col(source_id).transpose();
		g.Wsh_.block(0, offset3, theta.hidden_, theta.embedding_)
		  += deltas_.col(pos) * theta.target().col(target_id).transpose();
		g.Bsh_.block(0, 0, theta.hidden_, 1)
		  += deltas_.col(pos);
		
		g.head_source(source_id) += backward.loss_ * layers_.col(pos - 1);
		g.head_target(target_id) += backward.loss_ * layers_.col(pos - 1);
		
		deltas_.col(pos - 1)
		  = (layers_.col(pos - 1).array().unaryExpr(model_type::dactivation())
		     * ((theta.Wsh_.block(0, offset1, theta.hidden_, theta.hidden_).transpose()
			 * deltas_.col(pos))
			+ ((theta.head_source().col(source_id) + theta.head_target().col(target_id)) * backward.loss_)).array());
		
		g.source(source_id) += (theta.Wsh_.block(0, offset2, theta.hidden_, theta.embedding_).transpose()
					* deltas_.col(pos));
		g.target(target_id) += (theta.Wsh_.block(0, offset3, theta.hidden_, theta.embedding_).transpose()
					* deltas_.col(pos));
	      }
	      
	      backward_type& ant = backward_[node.derivation().state()];
	      
	      if (! ant.delta_.rows())
		ant.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant.delta_ += deltas_.col(0);
	      ant.loss_  += backward.loss_;
	      
	      nodes_[node.derivation().state().step()].insert(node.derivation());
	    } break;
	    case operation_type::REDUCE_STRAIGHT: {
	      const size_type offset1 = 0;
	      const size_type offset2 = theta.hidden_;
	      const size_type offset_skipped = theta.offset_skip(node.state().operation().skipped());
	      
	      // feature set
	      if (option.learn_classification()) {
		const feature_vector_type& feats = *node.state().feature_vector();
		
		feature_vector_type::const_iterator fiter_end = feats.end();
		for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
		  g.Wf_[fiter->first] += backward.loss_ * fiter->second;
	      }
	      
	      // classification
	      g.Wc_ += backward.loss_ * node.state().layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (node.state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.transpose() * backward.loss_).array());
	      
	      // NN
	      g.Wrs_.block(offset_skipped, offset1, theta.hidden_, theta.hidden_)
		+= backward.delta_ * node.derivation().state().layer(theta.hidden_).transpose();
	      g.Wrs_.block(offset_skipped, offset2, theta.hidden_, theta.hidden_)
		+= backward.delta_ * node.reduced().state().layer(theta.hidden_).transpose();
	      g.Brs_.block(offset_skipped, 0, theta.hidden_, 1)
		+= backward.delta_;
	      
	      // propagate to ancedent
	      backward_type& ant1 = backward_[node.derivation().state()];
	      backward_type& ant2 = backward_[node.reduced().state()];
	      
	      if (! ant1.delta_.rows())
		ant1.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      if (! ant2.delta_.rows())
		ant2.delta_ = tensor_type::Zero(theta.hidden_, 1);
	      
	      ant1.delta_.array()
		+= (node.derivation().state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wrs_.block(offset_skipped, offset1, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
	      ant2.delta_.array()
		+= (node.reduced().state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wrs_.block(offset_skipped, offset2, theta.hidden_, theta.hidden_).transpose()
		       * backward.delta_).array());
	      
	      ant1.loss_ += backward.loss_;
	      ant2.loss_ += backward.loss_;
	      
	      nodes_[node.derivation().state().step()].insert(node.derivation());
	    } break;
	    case operation_type::REDUCE_INVERSION: {
	      const size_type offset1 = 0;
	      const size_type offset2 = theta.hidden_;
	      const size_type offset_skipped = theta.offset_skip(node.state().operation().skipped());
            
	      // feature set
	      if (option.learn_classification()) {
		const feature_vector_type& feats = *node.state().feature_vector();
		
		feature_vector_type::const_iterator fiter_end = feats.end();
		for (feature_vector_type::const_iterator fiter = feats.begin(); fiter != fiter_end; ++ fiter)
		  g.Wf_[fiter->first] += backward.loss_ * fiter->second;
	      }
	      
	      // classification
	      g.Wc_ += backward.loss_ * node.state().layer(theta.hidden_).transpose();
	      
	      // propagate to delta
	      backward.delta_.array()
		+= (node.state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		    * (theta.Wc_.transpose() * backward.loss_).array());
	      
	      // NN
	      g.Wri_.block(offset_skipped, offset1, theta.hidden_, theta.hidden_)
		+= backward.delta_ * node.derivation().state().layer(theta.hidden_).transpose();
	      g.Wri_.block(offset_skipped, offset2, theta.hidden_, theta.hidden_)
		+= backward.delta_ * node.reduced().state().layer(theta.hidden_).transpose();
	      g.Bri_.block(offset_skipped, 0, theta.hidden_, 1)
		+= backward.delta_;
	      
	      // propagate to ancedent
	      backward_type& ant1 = backward_[node.derivation().state()];
	      backward_type& ant2 = backward_[node.reduced().state()];
	      
	       if (! ant1.delta_.rows())
		 ant1.delta_ = tensor_type::Zero(theta.hidden_, 1);
	       if (! ant2.delta_.rows())
		 ant2.delta_ = tensor_type::Zero(theta.hidden_, 1);
	       
	       ant1.delta_.array()
		 += (node.derivation().state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		     * (theta.Wri_.block(offset_skipped, offset1, theta.hidden_, theta.hidden_).transpose()
			* backward.delta_).array());
	       ant2.delta_.array()
		 += (node.reduced().state().layer(theta.hidden_).array().unaryExpr(model_type::dactivation())
		     * (theta.Wri_.block(offset_skipped, offset2, theta.hidden_, theta.hidden_).transpose()
			* backward.delta_).array());
	       
	       ant1.loss_ += backward.loss_;
	       ant2.loss_ += backward.loss_;
	       
	       nodes_[node.derivation().state().step()].insert(node.derivation());
	    } break;
	    }
	  }
	}
      }

    public:
      error_type        error_;
      sentence_set_type translations_;
    };
  };
};

#endif
