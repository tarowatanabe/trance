// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OBJECTIVE__HPP__
#define __RNNP__OBJECTIVE__HPP__ 1

#include <vector>

#include <rnnp/unigram.hpp>
#include <rnnp/derivation.hpp>
#include <rnnp/decoder.hpp>
#include <rnnp/bitext.hpp>
#include <rnnp/model.hpp>
#include <rnnp/gradient.hpp>
#include <rnnp/learn_option.hpp>
#include <rnnp/loss.hpp>

#include <utils/unordered_map.hpp>
#include <utils/compact_set.hpp>

namespace rnnp
{
  struct Objective
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef Bitext bitext_type;

    typedef Unigram  unigram_type;    
    typedef Model    model_type;
    typedef Gradient gradient_type;

    typedef model_type::word_type      word_type;
    typedef model_type::parameter_type parameter_type;
    typedef model_type::tensor_type    tensor_type;
    typedef model_type::matrix_type    matrix_type;

    typedef LearnOption option_type;

    typedef Decoder    decoder_type;
    typedef Derivation derivation_type;    

    typedef decoder_type::operation_type      operation_type;
    typedef decoder_type::state_type          state_type;
    typedef decoder_type::node_type           node_type;
    typedef decoder_type::phrase_span_type    phrase_span_type;
    typedef decoder_type::sentence_type       sentence_type;

    typedef Loss loss_type;
    
    struct backward_type
    {
      double      loss_;
      tensor_type delta_;
      
      backward_type() : loss_(0.0), delta_() {}
    };
    
    typedef utils::unordered_map<state_type, backward_type,
				 boost::hash<state_type>, std::equal_to<state_type>,
				 std::allocator<std::pair<const state_type, backward_type> > >::type backward_set_type;
    
    typedef utils::compact_set<node_type, 
			       utils::unassigned<node_type>, utils::unassigned<node_type>,
			       boost::hash<node_type>, std::equal_to<node_type>,
			       std::allocator<node_type> > node_set_type;
    typedef std::vector<node_set_type, std::allocator<node_set_type> > node_map_type;

    void initialize(const bitext_type& bitext)
    {
      backward_.clear();
      
      nodes_.clear();
      nodes_.resize(bitext.source_.size() * 2 + 1);
    }
    
    derivation_type   derivation_;
    backward_set_type backward_;
    node_map_type     nodes_;

    tensor_type layers_;
    tensor_type deltas_;
  };
  
};

#endif
