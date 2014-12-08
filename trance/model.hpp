// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__MODEL__HPP__
#define __TRANCE__MODEL__HPP__ 1

#include <cmath>
#include <vector>
#include <iostream>

#include <trance/symbol.hpp>
#include <trance/feature.hpp>
#include <trance/grammar.hpp>
#include <trance/signature.hpp>
#include <trance/weight_vector.hpp>
#include <trance/operation.hpp>

#include <trance/model/model_type.hpp>

#include <utils/bithack.hpp>

#include <Eigen/Core>

#include <boost/filesystem/path.hpp>

namespace trance
{
  class Model
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef float     parameter_type;
    
    typedef Symbol symbol_type;
    typedef Symbol word_type;
    
    typedef Feature feature_type;

    typedef Grammar   grammar_type;
    typedef Signature signature_type;

    typedef Operation operation_type;
    
    typedef boost::filesystem::path path_type;
    
    typedef Eigen::Matrix<parameter_type, Eigen::Dynamic, Eigen::Dynamic> tensor_type;
    typedef Eigen::Map<tensor_type>                                       adapted_type;

    typedef WeightVector<parameter_type, std::allocator<parameter_type> > weights_type;

    typedef symbol_type category_type;
    
    typedef std::vector<bool, std::allocator<bool> >                   terminal_set_type;
    typedef std::vector<category_type, std::allocator<category_type> > category_set_type;

    typedef model::ModelType model_type;
    
    struct activation
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	//return std::min(std::max(x, Tp(0)), Tp(63));
	//return Tp(1) / (Tp(1) + std::exp(-x));
	return std::min(std::max(x, Tp(-1)), Tp(1));
	//return std::tanh(x);
      }
    };
    
    struct dactivation
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	//return Tp(0) < x && x < Tp(63);
	//return x * (Tp(1) - x);
	return Tp(- 1) < x && x < Tp(1);
	//return Tp(1) - x * x;
      }
    };

    struct sigmoid
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	// We use a hard version
	return std::min(std::max(x, Tp(0)), Tp(1));
	//return Tp(1) / (Tp(1) + std::exp(-x));
      }
    };
    
    struct dsigmoid
    {
      template <typename Tp>
      Tp operator()(const Tp& x) const
      {
	// We use a hard version
	return Tp(0) < x && x < Tp(1);
	//return x * (Tp(1) - x);
      }
    };

  public:
    static model_type model(const path_type& path);

  public:
    Model() : hidden_(0), embedding_(0) {}
    Model(const size_type& hidden,
	  const size_type& embedding)
      : hidden_(hidden), embedding_(embedding) {}
    Model(const size_type& hidden,
	  const size_type& embedding,
	  const grammar_type& grammar)
      : hidden_(hidden), embedding_(embedding)
    {
      initialize(hidden, embedding, grammar);
    }

    void initialize(const size_type& hidden,
		    const size_type& embedding,
		    const grammar_type& grammar);

    void swap(Model& x)
    {
      std::swap(hidden_,    x.hidden_);
      std::swap(embedding_, x.embedding_);
      
      vocab_terminal_.swap(x.vocab_terminal_);
      vocab_category_.swap(x.vocab_category_);
    }

    void clear() { }
    
    // path based interface
    void write_embedding(const path_type& path_txt,
			 const path_type& path_bin,
			 const tensor_type& matrix) const;
    void read_embedding(const path_type& path_txt,
			const path_type& path_bin,
			tensor_type& matrix);
    
    void write_weights(const path_type& path,
		       const weights_type& weights) const;
    void read_weights(const path_type& path,
		      weights_type& weights);

    void write_category(const path_type& path_txt,
			const path_type& path_bin,
			const tensor_type& matrix,
			const size_type rows,
			const size_type cols) const;
    void read_category(const path_type& path_txt,
		       const path_type& path_bin,
		       tensor_type& matrix,
		       const size_type rows,
		       const size_type cols);
    
    void write_matrix(const path_type& path_txt,
		      const path_type& path_bin,
		      const tensor_type& matrix) const;
    void read_matrix(const path_type& path_txt,
		     const path_type& path_bin,
		     tensor_type& matrix);

    // iostream based interface
    void write_embedding(std::ostream& os,
			 const tensor_type& matrix) const;
    void read_embedding(std::istream& is,
			tensor_type& matrix);

    void write_weights(std::ostream& os,
		       const weights_type& weights) const;
    void read_weights(std::istream& is,
		      weights_type& weights);
    
    void write_category(std::ostream& os,
			const tensor_type& matrix,
			const size_type rows,
			const size_type cols) const;
    void read_category(std::istream& is,
		       tensor_type& matrix,
		       const size_type rows,
		       const size_type cols);
    
    void write_matrix(std::ostream& os,
		      const tensor_type& matrix) const;
    void read_matrix(std::istream& is,
		     tensor_type& matrix);

  public:
    tensor_type& plus_equal(tensor_type& x, const tensor_type& y);
    tensor_type& minus_equal(tensor_type& x, const tensor_type& y);
    
    weights_type& plus_equal(weights_type& x, const weights_type& y)
    {
      x += y;
      return x;
    }
    weights_type& minus_equal(weights_type& x, const weights_type& y)
    {
      x -= y;
      return x;
    }

  public:
    word_type::id_type terminal(const word_type& x) const
    {
      return utils::bithack::branch(x.id() >= vocab_terminal_.size() || ! vocab_terminal_[x.id()],
				    symbol_type::UNK.id(),
				    x.id());
    }
    
    word_type::id_type terminal(const signature_type& signature, const word_type& x) const
    {
      if (x.id() < vocab_terminal_.size() && vocab_terminal_[x.id()])
	return x.id();
      else {
	const word_type sig = signature(x);
	
	return utils::bithack::branch(sig.id() >= vocab_terminal_.size() || ! vocab_terminal_[sig.id()],
				      symbol_type::UNK.id(),
				      sig.id());
      }
    }
    
    size_type offset_classification(const category_type& cat) const
    {
      return cat.non_terminal_id();
    }
    
    size_type offset_category(const category_type& cat) const
    {
      return cat.non_terminal_id() * hidden_;
    }

    size_type index_operation(const operation_type& operation) const
    {
      switch (operation.operation()) {
      case operation_type::SHIFT:
	return 0;
      case operation_type::REDUCE:
      case operation_type::REDUCE_LEFT:
      case operation_type::REDUCE_RIGHT:
	return 1;
      case operation_type::UNARY:
      case operation_type::FINAL:
      case operation_type::IDLE:
	return 2;
      default:
	return size_type(-1);
      }
    }
    
  public:
    // parameters
    size_type hidden_;
    size_type embedding_;
    
    // word set
    terminal_set_type vocab_terminal_;
    category_set_type vocab_category_;
  };
};

namespace std
{
  inline
  void swap(trance::Model& x, trance::Model& y)
  {
    x.swap(y);
  }
};

#endif
