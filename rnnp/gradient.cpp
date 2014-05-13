//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/spirit/include/phoenix_core.hpp>

#include <boost/fusion/tuple.hpp>

#include "gradient.hpp"

#include "utils/repository.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"

namespace rnnp
{
  void Gradient::initialize(const size_type& hidden,
			    const size_type& embedding)
  {
    hidden_    = hidden;
    embedding_ = embedding;
    count_     = 0;
    shared_    = 0;
    
    if (hidden_ == 0)
      throw std::runtime_error("invalid dimension");
    if (embedding_ == 0)
      throw std::runtime_error("invalid dimension");
    
    terminal_.clear();
    
    // initialize matrix    
    Wc_.clear();
    
    Wsh_.clear();
    Bsh_.clear();
    
    Wre_.clear();
    Bre_.clear();
    
    Wu_.clear();
    Bu_.clear();
    
    Wf_ = tensor_type::Zero(hidden_, hidden_);
    Bf_ = tensor_type::Zero(hidden_, 1);
    
    Wi_ = tensor_type::Zero(hidden_, hidden_);
    Bi_ = tensor_type::Zero(hidden_, 1);
    
    Ba_ = tensor_type::Zero(hidden_, 1);
  }
  
  template <typename Embedding>
  inline
  void write_embedding(std::ostream& os, const Embedding& embedding)
  {
    size_t size = embedding.size();
    size_t rows = size_t(-1);
    size_t cols = size_t(-1);
    
    os.write((char*) &size, sizeof(size_t));
    
    typename Embedding::const_iterator eiter_end = embedding.end();
    for (typename Embedding::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_t(-1)) {
	rows = eiter->second.rows();
	cols = eiter->second.cols();
	
	os.write((char*) &rows, sizeof(size_t));
	os.write((char*) &cols, sizeof(size_t));
      }
      
      const size_t word_size = eiter->first.size();
      
      os.write((char*) &word_size, sizeof(size_t));
      os.write((char*) &(*eiter->first.begin()), word_size);
      os.write((char*) eiter->second.data(), sizeof(typename Embedding::mapped_type::Scalar) * rows * cols);
    }
  }
  
  template <typename Embedding>
  inline
  void read_embedding(std::istream& is, Embedding& embedding)
  {
    typedef typename Embedding::key_type    word_type;
    typedef typename Embedding::mapped_type tensor_type;
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    embedding.clear();
    
    buffer_type buffer;
    
    size_t size;
    size_t rows;
    size_t cols;
    
    is.read((char*) &size, sizeof(size_t));
    
    if (! size) return;
    
    is.read((char*) &rows, sizeof(size_t));
    is.read((char*) &cols, sizeof(size_t));
    
    for (size_t i = 0; i != size; ++ i) {
      size_t word_size = 0;
      is.read((char*) &word_size, sizeof(size_t));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      tensor_type& matrix = embedding[word_type(buffer.begin(), buffer.end())];
      
      matrix.resize(rows, cols);
      
      is.read((char*) matrix.data(), sizeof(typename tensor_type::Scalar) * rows * cols);
    }
  }
  
  template <typename Tensor>
  inline
  void write_matrix(std::ostream& os, const Tensor& matrix)
  {
    const typename Tensor::Index rows = matrix.rows();
    const typename Tensor::Index cols = matrix.cols();
    
    os.write((char*) &rows, sizeof(typename Tensor::Index));
    os.write((char*) &cols, sizeof(typename Tensor::Index));
    
    os.write((char*) matrix.data(), sizeof(typename Tensor::Scalar) * rows * cols);
  }

  template <typename Tensor>
  inline
  void read_matrix(std::istream& is, Tensor& matrix)
  {
    typename Tensor::Index rows;
    typename Tensor::Index cols;
    
    is.read((char*) &rows, sizeof(typename Tensor::Index));
    is.read((char*) &cols, sizeof(typename Tensor::Index));
    
    matrix.resize(rows, cols);
    
    is.read((char*) matrix.data(), sizeof(typename Tensor::Scalar) * rows * cols);
  }

#define GRADIENT_STREAM_OPERATOR(Op1, Op2, Stream) \
  Op1(Stream, theta.terminal_); \
  \
  Op1(Stream, theta.Wc_); \
  \
  Op1(Stream, theta.Wsh_); \
  Op1(Stream, theta.Bsh_); \
  \
  Op1(Stream, theta.Wre_); \
  Op1(Stream, theta.Bre_); \
  \
  Op1(Stream, theta.Wu_); \
  Op1(Stream, theta.Bu_); \
  \
  Op2(Stream, theta.Wf_); \
  Op2(Stream, theta.Bf_); \
  \
  Op2(Stream, theta.Wi_); \
  Op2(Stream, theta.Bi_); \
  \
  Op2(Stream, theta.Bi_);

  std::ostream& operator<<(std::ostream& os, const Gradient& theta)
  {
    os.write((char*) &theta.hidden_,    sizeof(theta.hidden_));
    os.write((char*) &theta.embedding_, sizeof(theta.embedding_));
    os.write((char*) &theta.count_,     sizeof(theta.count_));

    GRADIENT_STREAM_OPERATOR(write_embedding, write_matrix, os);
    
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Gradient& theta)
  {
    is.read((char*) &theta.hidden_,    sizeof(theta.hidden_));
    is.read((char*) &theta.embedding_, sizeof(theta.embedding_));
    is.read((char*) &theta.count_,     sizeof(theta.count_));

    GRADIENT_STREAM_OPERATOR(read_embedding, read_matrix, is);
    
    return is;
  }

#undef GRADIENT_STREAM_OPERATOR

  template <typename Embedding>
  inline 
  void plus_equal_embedding(Embedding& x, const Embedding& y)
  {
    typename Embedding::const_iterator iter_end = y.end();
    for (typename Embedding::const_iterator iter = y.begin(); iter != iter_end; ++ iter) {
      typename Embedding::mapped_type& matrix = x[iter->first];

      if (! matrix.rows())
	matrix = iter->second;
      else
	matrix += iter->second;
    }
  }

  template <typename Embedding>
  inline 
  void minus_equal_embedding(Embedding& x, const Embedding& y)
  {
    typename Embedding::const_iterator iter_end = y.end();
    for (typename Embedding::const_iterator iter = y.begin(); iter != iter_end; ++ iter) {
      typename Embedding::mapped_type& matrix = x[iter->first];

      if (! matrix.rows())
	matrix = - iter->second;
      else
	matrix -= iter->second;
    }    
  }

  template <typename Tensor>
  inline 
  void plus_equal(Tensor& x, const Tensor& y)
  {
    x += y;
  }

  template <typename Tensor>
  inline 
  void minus_equal(Tensor& x, const Tensor& y)
  {
    x -= y;
  }

#define GRADIENT_BINARY_OPERATOR(Op1, Op2)	\
  Op1(terminal_, x.terminal_); \
  \
  Op1(Wc_, x.Wc_); \
  \
  Op1(Wsh_, x.Wsh_); \
  Op1(Bsh_, x.Bsh_); \
  \
  Op1(Wre_, x.Wre_); \
  Op1(Bre_, x.Bre_); \
  \
  Op1(Wu_, x.Wu_); \
  Op1(Bu_, x.Bu_); \
  \
  Op2(Wf_, x.Wf_); \
  Op2(Bf_, x.Bf_); \
  \
  Op2(Wi_, x.Wi_); \
  Op2(Bi_, x.Bi_); \
  \
  Op2(Ba_, x.Ba_);

  
  Gradient& Gradient::operator+=(const Gradient& x)
  {
    GRADIENT_BINARY_OPERATOR(plus_equal_embedding, plus_equal);

    return *this;
  }
  
  Gradient& Gradient::operator-=(const Gradient& x)
  {
    GRADIENT_BINARY_OPERATOR(minus_equal_embedding, minus_equal);

    return *this;
  }

#undef GRADIENT_BINARY_OPERATOR

};
