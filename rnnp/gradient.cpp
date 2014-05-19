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
    
    Wsh_ = tensor_type::Zero(hidden_, hidden_ + embedding_);
    Bsh_ = tensor_type::Zero(hidden_, 1);
    
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
  
  inline
  void write_matrix(std::ostream& os, const Gradient::embedding_type& embedding)
  {
    size_t size = embedding.size();
    size_t rows = size_t(-1);
    
    os.write((char*) &size, sizeof(size_t));
    
    Gradient::embedding_type::const_iterator eiter_end = embedding.end();
    for (Gradient::embedding_type::const_iterator eiter = embedding.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_t(-1)) {
	rows = eiter->second.rows();
	
	os.write((char*) &rows, sizeof(size_t));
      }
      
      const size_t word_size = eiter->first.size();
      
      os.write((char*) &word_size, sizeof(size_t));
      os.write((char*) &(*eiter->first.begin()), word_size);
      os.write((char*) eiter->second.data(), sizeof(Gradient::tensor_type::Scalar) * rows);
    }
  }
  
  inline
  void read_matrix(std::istream& is, Gradient::embedding_type& embedding)
  {
    typedef Gradient::word_type word_type;
    typedef Gradient::tensor_type tensor_type;
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    embedding.clear();
    
    buffer_type buffer;
    
    size_t size;
    size_t rows;
    
    is.read((char*) &size, sizeof(size_t));
    
    if (! size) return;
    
    is.read((char*) &rows, sizeof(size_t));
    
    for (size_t i = 0; i != size; ++ i) {
      size_t word_size = 0;
      is.read((char*) &word_size, sizeof(size_t));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      tensor_type& tensor = embedding[word_type(buffer.begin(), buffer.end())];
      
      tensor.resize(rows, 1);
      
      is.read((char*) tensor.data(), sizeof(tensor_type::Scalar) * rows);
    }
  }

  inline
  void write_matrix(std::ostream& os, const Gradient::matrix_unary_type& matrix)
  {
    size_t size = matrix.size();
    size_t rows = size_t(-1);
    size_t cols = size_t(-1);
    
    os.write((char*) &size, sizeof(size_t));
    
    Gradient::matrix_unary_type::const_iterator eiter_end = matrix.end();
    for (Gradient::matrix_unary_type::const_iterator eiter = matrix.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_t(-1)) {
	rows = eiter->second.rows();
	cols = eiter->second.cols();
	
	os.write((char*) &rows, sizeof(size_t));
	os.write((char*) &cols, sizeof(size_t));
      }
      
      const size_t word_size = eiter->first.size();
      
      os.write((char*) &word_size, sizeof(size_t));
      os.write((char*) &(*eiter->first.begin()), word_size);
      os.write((char*) eiter->second.data(), sizeof(Gradient::tensor_type::Scalar) * rows * cols);
    }
  }
  
  inline
  void read_matrix(std::istream& is, Gradient::matrix_unary_type& matrix)
  {
    typedef Gradient::word_type word_type;
    typedef Gradient::tensor_type tensor_type;
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    matrix.clear();
    
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
      
      tensor_type& tensor = matrix[word_type(buffer.begin(), buffer.end())];
      
      tensor.resize(rows, cols);
      
      is.read((char*) tensor.data(), sizeof(tensor_type::Scalar) * rows * cols);
    }
  }
  
  inline
  void write_matrix(std::ostream& os, const Gradient::matrix_binary_type& matrix)
  {
    size_t size = matrix.size();
    size_t rows = size_t(-1);
    size_t cols = size_t(-1);
    
    os.write((char*) &size, sizeof(size_t));
    
    Gradient::matrix_binary_type::const_iterator eiter_end = matrix.end();
    for (Gradient::matrix_binary_type::const_iterator eiter = matrix.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_t(-1)) {
	rows = eiter->second.rows();
	cols = eiter->second.cols();
	
	os.write((char*) &rows, sizeof(size_t));
	os.write((char*) &cols, sizeof(size_t));
      }

      const size_t left_size  = eiter->first.first.size();
      const size_t right_size = eiter->first.second.size();
      
      os.write((char*) &left_size, sizeof(size_t));
      os.write((char*) &(*eiter->first.first.begin()), left_size);

      os.write((char*) &right_size, sizeof(size_t));
      os.write((char*) &(*eiter->first.second.begin()), right_size);
      
      os.write((char*) eiter->second.data(), sizeof(Gradient::tensor_type::Scalar) * rows * cols);
    }
  }
  
  inline
  void read_matrix(std::istream& is, Gradient::matrix_binary_type& matrix)
  {
    typedef Gradient::word_type word_type;
    typedef Gradient::tensor_type tensor_type;
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    matrix.clear();
    
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

      const word_type left(buffer.begin(), buffer.end());
      
      is.read((char*) &word_size, sizeof(size_t));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      const word_type right(buffer.begin(), buffer.end());
      
      tensor_type& tensor = matrix[std::make_pair(left, right)];
      
      tensor.resize(rows, cols);
      
      is.read((char*) tensor.data(), sizeof(tensor_type::Scalar) * rows * cols);
    }
  }  
  
  inline
  void write_matrix(std::ostream& os, const Gradient::tensor_type& matrix)
  {
    const Gradient::tensor_type::Index rows = matrix.rows();
    const Gradient::tensor_type::Index cols = matrix.cols();
    
    os.write((char*) &rows, sizeof(Gradient::tensor_type::Index));
    os.write((char*) &cols, sizeof(Gradient::tensor_type::Index));
    
    os.write((char*) matrix.data(), sizeof(Gradient::tensor_type::Scalar) * rows * cols);
  }

  inline
  void read_matrix(std::istream& is, Gradient::tensor_type& matrix)
  {
    Gradient::tensor_type::Index rows;
    Gradient::tensor_type::Index cols;
    
    is.read((char*) &rows, sizeof(Gradient::tensor_type::Index));
    is.read((char*) &cols, sizeof(Gradient::tensor_type::Index));
    
    matrix.resize(rows, cols);
    
    is.read((char*) matrix.data(), sizeof(Gradient::tensor_type::Scalar) * rows * cols);
  }

#define GRADIENT_STREAM_OPERATOR(Op, Stream) \
  Op(Stream, theta.terminal_); \
  \
  Op(Stream, theta.Wc_); \
  \
  Op(Stream, theta.Wsh_); \
  Op(Stream, theta.Bsh_); \
  \
  Op(Stream, theta.Wre_); \
  Op(Stream, theta.Bre_); \
  \
  Op(Stream, theta.Wu_); \
  Op(Stream, theta.Bu_); \
  \
  Op(Stream, theta.Wf_); \
  Op(Stream, theta.Bf_); \
  \
  Op(Stream, theta.Wi_); \
  Op(Stream, theta.Bi_); \
  \
  Op(Stream, theta.Bi_);

  std::ostream& operator<<(std::ostream& os, const Gradient& theta)
  {
    os.write((char*) &theta.hidden_,    sizeof(theta.hidden_));
    os.write((char*) &theta.embedding_, sizeof(theta.embedding_));
    os.write((char*) &theta.count_,     sizeof(theta.count_));

    GRADIENT_STREAM_OPERATOR(write_matrix, os);
    
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Gradient& theta)
  {
    is.read((char*) &theta.hidden_,    sizeof(theta.hidden_));
    is.read((char*) &theta.embedding_, sizeof(theta.embedding_));
    is.read((char*) &theta.count_,     sizeof(theta.count_));

    GRADIENT_STREAM_OPERATOR(read_matrix, is);
    
    return is;
  }

#undef GRADIENT_STREAM_OPERATOR

  template <typename Matrix>
  inline 
  void plus_equal(Matrix& x, const Matrix& y)
  {
    typename Matrix::const_iterator iter_end = y.end();
    for (typename Matrix::const_iterator iter = y.begin(); iter != iter_end; ++ iter) {
      typename Matrix::mapped_type& matrix = x[iter->first];

      if (! matrix.rows())
	matrix = iter->second;
      else
	matrix += iter->second;
    }
  }

  template <typename Matrix>
  inline 
  void minus_equal(Matrix& x, const Matrix& y)
  {
    typename Matrix::const_iterator iter_end = y.end();
    for (typename Matrix::const_iterator iter = y.begin(); iter != iter_end; ++ iter) {
      typename Matrix::mapped_type& matrix = x[iter->first];

      if (! matrix.rows())
	matrix = - iter->second;
      else
	matrix -= iter->second;
    }    
  }

  inline 
  void plus_equal(Gradient::tensor_type& x, const Gradient::tensor_type& y)
  {
    x += y;
  }

  inline 
  void minus_equal(Gradient::tensor_type& x, const Gradient::tensor_type& y)
  {
    x -= y;
  }

#define GRADIENT_BINARY_OPERATOR(Op) \
  Op(terminal_, x.terminal_); \
  \
  Op(Wc_, x.Wc_); \
  \
  Op(Wsh_, x.Wsh_); \
  Op(Bsh_, x.Bsh_); \
  \
  Op(Wre_, x.Wre_); \
  Op(Bre_, x.Bre_); \
  \
  Op(Wu_, x.Wu_); \
  Op(Bu_, x.Bu_); \
  \
  Op(Wf_, x.Wf_); \
  Op(Bf_, x.Bf_); \
  \
  Op(Wi_, x.Wi_); \
  Op(Bi_, x.Bi_); \
  \
  Op(Ba_, x.Ba_);

  
  Gradient& Gradient::operator+=(const Gradient& x)
  {
    GRADIENT_BINARY_OPERATOR(plus_equal);

    return *this;
  }
  
  Gradient& Gradient::operator-=(const Gradient& x)
  {
    GRADIENT_BINARY_OPERATOR(minus_equal);

    return *this;
  }

#undef GRADIENT_BINARY_OPERATOR

};
