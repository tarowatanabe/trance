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
  }
  
  void Gradient::write_matrix(std::ostream& os, const matrix_embedding_type& matrix) const
  {
    size_type size = matrix.size();
    size_type rows = size_type(-1);
    
    os.write((char*) &size, sizeof(size_type));
    
    matrix_embedding_type::const_iterator eiter_end = matrix.end();
    for (matrix_embedding_type::const_iterator eiter = matrix.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_type(-1)) {
	rows = eiter->second.rows();
	
	os.write((char*) &rows, sizeof(size_type));
      } else if (rows != eiter->second.rows())
	throw std::runtime_error("invalid rows?");
      
      const size_type word_size = eiter->first.size();
      
      os.write((char*) &word_size, sizeof(size_type));
      os.write((char*) &(*eiter->first.begin()), word_size);
      os.write((char*) eiter->second.data(), sizeof(tensor_type::Scalar) * rows);
    }
  }
  
  void Gradient::write_matrix(std::ostream& os, const matrix_category_type& matrix) const
  {
    size_type size = matrix.size();
    size_type rows = size_type(-1);
    size_type cols = size_type(-1);
    
    os.write((char*) &size, sizeof(size_type));
    
    matrix_category_type::const_iterator eiter_end = matrix.end();
    for (matrix_category_type::const_iterator eiter = matrix.begin(); eiter != eiter_end; ++ eiter) {
      if (rows == size_type(-1)) {
	rows = eiter->second.rows();
	cols = eiter->second.cols();
	
	os.write((char*) &rows, sizeof(size_type));
	os.write((char*) &cols, sizeof(size_type));
      } else if (rows != eiter->second.rows() || cols != eiter->second.cols())
	throw std::runtime_error("invalid matrix?");
      
      const size_type word_size = eiter->first.size();
      
      os.write((char*) &word_size, sizeof(size_type));
      os.write((char*) &(*eiter->first.begin()), word_size);
      os.write((char*) eiter->second.data(), sizeof(tensor_type::Scalar) * rows * cols);
    }
  }
  
  void Gradient::write_matrix(std::ostream& os, const tensor_type& matrix) const
  {
    const tensor_type::Index rows = matrix.rows();
    const tensor_type::Index cols = matrix.cols();
    
    os.write((char*) &rows, sizeof(tensor_type::Index));
    os.write((char*) &cols, sizeof(tensor_type::Index));
    
    os.write((char*) matrix.data(), sizeof(tensor_type::Scalar) * rows * cols);
  }
  
  void Gradient::read_matrix(std::istream& is, matrix_embedding_type& matrix)
  {
    typedef std::vector<char, std::allocator<char> > buffer_type;

    matrix.clear();
    
    size_type size = 0;
    size_type rows = 0;
    
    is.read((char*) &size, sizeof(size_type));
    
    if (! size) return;
    
    is.read((char*) &rows, sizeof(size_type));
    
    buffer_type buffer;
    
    for (size_type i = 0; i != size; ++ i) {
      size_type word_size = 0;
      is.read((char*) &word_size, sizeof(size_type));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      tensor_type& tensor = matrix[word_type(buffer.begin(), buffer.end())];
      
      tensor.resize(rows, 1);
      
      is.read((char*) tensor.data(), sizeof(tensor_type::Scalar) * rows);
    }
  }
  
  void Gradient::read_matrix(std::istream& is, matrix_category_type& matrix)
  {
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    matrix.clear();
    
    size_type size = 0;
    size_type rows = 0;
    size_type cols = 0;
    
    is.read((char*) &size, sizeof(size_type));
    
    if (! size) return;
    
    is.read((char*) &rows, sizeof(size_type));
    is.read((char*) &cols, sizeof(size_type));
    
    buffer_type buffer;
    
    for (size_type i = 0; i != size; ++ i) {
      size_type word_size = 0;
      is.read((char*) &word_size, sizeof(size_type));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      tensor_type& tensor = matrix[word_type(buffer.begin(), buffer.end())];
      
      tensor.resize(rows, cols);
      
      is.read((char*) tensor.data(), sizeof(tensor_type::Scalar) * rows * cols);
    }
  }
  
  void Gradient::read_matrix(std::istream& is, tensor_type& matrix)
  {
    tensor_type::Index rows = 0;
    tensor_type::Index cols = 0;
    
    is.read((char*) &rows, sizeof(tensor_type::Index));
    is.read((char*) &cols, sizeof(tensor_type::Index));
    
    matrix.resize(rows, cols);
    
    is.read((char*) matrix.data(), sizeof(tensor_type::Scalar) * rows * cols);
  }

  namespace impl
  {
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
  }
  
  Gradient::matrix_embedding_type& Gradient::plus_equal(matrix_embedding_type& x, const matrix_embedding_type& y)
  {
    impl::plus_equal(x, y);
    
    return x;
  }
  
  Gradient::matrix_embedding_type& Gradient::minus_equal(matrix_embedding_type& x, const matrix_embedding_type& y)
  {
    impl::minus_equal(x, y);
    
    return x;
  }

  Gradient::matrix_category_type& Gradient::plus_equal(matrix_category_type& x, const matrix_category_type& y)
  {
    impl::plus_equal(x, y);
    
    return x;
  }
  
  Gradient::matrix_category_type& Gradient::minus_equal(matrix_category_type& x, const matrix_category_type& y)
  {
    impl::minus_equal(x, y);
    
    return x;
  }
  
  
  Gradient::tensor_type& Gradient::plus_equal(tensor_type& x, const tensor_type& y)
  {
    if (x.rows() == y.rows() && x.cols() == y.cols())
      x += y;
    else if (x.rows() == y.rows()) {
      if (x.cols() > y.cols())
	x.block(0, 0, y.rows(), y.cols()) += y;
      else {
	const tensor_type::Index cols = x.cols();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(0, cols, x.rows(), y.cols() - cols).setZero();
	
	x += y;
      }
    } else if (x.cols() == y.cols()) {
      if (x.rows() > y.rows())
	x.block(0, 0, y.rows(), y.cols()) += y;
      else {
	const tensor_type::Index rows = x.rows();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(rows, 0, y.rows() - rows, x.cols()).setZero();
	
	x += y;
      }
    } else {
      // both differ...
      const tensor_type::Index rows_new = utils::bithack::max(x.rows(), y.rows());
      const tensor_type::Index cols_new = utils::bithack::max(x.cols(), y.cols());
      
      tensor_type x_new = tensor_type::Zero(rows_new, cols_new);
      x_new.block(0, 0, x.rows(), x.cols()) = x;
      x_new.block(0, 0, y.rows(), y.cols()) += y;
      
      x.swap(x_new);
    }
    
    return x;
  }

  Gradient::tensor_type& Gradient::minus_equal(tensor_type& x, const tensor_type& y)
  {
    if (x.rows() == y.rows() && x.cols() == y.cols())
      x -= y;
    else if (x.rows() == y.rows()) {
      if (x.cols() > y.cols())
	x.block(0, 0, y.rows(), y.cols()) -= y;
      else {
	const tensor_type::Index cols = x.cols();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(0, cols, x.rows(), y.cols() - cols).setZero();
	
	x -= y;
      }
    } else if (x.cols() == y.cols()) {
      if (x.rows() > y.rows())
	x.block(0, 0, y.rows(), y.cols()) -= y;
      else {
	const tensor_type::Index rows = x.rows();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(rows, 0, y.rows() - rows, x.cols()).setZero();
	
	x -= y;
      }
    } else {
      // both differ...
      const tensor_type::Index rows_new = utils::bithack::max(x.rows(), y.rows());
      const tensor_type::Index cols_new = utils::bithack::max(x.cols(), y.cols());
      
      tensor_type x_new = tensor_type::Zero(rows_new, cols_new);
      x_new.block(0, 0, x.rows(), x.cols()) = x;
      x_new.block(0, 0, y.rows(), y.cols()) -= y;
      
      x.swap(x_new);
    }
    
    return x;
  }
};
