//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/spirit/include/phoenix_core.hpp>

#include <boost/fusion/tuple.hpp>
#include <boost/range.hpp>

#include "model.hpp"

#include "utils/compress_stream.hpp"
#include "utils/repository.hpp"

namespace rnnp
{
  Model::model_type Model::model(const path_type& path)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;

    typedef utils::repository repository_type;
    
    if (! repository_type::exists(path))
      return model::NONE;
    
    repository_type rep(path, repository_type::read);

    repository_type::const_iterator riter = rep.find("model");
    if (riter == rep.end())
      throw std::runtime_error("no model parameter?");
    
    std::string::const_iterator iter     = riter->second.begin();
    std::string::const_iterator iter_end = riter->second.end();
    
    int num = 0;
    
    if (! qi::phrase_parse(iter, iter_end, qi::no_case["model"] >> qi::int_, standard::space, num) || iter != iter_end)
      return model::NONE;
    
    switch (num) {
    case 1: return model::MODEL1;
    case 2: return model::MODEL2;
    case 3: return model::MODEL3;
    case 4: return model::MODEL4;
    case 5: return model::MODEL5;
    case 6: return model::MODEL6;
    case 7: return model::MODEL7;
    case 8: return model::MODEL8;
    default: return model::NONE;
    }
  }

  void Model::initialize(const size_type& hidden,
			 const size_type& embedding,
			 const grammar_type& grammar)
  {
    hidden_    = hidden;
    embedding_ = embedding;
    
    if (hidden_ == 0)
      throw std::runtime_error("invalid dimension");
    if (embedding_ == 0)
      throw std::runtime_error("invalid dimension");
    
    // assign vocabulary
    vocab_terminal_.clear();
    vocab_category_.clear();
    
    // terminal index
    grammar_type::label_set_type::const_iterator titer_end = grammar.terminal_.end();
    for (grammar_type::label_set_type::const_iterator titer = grammar.terminal_.begin(); titer != titer_end; ++ titer) {
      if (titer->id() >= vocab_terminal_.size())
	vocab_terminal_.resize(titer->id() + 1);
      
      vocab_terminal_[titer->id()] = true;
    }
    
    // unary index
    if (symbol_type::AXIOM.non_terminal_id() >= vocab_category_.size())
      vocab_category_.resize(symbol_type::AXIOM.non_terminal_id() + 1);
    if (symbol_type::FINAL.non_terminal_id() >= vocab_category_.size())
      vocab_category_.resize(symbol_type::FINAL.non_terminal_id() + 1);
    if (symbol_type::IDLE.non_terminal_id() >= vocab_category_.size())
      vocab_category_.resize(symbol_type::IDLE.non_terminal_id() + 1);
    
    vocab_category_[symbol_type::AXIOM.non_terminal_id()] = symbol_type::AXIOM;
    vocab_category_[symbol_type::FINAL.non_terminal_id()] = symbol_type::FINAL;
    vocab_category_[symbol_type::IDLE.non_terminal_id()]  = symbol_type::IDLE;
    
    grammar_type::label_set_type::const_iterator niter_end = grammar.non_terminal_.end();
    for (grammar_type::label_set_type::const_iterator niter = grammar.non_terminal_.begin(); niter != niter_end; ++ niter) {
      if (niter->non_terminal_id() >= vocab_category_.size())
	vocab_category_.resize(niter->non_terminal_id() + 1);
      
      vocab_category_[niter->non_terminal_id()] = *niter;
    }
  }
  
  void Model::write_embedding(const path_type& path_txt,
			      const path_type& path_bin,
			      const tensor_type& matrix) const
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    const size_type          rows = matrix.rows();
    const word_type::id_type cols = utils::bithack::min(static_cast<size_t>(matrix.cols()), vocab_terminal_.size());
    
    utils::compress_ostream os_txt(path_txt, 1024 * 1024);
    utils::compress_ostream os_bin(path_bin, 1024 * 1024);
    
    os_txt.precision(10);
    
    for (word_type::id_type id = 0; id != cols; ++ id)  
      if (vocab_terminal_[id]) {
	const word_type word(id);

	const parameter_type* data = matrix.col(id).data();
	
	os_txt << word;
	for (size_type i = 0; i != rows; ++ i)
	  os_txt << ' ' << *(data + i);
	os_txt << '\n';
	
	os_bin.write((char*) matrix.col(id).data(), sizeof(tensor_type::Scalar) * rows);
      }
  }

  void Model::read_embedding(const path_type& path_txt,
			     const path_type& path_bin,
			     tensor_type& matrix)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
    typedef boost::fusion::tuple<std::string, parameter_set_type > embedding_parsed_type;
    typedef boost::spirit::istream_iterator iterator_type;
    
    if (path_txt != "-" && ! boost::filesystem::exists(path_txt))
      throw std::runtime_error("no text embedding: " + path_txt.string());
    
    if (path_bin != "-" && ! boost::filesystem::exists(path_bin))
      throw std::runtime_error("no binary embedding: " + path_bin.string());
    
    const size_type rows = matrix.rows();
    
    qi::rule<iterator_type, std::string(), standard::blank_type>           word;
    qi::rule<iterator_type, embedding_parsed_type(), standard::blank_type> parser; 
    
    word   %= qi::lexeme[+(standard::char_ - standard::space)];
    parser %= word >> *qi::double_ >> (qi::eol | qi::eoi);
    
    utils::compress_istream is_txt(path_txt, 1024 * 1024);
    utils::compress_istream is_bin(path_bin, 1024 * 1024);
    
    is_txt.unsetf(std::ios::skipws);
    
    iterator_type iter(is_txt);
    iterator_type iter_end;
    
    embedding_parsed_type parsed;
    
    size_type num_parsed = 0;
    
    while (iter != iter_end) {
      boost::fusion::get<0>(parsed).clear();
      boost::fusion::get<1>(parsed).clear();
      
      if (! boost::spirit::qi::phrase_parse(iter, iter_end, parser, standard::blank, parsed))
	if (iter != iter_end)
	  throw std::runtime_error("embedding parsing failed");
      
      if (boost::fusion::get<1>(parsed).size() != rows)
	throw std::runtime_error("invalid embedding size");
      
      const word_type word = boost::fusion::get<0>(parsed);
      
      if (word.id() >= matrix.cols())
	matrix.conservativeResize(rows, word.id() + 1);
      if (word.id() >= vocab_terminal_.size())
	vocab_terminal_.resize(word.id() + 1, false);
      
      // read from binary data
      if (! is_bin.read((char*) matrix.col(word.id()).data(), sizeof(tensor_type::Scalar) * rows))
	throw std::runtime_error("invalid read! " + path_bin.string());
      
      // assign words
      vocab_terminal_[word.id()] = true;
      
      ++ num_parsed;
    }
    
    const size_type file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(tensor_type::Scalar) * rows * num_parsed)
      throw std::runtime_error("file size does not match: " + path_bin.string());
  }

  void Model::write_weights(const path_type& path,
			    const weights_type& weights) const
  {
    utils::compress_ostream os(path, 1024 * 1024);
    os.precision(10);
    os << weights;
  }
  
  void Model::read_weights(const path_type& path,
			   weights_type& weights)
  {
    utils::compress_istream is(path, 1024 * 1024);
    is >> weights;
  }
  
  
  void Model::write_category(const path_type& path_txt,
			     const path_type& path_bin,
			     const tensor_type& matrix,
			     const size_type rows,
			     const size_type cols) const
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;

    if (cols != matrix.cols())
      throw std::runtime_error("column does not match");
    if (matrix.rows() % rows != 0)
      throw std::runtime_error("rows does not match");
    
    const size_type num_labels = utils::bithack::min(vocab_category_.size(), static_cast<size_type>(matrix.rows() / rows));
    
    utils::compress_ostream os_txt(path_txt, 1024 * 1024);
    utils::compress_ostream os_bin(path_bin, 1024 * 1024);
    
    os_txt.precision(10);
    
    for (size_type i = 0; i != num_labels; ++ i)
      if (vocab_category_[i] != category_type()) {
	os_txt << vocab_category_[i];
	
	for (size_type col = 0; col != cols; ++ col) {
	  const parameter_type* data = matrix.block(rows * i, 0, rows, cols).col(col).data();
	  
	  for (size_type i = 0; i != rows; ++ i)
	    os_txt << ' ' << *(data + i);
	  
	  os_bin.write((char*) data, sizeof(tensor_type::Scalar) * rows);
	}
	
	os_txt << '\n';
      }
  }
  
  void Model::read_category(const path_type& path_txt,
			    const path_type& path_bin,
			    tensor_type& matrix,
			    const size_type rows,
			    const size_type cols)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
    typedef boost::fusion::tuple<std::string, parameter_set_type > matrix_parsed_type;
    typedef boost::spirit::istream_iterator iterator_type;
    
    if (path_txt != "-" && ! boost::filesystem::exists(path_txt))
      throw std::runtime_error("no text matrix: " + path_txt.string());
    
    if (path_bin != "-" && ! boost::filesystem::exists(path_bin))
      throw std::runtime_error("no binary matrix: " + path_bin.string());
    
    if (cols != matrix.cols())
      throw std::runtime_error("column does not match");
    
    qi::rule<iterator_type, std::string(), standard::blank_type>        label;
    qi::rule<iterator_type, matrix_parsed_type(), standard::blank_type> parser;
    
    label  %= qi::lexeme[standard::char_('[') >> +(standard::char_ - standard::space - ']') >> standard::char_(']')];
    parser %= label >> *qi::double_ >> (qi::eol | qi::eoi);
    
    utils::compress_istream is_txt(path_txt, 1024 * 1024);
    utils::compress_istream is_bin(path_bin, 1024 * 1024);
    
    is_txt.unsetf(std::ios::skipws);
    
    iterator_type iter(is_txt);
    iterator_type iter_end;
    
    matrix_parsed_type parsed;
    
    size_type num_parsed = 0;
    
    while (iter != iter_end) {
      boost::fusion::get<0>(parsed).clear();
      boost::fusion::get<1>(parsed).clear();
      
      if (! boost::spirit::qi::phrase_parse(iter, iter_end, parser, standard::blank, parsed))
	if (iter != iter_end)
	  throw std::runtime_error("matrix parsing failed");
      
      if (boost::fusion::get<1>(parsed).size() != rows * cols)
	throw std::runtime_error("invalid matrix size");
      
      const word_type label = boost::fusion::get<0>(parsed);
      const word_type::id_type label_id = label.non_terminal_id();
      
      if (rows * (label_id + 1) > matrix.rows())
	matrix.conservativeResize(rows * (label_id + 1), cols);
      if (label_id >= vocab_category_.size())
	vocab_category_.resize(label_id + 1, word_type());
      
      for (size_type col = 0; col != cols; ++ col)
	is_bin.read((char*) matrix.block(rows * label_id, 0, rows, cols).col(col).data(), sizeof(tensor_type::Scalar) * rows);
      
      // assign labels
      vocab_category_[label_id] = label;
      
      ++ num_parsed;
    }
    
    const size_type file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(tensor_type::Scalar) * rows * cols * num_parsed)
      throw std::runtime_error("file size does not match: " + path_bin.string());
  }
  
  void Model::write_matrix(const path_type& path_txt,
			   const path_type& path_bin,
			   const tensor_type& matrix) const
  {
    utils::compress_ostream os_bin(path_bin, 1024 * 1024);
    utils::compress_ostream os_txt(path_txt, 1024 * 1024);
    os_txt.precision(10);
    
    os_bin.write((char*) matrix.data(), sizeof(tensor_type::Scalar) * matrix.rows() * matrix.cols());
    os_txt << matrix;
  }
  
  void Model::read_matrix(const path_type& path_txt,
			  const path_type& path_bin,
			  tensor_type& matrix)
  {
    const size_type file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(tensor_type::Scalar) * matrix.rows() * matrix.cols())
      throw std::runtime_error("file size does not match: " + path_bin.string());
    
    utils::compress_istream is(path_bin, 1024 * 1024);
    
    is.read((char*) matrix.data(), file_size);
  }
  
  void Model::write_embedding(std::ostream& os,
			      const tensor_type& matrix) const
  {
    const size_type rows = matrix.rows();
    const size_type cols = utils::bithack::min(static_cast<size_type>(matrix.cols()), vocab_terminal_.size());
    const size_type num_words = std::count(vocab_terminal_.begin(), vocab_terminal_.begin() + cols, true);
    
    os.write((char*) &rows,      sizeof(size_type));
    os.write((char*) &num_words, sizeof(size_type));
    
    for (word_type::id_type id = 0; id != cols; ++ id) 
      if (vocab_terminal_[id]) {
	const word_type word(id);
	const size_type word_size = word.size();
	
	os.write((char*) &word_size, sizeof(size_type));
	os.write((char*) &(*word.begin()), word_size);
	os.write((char*) matrix.col(id).data(), sizeof(tensor_type::Scalar) * rows);
      }
  }
  
  void Model::read_embedding(std::istream& is,
			     tensor_type& matrix)
  {
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    buffer_type buffer;
    
    size_type rows      = 0;
    size_type num_words = 0;
    is.read((char*) &rows,      sizeof(size_type));
    is.read((char*) &num_words, sizeof(size_type));

    //vocab_terminal_.clear();
    
    if (matrix.rows() != rows)
      matrix.conservativeResize(rows, matrix.cols());
    
    for (size_type i = 0; i != num_words; ++ i) {
      size_type word_size = 0;
      is.read((char*) &word_size, sizeof(size_type));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      const word_type word(buffer.begin(), buffer.end());
      
      if (word.id() >= matrix.cols())
	matrix.conservativeResize(rows, word.id() + 1);
      if (word.id() >= vocab_terminal_.size())
	vocab_terminal_.resize(word.id() + 1, false);
      
      is.read((char*) matrix.col(word.id()).data(), sizeof(tensor_type::Scalar) * rows);
      
      vocab_terminal_[word.id()] = true;
    }
  }

  void Model::write_weights(std::ostream& os,
			    const weights_type& weights) const
  {
    const size_type size = weights.size();
    
    os.write((char*) &size, sizeof(size_type));
    
    for (feature_type::id_type id = 0; id != size; ++ id) {
      const feature_type feature(id);
      
      const size_type feature_size = feature.size();
      const parameter_type value   = weights[id];
      
      os.write((char*) &feature_size, sizeof(size_type));
      os.write((char*) &(*feature.begin()), feature_size);
      os.write((char*) &value, sizeof(parameter_type));
    }
  }
  
  void Model::read_weights(std::istream& is,
			   weights_type& weights)
  {
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    weights.clear();
    
    buffer_type buffer;
    parameter_type value;
    
    size_type size = 0;
    is.read((char*) &size, sizeof(size_type));
    
    for (size_type i = 0; i != size; ++ i) {
      size_type feature_size = 0;
      is.read((char*) &feature_size, sizeof(size_type));
      
      buffer.resize(feature_size);
      is.read((char*) &(*buffer.begin()), feature_size);
      
      is.read((char*) &value, sizeof(parameter_type));
      
      weights[feature_type(buffer.begin(), buffer.end())] = value;
    }
  }
  
  void Model::write_category(std::ostream& os,
			     const tensor_type& matrix,
			     const size_type rows,
			     const size_type cols) const
  {
    if (cols != matrix.cols())
      throw std::runtime_error("column does not match");
    if (matrix.rows() % rows != 0)
      throw std::runtime_error("rows does not match");

    const size_type label_max = utils::bithack::min(vocab_category_.size(), static_cast<size_type>(matrix.rows() / rows));
    
    size_type num_labels = 0;
    for (size_type id = 0; id != label_max; ++ id) 
      num_labels += (vocab_category_[id] != category_type());
    
    os.write((char*) &rows,       sizeof(size_type));
    os.write((char*) &cols,       sizeof(size_type));
    os.write((char*) &num_labels, sizeof(size_type));
    
    for (size_type id = 0; id != label_max; ++ id) 
      if (vocab_category_[id] != category_type()) {
	const category_type& cat = vocab_category_[id];
	const size_type      cat_size = cat.size();
	
	os.write((char*) &cat_size, sizeof(size_type));
	os.write((char*) &(*cat.begin()), cat_size);
	
	for (size_type col = 0; col != cols; ++ col)
	  os.write((char*) matrix.block(rows * id, 0, rows, cols).col(col).data(), sizeof(tensor_type::Scalar) * rows);
      }
  }
  
  void Model::read_category(std::istream& is,
			    tensor_type& matrix,
			    const size_type rows_hint,
			    const size_type cols_hint)
  {
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    buffer_type buffer;
 
    size_type rows       = 0;
    size_type cols       = 0;
    size_type num_labels = 0;
    
    is.read((char*) &rows,       sizeof(size_type));
    is.read((char*) &cols,       sizeof(size_type));
    is.read((char*) &num_labels, sizeof(size_type));
    
    if (rows != rows_hint)
      throw std::runtime_error("invlaid rows for read category");
    if (cols != cols_hint)
      throw std::runtime_error("invlaid cols for read category");
    
    //vocab_category_.clear();
    
    if (matrix.cols() != cols)
      matrix.conservativeResize(matrix.rows(), cols);
    
    for (size_type i = 0; i != num_labels; ++ i) {
      size_type label_size = 0;
      is.read((char*) &label_size, sizeof(size_type));
      
      buffer.resize(label_size);
      is.read((char*) &(*buffer.begin()), label_size);
      
      const word_type label(buffer.begin(), buffer.end());
      const word_type::id_type label_id = label.non_terminal_id();
      
      if (rows * (label_id + 1) > matrix.rows())
	matrix.conservativeResize(rows * (label_id + 1), cols);
      if (label_id >= vocab_category_.size())
	vocab_category_.resize(label_id + 1, word_type());
      
      for (size_type col = 0; col != cols; ++ col)
	is.read((char*) matrix.block(rows * label_id, 0, rows, cols).col(col).data(), sizeof(tensor_type::Scalar) * rows);
      
      vocab_category_[label_id] = label;
    }
  }
  
  void Model::write_matrix(std::ostream& os,
			   const tensor_type& matrix) const
  {
    const tensor_type::Index rows = matrix.rows();
    const tensor_type::Index cols = matrix.cols();
    
    os.write((char*) &rows, sizeof(tensor_type::Index));
    os.write((char*) &cols, sizeof(tensor_type::Index));
    
    os.write((char*) matrix.data(), sizeof(tensor_type::Scalar) * rows * cols);
  }
  
  void Model::read_matrix(std::istream& is,
			  tensor_type& matrix)
  {
    tensor_type::Index rows = 0;
    tensor_type::Index cols = 0;
    
    is.read((char*) &rows, sizeof(tensor_type::Index));
    is.read((char*) &cols, sizeof(tensor_type::Index));
    
    matrix.resize(rows, cols);
    
    is.read((char*) matrix.data(), sizeof(tensor_type::Scalar) * rows * cols);
  }
  
  Model::tensor_type& Model::plus_equal(tensor_type& x, const tensor_type& y)
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

  Model::tensor_type& Model::minus_equal(tensor_type& x, const tensor_type& y)
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
