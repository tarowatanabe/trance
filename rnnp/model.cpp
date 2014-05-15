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

#include "utils/repository.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"

namespace rnnp
{
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
    vocab_non_terminal_.clear();
    
    grammar_type::label_set_type::const_iterator titer_end = grammar.terminal_.end();
    for (grammar_type::label_set_type::const_iterator titer = grammar.terminal_.begin(); titer != titer_end; ++ titer) {
      if (titer->id() >= vocab_terminal_.size())
	vocab_terminal_.resize(titer->id() + 1);
      
      vocab_terminal_[titer->id()] = true;
    }
    
    grammar_type::label_set_type::const_iterator niter_end = grammar.non_terminal_.end();
    for (grammar_type::label_set_type::const_iterator niter = grammar.non_terminal_.begin(); niter != niter_end; ++ niter) {
      if (niter->non_terminal_id() >= vocab_non_terminal_.size())
	vocab_non_terminal_.resize(niter->non_terminal_id() + 1);
      
      vocab_non_terminal_[niter->non_terminal_id()] = *niter;
    }
    
    // initialize matrixx
    terminal_ = tensor_type::Zero(embedding_, vocab_terminal_.size());
    
    Wc_  = tensor_type::Zero(1 * vocab_non_terminal_.size(), hidden_);
    
    Wsh_ = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), hidden_ + embedding_);
    Bsh_ = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), 1);
    
    Wre_ = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), hidden_ + hidden_);
    Bre_ = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), 1);

    Wu_  = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), hidden_);
    Bu_  = tensor_type::Zero(hidden_ * vocab_non_terminal_.size(), 1);
    
    Wf_ = tensor_type::Zero(hidden_, hidden_);
    Bf_ = tensor_type::Zero(hidden_, 1);

    Wi_ = tensor_type::Zero(hidden_, hidden_);
    Bi_ = tensor_type::Zero(hidden_, 1);
    
    Ba_ = tensor_type::Zero(hidden_, 1);
  }
  
  template <typename Tp>
  struct real_policy : boost::spirit::karma::real_policies<Tp>
  {
    static unsigned int precision(Tp)
    {
      return 10;
    }
  };
  
  template <typename Path, typename Tensor, typename WordMap>
  inline
  void write_embedding(const Path& path_txt,
		       const Path& path_bin,
		       const Tensor& embedding,
		       const WordMap& words)
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    typedef rnnp::Model::parameter_type parameter_type;
    typedef rnnp::Model::word_type      word_type;
    
    const word_type::id_type rows = embedding.rows();
    const word_type::id_type cols = utils::bithack::min(static_cast<size_t>(embedding.cols()), words.size());
    
    utils::compress_ostream os_txt(path_txt, 1024 * 1024);
    utils::compress_ostream os_bin(path_bin, 1024 * 1024);
    std::ostream_iterator<char> iter(os_txt);
    
    karma::real_generator<parameter_type, real_policy<parameter_type> > float10;
    
    for (word_type::id_type id = 0; id != cols; ++ id)  
      if (words[id]) {
	// text output
	const word_type word(id);
	
	karma::generate(iter, standard::string, word);
	
	for (size_t j = 0; j != rows; ++ j)
	  karma::generate(iter, karma::lit(' ') << float10, embedding(j, id));
	
	karma::generate(iter, karma::lit('\n'));
	
	// binary output
	os_bin.write((char*) embedding.col(id).data(), sizeof(typename Tensor::Scalar) * rows);
      }
  }
  
  template <typename Path, typename Tensor, typename WordSet>
  inline
  void write_label_matrix(const Path& path_txt,
			  const Path& path_bin,
			  const Tensor& matrix,
			  const WordSet& words,
			  const Model::size_type rows,
			  const Model::size_type cols)
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    typedef rnnp::Model::size_type      size_type;
    typedef rnnp::Model::parameter_type parameter_type;
    typedef rnnp::Model::word_type      word_type;
    
    const size_type num_labels = utils::bithack::min(words.size(), static_cast<size_type>(matrix.rows() / rows));

    utils::compress_ostream os_txt(path_txt, 1024 * 1024);
    utils::compress_ostream os_bin(path_bin, 1024 * 1024);
    std::ostream_iterator<char> iter(os_txt);
    
    karma::real_generator<parameter_type, real_policy<parameter_type> > float10;

    for (size_type i = 0; i != num_labels; ++ i)
      if (words[i] != word_type()) {
	
	karma::generate(iter, standard::string, words[i]);

	for (size_type col = 0; col != cols; ++ col) {
	  const parameter_type* data = matrix.block(rows * i, 0, rows, cols).col(col).data();
	  
	  karma::generate(iter, +(' ' << float10), boost::make_iterator_range(data, data + rows));
	  os_bin.write((char*) data, sizeof(typename Tensor::Scalar) * rows);
	}

	karma::generate(iter, '\n');
      }
  }
  
  template <typename Path, typename Tensor>
  inline
  void write_matrix(const Path& path_txt,
		    const Path& path_bin,
		    const Tensor& matrix)
  {
    {
      utils::compress_ostream os(path_txt, 1024 * 1024);
      os.precision(10);
      os << matrix;
    }
    
    {
      utils::compress_ostream os(path_bin, 1024 * 1024);
      
      const typename Tensor::Index rows = matrix.rows();
      const typename Tensor::Index cols = matrix.cols();
      
      os.write((char*) matrix.data(), sizeof(typename Tensor::Scalar) * rows * cols);
    }
  }

  void Model::write(const path_type& path) const
  {
    typedef utils::repository repository_type;
    
    repository_type rep(path, repository_type::write);
    
    rep["embedding"] = utils::lexical_cast<std::string>(embedding_);
    rep["hidden"]    = utils::lexical_cast<std::string>(hidden_);

    write_embedding(rep.path("terminal.txt.gz"), rep.path("terminal.bin"), terminal_, vocab_terminal_);
    
    write_label_matrix(rep.path("Wc.txt.gz"),  rep.path("Wc.bin"),  Wc_,  vocab_non_terminal_, 1, hidden_);
    
    write_label_matrix(rep.path("Wsh.txt.gz"), rep.path("Wsh.bin"), Wsh_, vocab_non_terminal_, hidden_, hidden_ + embedding_);
    write_label_matrix(rep.path("Bsh.txt.gz"), rep.path("Bsh.bin"), Bsh_, vocab_non_terminal_, hidden_, 1);
    
    write_label_matrix(rep.path("Wre.txt.gz"), rep.path("Wre.bin"), Wre_, vocab_non_terminal_, hidden_, hidden_ + hidden_);
    write_label_matrix(rep.path("Bre.txt.gz"), rep.path("Bre.bin"), Bre_, vocab_non_terminal_, hidden_, 1);
    
    write_label_matrix(rep.path("Wu.txt.gz"),  rep.path("Wu.bin"),  Wu_,  vocab_non_terminal_, hidden_, hidden_);
    write_label_matrix(rep.path("Bu.txt.gz"),  rep.path("Bu.bin"),  Bu_,  vocab_non_terminal_, hidden_, 1);
    
    write_matrix(rep.path("Wf.txt.gz"), rep.path("Wf.bin"), Wf_);
    write_matrix(rep.path("Bf.txt.gz"), rep.path("Bf.bin"), Bf_);
    
    write_matrix(rep.path("Wi.txt.gz"), rep.path("Wi.bin"), Wi_);
    write_matrix(rep.path("Bi.txt.gz"), rep.path("Bi.bin"), Bi_);
    
    write_matrix(rep.path("Ba.txt.gz"), rep.path("Ba.bin"), Ba_);
  }

  template <typename Path, typename Tensor, typename WordSet>
  inline
  void inline_read_embedding(const Path& path_txt,
			     const Path& path_bin,
			     Tensor& embedding,
			     WordSet& words,
			     size_t rows)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;

    typedef rnnp::Model::parameter_type parameter_type;
    typedef rnnp::Model::word_type      word_type;
    
    typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
    typedef boost::fusion::tuple<std::string, parameter_set_type > embedding_parsed_type;
    typedef boost::spirit::istream_iterator iterator_type;
    
    if (path_txt != "-" && ! boost::filesystem::exists(path_txt))
      throw std::runtime_error("no text embedding: " + path_txt.string());

    if (path_bin != "-" && ! boost::filesystem::exists(path_bin))
      throw std::runtime_error("no binary embedding: " + path_bin.string());

    if (embedding.rows() != rows)
      embedding.conservativeResize(rows, embedding.cols());

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

    size_t num_parsed = 0;

    while (iter != iter_end) {
      boost::fusion::get<0>(parsed).clear();
      boost::fusion::get<1>(parsed).clear();
      
      if (! boost::spirit::qi::phrase_parse(iter, iter_end, parser, standard::blank, parsed))
	if (iter != iter_end)
	  throw std::runtime_error("embedding parsing failed");
      
      if (boost::fusion::get<1>(parsed).size() != rows)
	throw std::runtime_error("invalid embedding size");
      
      const word_type word = boost::fusion::get<0>(parsed);
      
      if (word.id() >= embedding.cols())
	embedding.conservativeResize(rows, word.id() + 1);
      if (word.id() >= words.size())
	words.resize(word.id() + 1, false);
      
      // read from binary data
      if (! is_bin.read((char*) embedding.col(word.id()).data(), sizeof(typename Tensor::Scalar) * rows))
	throw std::runtime_error("invalid read! " + path_bin.string());
      
      // assign words
      words[word.id()] = true;

      ++ num_parsed;
    }

    const size_t file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(typename Tensor::Scalar) * rows * num_parsed)
      throw std::runtime_error("file size does not match: " + path_bin.string());
  }
  
  template <typename Path, typename Tensor, typename WordSet>
  inline
  void read_label_matrix(const Path& path_txt,
			 const Path& path_bin,
			 Tensor& matrix,
			 WordSet& labels,
			 const Model::size_type rows,
			 const Model::size_type cols)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    typedef rnnp::Model::parameter_type parameter_type;
    typedef rnnp::Model::word_type      word_type;
    typedef rnnp::Model::size_type      size_type;
    
    typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
    typedef boost::fusion::tuple<std::string, parameter_set_type > matrix_parsed_type;
    typedef boost::spirit::istream_iterator iterator_type;
    
    if (path_txt != "-" && ! boost::filesystem::exists(path_txt))
      throw std::runtime_error("no text matrix: " + path_txt.string());
    
    if (path_bin != "-" && ! boost::filesystem::exists(path_bin))
      throw std::runtime_error("no binary matrix: " + path_bin.string());

    if (matrix.cols() != cols)
      matrix.conservativeResize(matrix.rows(), cols);

    qi::rule<iterator_type, std::string(), standard::blank_type>        word;
    qi::rule<iterator_type, matrix_parsed_type(), standard::blank_type> parser; 
    
    word   %= qi::lexeme[+(standard::char_ - standard::space)];
    parser %= word >> *qi::double_ >> (qi::eol | qi::eoi);
    
    utils::compress_istream is_txt(path_txt, 1024 * 1024);
    utils::compress_istream is_bin(path_bin, 1024 * 1024);
    
    is_txt.unsetf(std::ios::skipws);
    
    iterator_type iter(is_txt);
    iterator_type iter_end;
    
    matrix_parsed_type parsed;

    size_t num_parsed = 0;

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
      if (label_id >= labels.size())
	labels.resize(label_id + 1, word_type());
      
      for (size_type col = 0; col != cols; ++ col)
	is_bin.read((char*) matrix.block(rows * label_id, 0, rows, cols).col(col).data(),
		    sizeof(typename Tensor::Scalar) * rows);
      
      // assign labels
      labels[label_id] = label;
      
      ++ num_parsed;
    }
    
    const size_t file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(typename Tensor::Scalar) * rows * cols * num_parsed)
      throw std::runtime_error("file size does not match: " + path_bin.string());
  }

  template <typename Path, typename Tensor>
  inline
  void read_matrix(const Path& path_txt,
		   const Path& path_bin,
		   Tensor& matrix)
  {
    const size_t file_size = boost::filesystem::file_size(path_bin);
    
    if (file_size != sizeof(typename Tensor::Scalar) * matrix.rows() * matrix.cols())
      throw std::runtime_error("file size does not match: " + path_bin.string());
    
    utils::compress_istream is(path_bin, 1024 * 1024);
    
    is.read((char*) matrix.data(), file_size);
  }

  template <typename Value>
  inline
  Value repository_value(const utils::repository& rep, const std::string& key)
  {
    utils::repository::const_iterator iter = rep.find(key);
    if (iter == rep.end())
      throw std::runtime_error("no " + key + "?");
    return utils::lexical_cast<Value>(iter->second);
  }
  
  void Model::read(const path_type& path)
  {
    typedef utils::repository repository_type;

    if (path.empty() || ! boost::filesystem::exists(path))
      throw std::runtime_error("no file? " + path.string());
    
    repository_type rep(path, repository_type::read);
    
    hidden_    = repository_value<size_type>(rep, "hidden");
    embedding_ = repository_value<size_type>(rep, "embedding");
    
    if (hidden_ == 0)
      throw std::runtime_error("invalid dimension");
    if (embedding_ == 0)
      throw std::runtime_error("invalid dimension");
    
    inline_read_embedding(rep.path("terminal.txt.gz"), rep.path("terminal.bin"), terminal_, vocab_terminal_, embedding_);
    
    read_label_matrix(rep.path("Wc.txt.gz"),  rep.path("Wc.bin"),  Wc_,  vocab_non_terminal_, 1, hidden_);
    
    read_label_matrix(rep.path("Wsh.txt.gz"), rep.path("Wsh.bin"), Wsh_, vocab_non_terminal_, hidden_, hidden_ + embedding_);
    read_label_matrix(rep.path("Bsh.txt.gz"), rep.path("Bsh.bin"), Bsh_, vocab_non_terminal_, hidden_, 1);
    
    read_label_matrix(rep.path("Wre.txt.gz"), rep.path("Wre.bin"), Wre_, vocab_non_terminal_, hidden_, hidden_ + hidden_);
    read_label_matrix(rep.path("Bre.txt.gz"), rep.path("Bre.bin"), Bre_, vocab_non_terminal_, hidden_, 1);
    
    read_label_matrix(rep.path("Wu.txt.gz"),  rep.path("Wu.bin"),  Wu_,  vocab_non_terminal_, hidden_, hidden_);
    read_label_matrix(rep.path("Bu.txt.gz"),  rep.path("Bu.bin"),  Bu_,  vocab_non_terminal_, hidden_, 1);
    
    Wf_ = tensor_type::Zero(hidden_, hidden_);
    Bf_ = tensor_type::Zero(hidden_, 1);
    
    Wi_ = tensor_type::Zero(hidden_, hidden_);
    Bi_ = tensor_type::Zero(hidden_, 1);
    
    Ba_ = tensor_type::Zero(hidden_, 1);
    
    read_matrix(rep.path("Wf.txt.gz"), rep.path("Wf.bin"), Wf_);
    read_matrix(rep.path("Bf.txt.gz"), rep.path("Bf.bin"), Bf_);
    
    read_matrix(rep.path("Wi.txt.gz"), rep.path("Wi.bin"), Wi_);
    read_matrix(rep.path("Bi.txt.gz"), rep.path("Bi.bin"), Bi_);
    
    read_matrix(rep.path("Ba.txt.gz"), rep.path("Ba.bin"), Ba_);
  }
  
  void Model::read_embedding(const path_type& path)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
    typedef boost::fusion::tuple<std::string, parameter_set_type > embedding_parsed_type;
    typedef boost::spirit::istream_iterator iterator_type;
    
    if (path != "-" && ! boost::filesystem::exists(path))
      throw std::runtime_error("no embedding: " + path.string());
    
    qi::rule<iterator_type, std::string(), standard::blank_type>           word;
    qi::rule<iterator_type, embedding_parsed_type(), standard::blank_type> parser; 
    
    word   %= qi::lexeme[+(standard::char_ - standard::space)];
    parser %= word >> *qi::double_ >> (qi::eol | qi::eoi);
    
    utils::compress_istream is(path, 1024 * 1024);
    is.unsetf(std::ios::skipws);
    
    iterator_type iter(is);
    iterator_type iter_end;
    
    embedding_parsed_type parsed;
    
    while (iter != iter_end) {
      boost::fusion::get<0>(parsed).clear();
      boost::fusion::get<1>(parsed).clear();
      
      if (! boost::spirit::qi::phrase_parse(iter, iter_end, parser, standard::blank, parsed))
	if (iter != iter_end)
	  throw std::runtime_error("embedding parsing failed");
      
      if (boost::fusion::get<1>(parsed).size() != embedding_)
	throw std::runtime_error("invalid embedding size");
      
      const word_type word = boost::fusion::get<0>(parsed);
      
      if (word.id() >= terminal_.cols())
	terminal_.conservativeResize(embedding_, word.id() + 1);
      if (word.id() >= vocab_terminal_.size())
	vocab_terminal_.resize(word.id() + 1, false);
      
      terminal_.col(word.id())
	= Eigen::Map<const tensor_type>(&(*boost::fusion::get<1>(parsed).begin()), embedding_, 1);
      
      vocab_terminal_[word.id()] = true;
    }
  }
  
  template <typename Tensor, typename WordSet>
  inline
  void write_embedding(std::ostream& os, const Tensor& embedding, const WordSet& words)
  {
    typedef Model::size_type size_type;
    typedef Model::word_type word_type;

    const size_type rows      = embedding.rows();
    const size_type cols      = utils::bithack::min(static_cast<size_t>(embedding.cols()), words.size());
    const size_type num_words = std::count(words.begin(), words.end(), true);
    
    os.write((char*) &rows,      sizeof(size_type));
    os.write((char*) &cols,      sizeof(size_type));
    os.write((char*) &num_words, sizeof(size_type));
    
    for (typename word_type::id_type id = 0; id != cols; ++ id) 
      if (words[id]) {
	const word_type word(id);
	const size_type word_size = word.size();
	
	os.write((char*) &word_size, sizeof(size_type));
	os.write((char*) &(*word.begin()), word_size);
	os.write((char*) embedding.col(id).data(), sizeof(typename Tensor::Scalar) * rows);
      }
  }

  template <typename Tensor, typename WordSet>
  inline
  void inline_read_embedding(std::istream& is, Tensor& embedding, WordSet& words)
  {
    typedef Model::size_type size_type;
    typedef Model::word_type word_type;
    
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    buffer_type buffer;
    
    size_type rows      = 0;
    size_type cols      = 0;
    size_type num_words = 0;
    is.read((char*) &rows,      sizeof(size_type));
    is.read((char*) &cols,      sizeof(size_type));
    is.read((char*) &num_words, sizeof(size_type));
    
    if (embedding.rows() != rows)
      embedding.conservativeResize(rows, embedding.cols());
    
    words.clear();
    
    for (size_type i = 0; i != num_words; ++ i) {
      size_type word_size = 0;
      is.read((char*) &word_size, sizeof(size_type));
      
      buffer.resize(word_size);
      is.read((char*) &(*buffer.begin()), word_size);
      
      const word_type word(buffer.begin(), buffer.end());
      
      if (word.id() >= embedding.cols())
	embedding.conservativeResize(rows, word.id() + 1);
      if (word.id() >= words.size())
	words.resize(word.id() + 1, false);
      
      is.read((char*) embedding.col(word.id()).data(), sizeof(typename Tensor::Scalar) * rows);

      words[word.id()] = true;
    }
  }
  
  template <typename Tensor, typename WordSet>
  inline
  void write_label_matrix(std::ostream& os,
			  const Tensor& matrix,
			  const WordSet& words,
			  const Model::size_type rows,
			  const Model::size_type cols)
  {
    typedef Model::size_type size_type;
    typedef Model::word_type word_type;
    
    const size_type num_labels = words.size() - std::count(words.begin(), words.end(), word_type());
    
    os.write((char*) &rows,       sizeof(size_type));
    os.write((char*) &cols,       sizeof(size_type));
    os.write((char*) &num_labels, sizeof(size_type));
    
    const size_type id_max = utils::bithack::min(words.size(), static_cast<size_type>(matrix.rows() / rows));
    for (size_type id = 0; id != id_max; ++ id) 
      if (words[id] != word_type()) {
	const word_type& word = words[id];
	const size_type  word_size = word.size();
	
	os.write((char*) &word_size, sizeof(size_type));
	os.write((char*) &(*word.begin()), word_size);
	
	for (size_type col = 0; col != cols; ++ col)
	  os.write((char*) matrix.block(rows * id, 0, rows, cols).col(col).data(),
		   sizeof(typename Tensor::Scalar) * rows);
      }
  }
  
  template <typename Tensor, typename WordSet>
  inline
  void read_label_matrix(std::istream& is,
			 Tensor& matrix,
			 WordSet& labels,
			 Model::size_type rows_hint,
			 Model::size_type cols_hint)
  {
    typedef Model::size_type size_type;
    typedef Model::word_type word_type;
   
    typedef std::vector<char, std::allocator<char> > buffer_type;
    
    buffer_type buffer;
 
    size_type rows       = 0;
    size_type cols       = 0;
    size_type num_labels = 0;

    is.read((char*) &rows,       sizeof(size_type));
    is.read((char*) &cols,       sizeof(size_type));
    is.read((char*) &num_labels, sizeof(size_type));

    if (rows != rows_hint)
      throw std::runtime_error("invlaid rows for read label-matrix");
    if (cols != cols_hint)
      throw std::runtime_error("invlaid cols for read label-matrix");
    
    for (size_type i = 0; i != num_labels; ++ i) {
      size_type label_size = 0;
      is.read((char*) &label_size, sizeof(size_type));
      
      buffer.resize(label_size);
      is.read((char*) &(*buffer.begin()), label_size);
      
      const word_type label(buffer.begin(), buffer.end());
      const word_type::id_type label_id = label.non_terminal_id();
      
      if (rows * (label_id + 1) > matrix.rows())
	matrix.conservativeResize(rows * (label_id + 1), cols);
      if (label_id >= labels.size())
	labels.resize(label_id + 1, word_type());

      for (size_type col = 0; col != cols; ++ col)
	is.read((char*) matrix.block(rows * label_id, 0, rows, cols).col(col).data(), 
		sizeof(typename Tensor::Scalar) * rows);
      
      labels[label_id] = label;
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

#define MODEL_STREAM_OPERATOR(Op1, Op2, Stream)	\
  Op1(Stream, theta.Wc_,  theta.vocab_non_terminal_, 1, theta.hidden_); \
  \
  Op1(Stream, theta.Wsh_, theta.vocab_non_terminal_, theta.hidden_, theta.hidden_ + theta.embedding_); \
  Op1(Stream, theta.Bsh_, theta.vocab_non_terminal_, theta.hidden_, 1); \
  \
  Op1(Stream, theta.Wre_, theta.vocab_non_terminal_, theta.hidden_, theta.hidden_ + theta.hidden_); \
  Op1(Stream, theta.Bre_, theta.vocab_non_terminal_, theta.hidden_, 1); \
  \
  Op1(Stream, theta.Wu_,  theta.vocab_non_terminal_, theta.hidden_, theta.hidden_); \
  Op1(Stream, theta.Bu_,  theta.vocab_non_terminal_, theta.hidden_, 1); \
  \
  Op2(Stream, theta.Wf_); \
  Op2(Stream, theta.Bf_); \
  \
  Op2(Stream, theta.Wi_); \
  Op2(Stream, theta.Bi_); \
  \
  Op2(Stream, theta.Ba_);

  std::ostream& operator<<(std::ostream& os, const Model& theta)
  {
    os.write((char*) &theta.hidden_,    sizeof(theta.hidden_));
    os.write((char*) &theta.embedding_, sizeof(theta.embedding_));
    
    write_embedding(os, theta.terminal_, theta.vocab_terminal_);
        
    MODEL_STREAM_OPERATOR(write_label_matrix, write_matrix, os);
    
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Model& theta)
  {
    is.read((char*) &theta.hidden_,    sizeof(theta.hidden_));
    is.read((char*) &theta.embedding_, sizeof(theta.embedding_));
    
    inline_read_embedding(is, theta.terminal_, theta.vocab_terminal_);
    
    MODEL_STREAM_OPERATOR(read_label_matrix, read_matrix, is);
    
    return is;
  }

#undef MODEL_STREAM_OPERATOR

  template <typename Tensor>
  inline 
  void plus_equal(Tensor& x, const Tensor& y)
  {
    if (x.rows() == y.rows() && x.cols() == y.cols())
      x += y;
    else if (x.rows() == y.rows()) {
      if (x.cols() > y.cols())
	x.block(0, 0, y.rows(), y.cols()) += y;
      else {
	typedef typename Tensor::Index index_type;
	const index_type cols = x.cols();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(0, cols, x.rows(), y.cols() - cols).setZero();
	
	x += y;
      }
    } else if (x.cols() == y.cols()) {
      if (x.rows() > y.rows())
	x.block(0, 0, y.rows(), y.cols()) += y;
      else {
	typedef typename Tensor::Index index_type;
	const index_type rows = x.rows();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(rows, 0, y.rows() - rows, x.cols()).setZero();
	
	x += y;
      }
    } else {
      // both differ...
      typedef typename Tensor::Index index_type;
      const index_type rows_new = utils::bithack::max(x.rows(), y.rows());
      const index_type cols_new = utils::bithack::max(x.cols(), y.cols());

      Tensor x_new = Tensor::Zero(rows_new, cols_new);
      x_new.block(0, 0, x.rows(), x.cols()) = x;
      x_new.block(0, 0, y.rows(), y.cols()) += y;

      x.swap(x_new);
    }
  }

  template <typename Tensor>
  inline 
  void minus_equal(Tensor& x, const Tensor& y)
  {
    if (x.rows() == y.rows() && x.cols() == y.cols())
      x -= y;
    else if (x.rows() == y.rows()) {
      if (x.cols() > y.cols())
	x.block(0, 0, y.rows(), y.cols()) -= y;
      else {
	typedef typename Tensor::Index index_type;
	const index_type cols = x.cols();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(0, cols, x.rows(), y.cols() - cols).setZero();
	
	x -= y;
      }
    } else if (x.cols() == y.cols()) {
      if (x.rows() > y.rows())
	x.block(0, 0, y.rows(), y.cols()) -= y;
      else {
	typedef typename Tensor::Index index_type;
	const index_type rows = x.rows();
	
	x.conservativeResize(y.rows(), y.cols());
	x.block(rows, 0, y.rows() - rows, x.cols()).setZero();
	
	x -= y;
      }
    } else {
      // both differ...
      typedef typename Tensor::Index index_type;
      const index_type rows_new = utils::bithack::max(x.rows(), y.rows());
      const index_type cols_new = utils::bithack::max(x.cols(), y.cols());

      Tensor x_new = Tensor::Zero(rows_new, cols_new);
      x_new.block(0, 0, x.rows(), x.cols()) = x;
      x_new.block(0, 0, y.rows(), y.cols()) -= y;

      x.swap(x_new);
    }    
  }
  
#define MODEL_BINARY_OPERATOR(Op) \
  Op(terminal_, x.terminal_); \
  \
  Op(Wc_,  x.Wc_); \
  \
  Op(Wsh_, x.Wsh_); \
  Op(Bsh_, x.Bsh_); \
  \
  Op(Wre_, x.Wre_); \
  Op(Bre_, x.Bre_); \
  \
  Op(Wu_,  x.Wu_); \
  Op(Bu_,  x.Bu_); \
  \
  Op(Wf_, x.Wf_); \
  Op(Bf_, x.Bf_); \
  \
  Op(Wi_, x.Wi_); \
  Op(Bi_, x.Bi_); \
  \
  Op(Ba_, x.Ba_);

  Model& Model::operator+=(const Model& x)
  {
    MODEL_BINARY_OPERATOR(plus_equal);

    return *this;
  }
  
  Model& Model::operator-=(const Model& x)
  {

    MODEL_BINARY_OPERATOR(minus_equal);

    return *this;
  }

#undef MODEL_BINARY_OPERATOR
  
#define MODEL_UNARY_OPERATOR(Op) \
    terminal_ Op x; \
    \
    Wc_  Op x; \
    \
    Wsh_ Op x; \
    Bsh_ Op x; \
    \
    Wre_ Op x; \
    Bre_ Op x; \
    \
    Wu_  Op x; \
    Bu_  Op x; \
    \
    Wf_ Op x; \
    Bf_ Op x; \
    \
    Wi_ Op x; \
    Bi_ Op x; \
    \
    Ba_ Op x;
  
  Model& Model::operator*=(const double& x)
  {
    MODEL_UNARY_OPERATOR(*=);

    return *this;
  }
  
  Model& Model::operator/=(const double& x)
  {
    MODEL_UNARY_OPERATOR(/=);
    
    return *this;
  }

#undef MODEL_UNARY_OPERATOR
};
