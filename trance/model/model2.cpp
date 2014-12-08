//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/spirit/include/phoenix_core.hpp>

#include <boost/fusion/tuple.hpp>

#include "model/model2.hpp"

#include "utils/repository.hpp"
#include "utils/compress_stream.hpp"
#include "utils/lexical_cast.hpp"

namespace trance
{
  namespace model
  {
    void Model2::precompute()
    {
      cache_ = tensor_type::Zero(hidden_ * vocab_category_.size(), terminal_.cols());
      
      const size_type id_max = terminal_.cols();
      
      for (size_type id = 0; id != id_max; ++ id)
	for (size_type c = 0; c != vocab_category_.size(); ++ c)
	  cache_.block(c * hidden_, id, hidden_, 1) = Wsh_.block(c * hidden_, hidden_, hidden_, embedding_) * terminal_.col(id);
    }

    void Model2::initialize(const size_type& hidden,
			    const size_type& embedding,
			    const grammar_type& grammar)
    {
      Model::initialize(hidden, embedding, grammar);
    
      // initialize matrix
      cache_.resize(0, 0);
      
      terminal_ = tensor_type::Zero(embedding_, vocab_terminal_.size());
    
      Wc_  = tensor_type::Zero(1 * vocab_category_.size(), hidden_ * 3);
      Bc_  = tensor_type::Zero(1 * vocab_category_.size(), 3);
      Wfe_.clear();
    
      Wsh_ = tensor_type::Zero(hidden_ * vocab_category_.size(), hidden_ + embedding_);
      Bsh_ = tensor_type::Zero(hidden_ * vocab_category_.size(), 1);
    
      Wre_ = tensor_type::Zero(hidden_ * vocab_category_.size(), hidden_ + hidden_);
      Bre_ = tensor_type::Zero(hidden_ * vocab_category_.size(), 1);

      Wu_  = tensor_type::Zero(hidden_ * vocab_category_.size(), hidden_);
      Bu_  = tensor_type::Zero(hidden_ * vocab_category_.size(), 1);
    
      Wf_ = tensor_type::Zero(hidden_, hidden_);
      Bf_ = tensor_type::Zero(hidden_, 1);

      Wi_ = tensor_type::Zero(hidden_, hidden_);
      Bi_ = tensor_type::Zero(hidden_, 1);
    
      Ba_ = tensor_type::Zero(hidden_, 1);
    }
  
  
    void Model2::write(const path_type& path) const
    {
      typedef utils::repository repository_type;
    
      repository_type rep(path, repository_type::write);
    
      rep["model"]     = "model2";
      rep["embedding"] = utils::lexical_cast<std::string>(embedding_);
      rep["hidden"]    = utils::lexical_cast<std::string>(hidden_);
    
      Model::write_embedding(rep.path("terminal.txt.gz"), rep.path("terminal.bin"), terminal_);
    
      Model::write_category(rep.path("Wc.txt.gz"), rep.path("Wc.bin"),  Wc_,  1, hidden_ * 3);
      Model::write_category(rep.path("Bc.txt.gz"), rep.path("Bc.bin"),  Bc_,  1, 3);
      Model::write_weights(rep.path("Wfe.txt.gz"), Wfe_);
    
      Model::write_category(rep.path("Wsh.txt.gz"), rep.path("Wsh.bin"), Wsh_, hidden_, hidden_ + embedding_);
      Model::write_category(rep.path("Bsh.txt.gz"), rep.path("Bsh.bin"), Bsh_, hidden_, 1);
    
      Model::write_category(rep.path("Wre.txt.gz"), rep.path("Wre.bin"), Wre_, hidden_, hidden_ + hidden_);
      Model::write_category(rep.path("Bre.txt.gz"), rep.path("Bre.bin"), Bre_, hidden_, 1);

      Model::write_category(rep.path("Wu.txt.gz"),  rep.path("Wu.bin"),  Wu_, hidden_, hidden_);
      Model::write_category(rep.path("Bu.txt.gz"),  rep.path("Bu.bin"),  Bu_, hidden_, 1);
    
      Model::write_matrix(rep.path("Wf.txt.gz"), rep.path("Wf.bin"), Wf_);
      Model::write_matrix(rep.path("Bf.txt.gz"), rep.path("Bf.bin"), Bf_);
    
      Model::write_matrix(rep.path("Wi.txt.gz"), rep.path("Wi.bin"), Wi_);
      Model::write_matrix(rep.path("Bi.txt.gz"), rep.path("Bi.bin"), Bi_);
    
      Model::write_matrix(rep.path("Ba.txt.gz"), rep.path("Ba.bin"), Ba_);
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
  
    void Model2::read(const path_type& path)
    {
      typedef utils::repository repository_type;

      if (path.empty() || ! boost::filesystem::exists(path))
	throw std::runtime_error("no file? " + path.string());
    
      repository_type rep(path, repository_type::read);

      if (repository_value<std::string>(rep, "model") != "model2")
	throw std::runtime_error("this is not model2!");
    
      hidden_    = repository_value<size_type>(rep, "hidden");
      embedding_ = repository_value<size_type>(rep, "embedding");
    
      if (hidden_ == 0)
	throw std::runtime_error("invalid dimension");
      if (embedding_ == 0)
	throw std::runtime_error("invalid dimension");
    
      vocab_terminal_.clear();
      vocab_category_.clear();
      
      // first, resize
      cache_.resize(0, 0);
      
      terminal_ = tensor_type::Zero(embedding_, terminal_.cols());

      Wc_  = tensor_type::Zero(Wc_.rows(), hidden_ * 3);
      Bc_  = tensor_type::Zero(Bc_.rows(), 3);
      Wfe_.clear();
    
      Wsh_ = tensor_type::Zero(Wsh_.rows(), hidden_ + embedding_);
      Bsh_ = tensor_type::Zero(Bsh_.rows(), 1);
    
      Wre_ = tensor_type::Zero(Wre_.rows(), hidden_ + hidden_);
      Bre_ = tensor_type::Zero(Bre_.rows(), 1);

      Wu_  = tensor_type::Zero(Wu_.rows(), hidden_);
      Bu_  = tensor_type::Zero(Bu_.rows(), 1);

      Wf_ = tensor_type::Zero(hidden_, hidden_);
      Bf_ = tensor_type::Zero(hidden_, 1);
    
      Wi_ = tensor_type::Zero(hidden_, hidden_);
      Bi_ = tensor_type::Zero(hidden_, 1);
    
      Ba_ = tensor_type::Zero(hidden_, 1);
    
      // then, read!
      Model::read_embedding(rep.path("terminal.txt.gz"), rep.path("terminal.bin"), terminal_);
    
      Model::read_category(rep.path("Wc.txt.gz"), rep.path("Wc.bin"),  Wc_,  1, hidden_ * 3);
      Model::read_category(rep.path("Bc.txt.gz"), rep.path("Bc.bin"),  Bc_,  1, 3);
      Model::read_weights(rep.path("Wfe.txt.gz"), Wfe_);
    
      Model::read_category(rep.path("Wsh.txt.gz"), rep.path("Wsh.bin"), Wsh_, hidden_, hidden_ + embedding_);
      Model::read_category(rep.path("Bsh.txt.gz"), rep.path("Bsh.bin"), Bsh_, hidden_, 1);
    
      Model::read_category(rep.path("Wre.txt.gz"), rep.path("Wre.bin"), Wre_, hidden_, hidden_ + hidden_);
      Model::read_category(rep.path("Bre.txt.gz"), rep.path("Bre.bin"), Bre_, hidden_, 1);

      Model::read_category(rep.path("Wu.txt.gz"),  rep.path("Wu.bin"),  Wu_, hidden_, hidden_);
      Model::read_category(rep.path("Bu.txt.gz"),  rep.path("Bu.bin"),  Bu_, hidden_, 1);
    
      Model::read_matrix(rep.path("Wf.txt.gz"), rep.path("Wf.bin"), Wf_);
      Model::read_matrix(rep.path("Bf.txt.gz"), rep.path("Bf.bin"), Bf_);
    
      Model::read_matrix(rep.path("Wi.txt.gz"), rep.path("Wi.bin"), Wi_);
      Model::read_matrix(rep.path("Bi.txt.gz"), rep.path("Bi.bin"), Bi_);
    
      Model::read_matrix(rep.path("Ba.txt.gz"), rep.path("Ba.bin"), Ba_);
    }
  
    void Model2::embedding(const path_type& path)
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
    
#define MODEL_STREAM_OPERATOR(Theta, OpEmbedding, OpCategory, OpWeights, OpMatrix, Stream) \
    Theta.OpEmbedding(Stream, Theta.terminal_);				\
									\
    Theta.OpCategory(Stream, Theta.Wc_,  1, Theta.hidden_ * 3);		\
    Theta.OpCategory(Stream, Theta.Bc_,  1, 3);				\
    Theta.OpWeights(Stream,  Theta.Wfe_);				\
									\
    Theta.OpCategory(Stream, Theta.Wsh_, Theta.hidden_, Theta.hidden_ + Theta.embedding_); \
    Theta.OpCategory(Stream, Theta.Bsh_, Theta.hidden_, 1);		\
									\
    Theta.OpCategory(Stream, Theta.Wre_, Theta.hidden_, Theta.hidden_ + Theta.hidden_); \
    Theta.OpCategory(Stream, Theta.Bre_, Theta.hidden_, 1);		\
									\
    Theta.OpCategory(Stream, Theta.Wu_,  Theta.hidden_, Theta.hidden_); \
    Theta.OpCategory(Stream, Theta.Bu_,  Theta.hidden_, 1);		\
									\
    Theta.OpMatrix(Stream, Theta.Wf_);					\
    Theta.OpMatrix(Stream, Theta.Bf_);					\
									\
    Theta.OpMatrix(Stream, Theta.Wi_);					\
    Theta.OpMatrix(Stream, Theta.Bi_);					\
									\
    Theta.OpMatrix(Stream, Theta.Ba_);
    
    std::ostream& operator<<(std::ostream& os, const Model2& theta)
    {
      os.write((char*) &theta.hidden_,    sizeof(theta.hidden_));
      os.write((char*) &theta.embedding_, sizeof(theta.embedding_));
      
      MODEL_STREAM_OPERATOR(theta, write_embedding, write_category, write_weights, write_matrix, os);
      
      return os;
    }
    
    std::istream& operator>>(std::istream& is, Model2& theta)
    {
      is.read((char*) &theta.hidden_,    sizeof(theta.hidden_));
      is.read((char*) &theta.embedding_, sizeof(theta.embedding_));
      
      MODEL_STREAM_OPERATOR(theta, read_embedding, read_category, read_weights, read_matrix, is);

      theta.cache_.resize(0, 0);
      
      return is;
    }
    
#undef MODEL_STREAM_OPERATOR
  
#define MODEL_BINARY_OPERATOR(Op, Theta)	\
    Op(terminal_, Theta.terminal_);		\
						\
    Op(Wc_,  Theta.Wc_);			\
    Op(Bc_,  Theta.Bc_);			\
    Op(Wfe_, Theta.Wfe_);			\
						\
    Op(Wsh_, Theta.Wsh_);			\
    Op(Bsh_, Theta.Bsh_);			\
						\
    Op(Wre_, Theta.Wre_);			\
    Op(Bre_, Theta.Bre_);			\
						\
    Op(Wu_,  Theta.Wu_);			\
    Op(Bu_,  Theta.Bu_);			\
						\
    Op(Wf_, Theta.Wf_);				\
    Op(Bf_, Theta.Bf_);				\
						\
    Op(Wi_, Theta.Wi_);				\
    Op(Bi_, Theta.Bi_);				\
						\
    Op(Ba_, Theta.Ba_);

    Model2& Model2::operator+=(const Model2& theta)
    {
      MODEL_BINARY_OPERATOR(Model::plus_equal, theta);

      cache_.resize(0, 0);
      
      return *this;
    }
    
    Model2& Model2::operator-=(const Model2& theta)
    {
      MODEL_BINARY_OPERATOR(Model::minus_equal, theta);

      cache_.resize(0, 0);

      return *this;
    }

#undef MODEL_BINARY_OPERATOR
  
#define MODEL_UNARY_OPERATOR(Op)		\
    terminal_ Op x;				\
						\
    Wc_  Op x;					\
    Bc_  Op x;					\
    Wfe_ Op x;					\
						\
    Wsh_ Op x;					\
    Bsh_ Op x;					\
						\
    Wre_ Op x;					\
    Bre_ Op x;					\
						\
    Wu_  Op x;					\
    Bu_  Op x;					\
						\
    Wf_ Op x;					\
    Bf_ Op x;					\
						\
    Wi_ Op x;					\
    Bi_ Op x;					\
						\
    Ba_ Op x;
  
    Model2& Model2::operator*=(const double& x)
    {
      MODEL_UNARY_OPERATOR(*=);

      cache_.resize(0, 0);

      return *this;
    }
  
    Model2& Model2::operator/=(const double& x)
    {
      MODEL_UNARY_OPERATOR(/=);

      cache_.resize(0, 0);
    
      return *this;
    }

#undef MODEL_UNARY_OPERATOR
  };
};
