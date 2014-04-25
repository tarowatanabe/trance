//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <boost/fusion/include/adapt_struct.hpp>

#include <iterator>

#include "tree.hpp"

#include "utils/getline.hpp"


namespace rnnp
{
  namespace impl
  {
    struct treebank_type
    {
      typedef std::vector<treebank_type, std::allocator<treebank_type> > antecedent_type;
      
      std::string     label_;
      antecedent_type antecedent_;

      treebank_type() {}
      treebank_type(const std::string& label) : label_(label) {}
      
      void clear()
      {
	label_.clear();
	antecedent_.clear();
      }

      void transform(rnnp::Tree& tree) const
      {
	// pre-order traversal...

	if (antecedent_.empty()) // terminal
	  tree.label_ = label_;
	else {
	  tree.label_ = '[' + label_ + ']'; // non-terminal
	  tree.antecedent_ = rnnp::Tree::antecedent_type(antecedent_.size());
	  
	  for (size_t i = 0; i != tree.antecedent_.size(); ++ i)
	    antecedent_[i].transform(tree.antecedent_[i]);
	}
      }
    };

  };
};

BOOST_FUSION_ADAPT_STRUCT(
			  rnnp::impl::treebank_type,
			  (std::string, label_)
			  (rnnp::impl::treebank_type::antecedent_type, antecedent_)
			  )

namespace rnnp
{
  namespace impl
  {
    template <typename Iterator>
    struct treebank_grammar : boost::spirit::qi::grammar<Iterator, treebank_type(), boost::spirit::standard::space_type>
    {
      treebank_grammar() : treebank_grammar::base_type(root)
      {
	namespace qi = boost::spirit::qi;
	namespace standard = boost::spirit::standard;
	
	label %= qi::lexeme[+(standard::char_ - standard::space - '(' - ')')];
	treebank %= qi::hold['(' >> label >> +treebank >> ')'] | label;
	root %= (qi::hold['(' >> label >> +treebank >> ')']
		 | qi::hold['(' >> qi::attr("ROOT") >> +treebank >> ')']
		 | qi::lit('(') >> qi::attr("") >> qi::lit('(') >> qi::lit(')') >> qi::lit(')'));
      }
      
      boost::spirit::qi::rule<Iterator, std::string(),   boost::spirit::standard::space_type> label;
      boost::spirit::qi::rule<Iterator, treebank_type(), boost::spirit::standard::space_type> treebank;
      boost::spirit::qi::rule<Iterator, treebank_type(), boost::spirit::standard::space_type> root;
    };
    
  };

  template <typename Iterator>
  bool tree_parser(Iterator& iter, Iterator end, Tree& tree)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    impl::treebank_grammar<Iterator> parser;
    impl::treebank_type              parsed;
    
    tree.clear();
    
    if (qi::phrase_parse(iter, end, parser, standard::space, parsed)) {
      parsed.transform(tree);
      return true;
    } else
      return false;
  }
  
  template <typename Iterator>
  bool tree_generator(Iterator iter, const Tree& tree)
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;

    if (tree.empty())
      return karma::generate(iter, "(())");
    else if (tree.leaf())
      return karma::generate(iter, standard::string, tree.label_);
    else {
      if (! karma::generate(iter, '(' << standard::string << ' ', tree.label_.strip()))
	return false;
      
      for (Tree::const_iterator aiter = tree.begin(); aiter != tree.end(); ++ aiter)
	if (! tree_generator(iter, *aiter))
	  return false;
      
      return karma::generate(iter, ')');
    }
  }

  void Tree::assign(const utils::piece& x)
  {
    utils::piece::const_iterator iter(x.begin());
    utils::piece::const_iterator end(x.end());
    
    const bool result = assign(iter, end);
    
    if (! result || iter != end)
      throw std::runtime_error("tree parsing failed");
  }
  
  bool Tree::assign(std::string::const_iterator& iter, std::string::const_iterator end)
  {
    return tree_parser(iter, end, *this);
  }
  
  bool Tree::assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end)
  {
    return tree_parser(iter, end, *this);
  }

  std::string Tree::string() const
  {
    std::string out;
    
    tree_generator(std::back_inserter(out), *this);

    return out;
  }
  
  std::ostream& operator<<(std::ostream& os, const Tree& tree)
  {
    tree_generator(std::ostream_iterator<char>(os), tree);
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Tree& tree)
  {
    std::string line;
    
    if (utils::getline(is, line))
      tree.assign(line);
    else
      tree.clear();
    
    return is;
  }

};
