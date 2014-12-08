//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <iterator>

#include "rule.hpp"

#include "utils/getline.hpp"

namespace trance
{
  void Rule::assign(const utils::piece& x)
  {
    utils::piece::const_iterator iter(x.begin());
    utils::piece::const_iterator end(x.end());
    
    const bool result = assign(iter, end);
    
    if (! result || iter != end)
      throw std::runtime_error("rule parsing failed: " + x);
  }
  
  template <typename Iterator>
  inline
  bool rule_parser(Iterator& iter, Iterator end, Rule& rule)
  {
    typedef std::vector<std::string, std::allocator<std::string> > phrase_type;

    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    qi::rule<Iterator, std::string(), standard::space_type> label;
    qi::rule<Iterator, std::string(), standard::space_type> terminal;
    
    label    %= qi::lexeme[standard::char_('[') >> +(standard::char_ - standard::space - ']') >> standard::char_(']')];
    terminal %= qi::lexeme[+(standard::char_ - standard::space)];
    
    std::string lhs;
    phrase_type rhs;
    
    const bool result = qi::phrase_parse(iter, end, label >> -(qi::omit["->"] >> +terminal), standard::space, lhs, rhs);
    
    if (result) {
      rule.lhs_ = lhs;
      rule.rhs_.assign(rhs.begin(), rhs.end());
    } else {
      rule.lhs_ = Rule::lhs_type();
      rule.rhs_ = Rule::rhs_type();
    }
    
    return result;
  }
  
  bool Rule::assign(std::string::const_iterator& iter, std::string::const_iterator end)
  {
    return rule_parser(iter, end, *this);
  }
  
  bool Rule::assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end)
  {
    return rule_parser(iter, end, *this);
  }

  std::string Rule::string() const
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    if (rhs_.empty())
      return lhs_;
    else {
      std::string out;
      
      karma::generate(std::back_inserter(out),
		      standard::string << " -> " << (standard::string % ' '),
		      lhs_,
		      rhs_);
      
      return out;
    }
  }

  std::ostream& operator<<(std::ostream& os, const Rule& rule)
  {
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    if (rule.rhs_.empty())
      karma::generate(std::ostream_iterator<char>(os), standard::string, rule.lhs_);
    else
      karma::generate(std::ostream_iterator<char>(os),
		      standard::string << " -> " << (standard::string % ' '),
		      rule.lhs_,
		      rule.rhs_);
    
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Rule& rule)
  {
    std::string line;
    
    if (utils::getline(is, line))
      rule.assign(line);
    else {
      rule.lhs_ = Rule::lhs_type();
      rule.rhs_ = Rule::rhs_type();
    }
    
    return is;
  }
  
};
