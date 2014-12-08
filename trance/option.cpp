//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <iterator>

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include "option.hpp"

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <boost/fusion/adapted/std_pair.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <boost/fusion/tuple.hpp>
#include <boost/fusion/adapted.hpp>

#include "utils/c_string_parser.hpp"
#include "utils/c_string_generator.hpp"

namespace trance
{
  
  typedef std::pair<std::string, std::string> value_parsed_type;
  typedef std::vector<value_parsed_type, std::allocator<value_parsed_type> > value_parsed_set_type;
  
  typedef std::pair<std::string, value_parsed_set_type> option_parsed_type;

  template <typename Iterator>
  struct option_parser : boost::spirit::qi::grammar<Iterator, option_parsed_type(), boost::spirit::standard::space_type>
  {
    option_parser() : option_parser::base_type(option)
    {
      namespace qi = boost::spirit::qi;
      namespace standard = boost::spirit::standard;
      
      param %= qi::lexeme[+(standard::char_ - standard::space - ':')];
      key   %= qi::lexeme[+(standard::char_ - standard::space - '=')];
      value %= qi::lexeme[+(standard::char_ - standard::space - ',')];
      key_values %= ((qi::hold[escaped] | key) >> '=' >> (qi::hold[escaped] | value)) % ',';
      option %= (qi::hold[escaped] | param) >> -(':' >> key_values);
    }
    
    typedef boost::spirit::standard::space_type space_type;
    
    utils::c_string_parser<Iterator> escaped;
    
    boost::spirit::qi::rule<Iterator, std::string(), space_type>           param;
    boost::spirit::qi::rule<Iterator, std::string(), space_type>           key;
    boost::spirit::qi::rule<Iterator, std::string(), space_type>           value;
    boost::spirit::qi::rule<Iterator, value_parsed_set_type(), space_type> key_values;
    boost::spirit::qi::rule<Iterator, option_parsed_type(), space_type> option;
  };
  
  void Option::parse(const utils::piece& option)
  {
    typedef utils::piece::const_iterator iter_type;
    typedef option_parser<iter_type> parser_type;
    
    attr_.clear();
    values_.clear();
    
    parser_type parser;

    option_parsed_type parsed;

    iter_type iter     = option.begin();
    iter_type iter_end = option.end();
    
    const bool result = boost::spirit::qi::phrase_parse(iter, iter_end, parser, boost::spirit::standard::space, parsed);
    
    if (! result || iter != iter_end)
      throw std::runtime_error(std::string("option parsing failed: ") + option);
    
    attr_   = parsed.first;
    values_.insert(values_.end(), parsed.second.begin(), parsed.second.end());
  }
  
  std::ostream& operator<<(std::ostream& os, const Option& x)
  {
    typedef std::ostream_iterator<char> iterator_type;

    utils::c_string_generator<iterator_type> escaped;
    boost::spirit::karma::rule<iterator_type, value_parsed_type()> value;
    
    value %= escaped << '=' << escaped;
    
    iterator_type iter(os);
    boost::spirit::karma::generate(iter, escaped, x.attr_);
    if (! x.values_.empty()) {
      os << ':';
      boost::spirit::karma::generate(iter, value % ',', x.values_);
    }
    
    return os;
  }
};
