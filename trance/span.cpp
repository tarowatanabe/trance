//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/range.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>

#include <boost/fusion/adapted/std_pair.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include "span.hpp"

BOOST_FUSION_ADAPT_STRUCT(
			  trance::Span,
			  (trance::Span::index_type, first_)
			  (trance::Span::index_type, last_)
			  )

namespace trance
{
  void Span::assign(const utils::piece& x)
  {
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    utils::piece::const_iterator iter(x.begin());
    utils::piece::const_iterator end(x.end());
    
    const bool result = qi::phrase_parse(iter, end,
					 qi::lexeme[qi::int_ >> ".." >> qi::int_],
					 standard::space, *this);
    if (! result || iter != end)
      throw std::runtime_error("invalid span format? " + x);
  }

  std::istream& operator>>(std::istream& is, Span& x)
  {
    std::string span;

    if (is >> span)
      x.assign(span);
    else
      x = Span();

    return is;
  }
  
  std::ostream& operator<<(std::ostream& os, const Span& x)
  {
    typedef std::ostream_iterator<char> iterator_type;
    
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    iterator_type iter(os);
    
    karma::int_generator<Span::index_type> int_;

    if (! karma::generate(iter, int_ << ".." << int_, x.first_, x.last_))
      throw std::runtime_error("span generation failed...?");
    
    return os;
  }
}
