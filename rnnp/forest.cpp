//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_SPIRIT_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_container.hpp>
#include <boost/spirit/include/phoenix_statement.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>


#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/adapted.hpp>

#include <boost/algorithm/string/trim.hpp>

#include <iterator>

#include "forest.hpp"
#include "sort.hpp"

#include "utils/getline.hpp"

BOOST_FUSION_ADAPT_STRUCT(
			  rnnp::Rule,
			  (rnnp::Rule::lhs_type, lhs_)
			  (rnnp::Rule::rhs_type, rhs_)
			  )

namespace rnnp
{
  
  inline
  void assign_tree(const Forest::id_type node,
		   const Tree& tree,
		   Forest& forest)
  {
    typedef Forest::id_type   id_type;
    typedef Forest::rule_type rule_type;
    typedef Forest::edge_type edge_type;
    typedef Forest::node_type node_type;

    typedef Tree tree_type;

    typedef std::vector<id_type, std::allocator<id_type> > tail_type;

    if (tree.antecedent_.empty()) return;
    
    rule_type rule(tree.label_);
    tail_type tail;
    
    tree_type::const_iterator aiter_end = tree.end();
    for (tree_type::const_iterator aiter = tree.begin(); aiter != aiter_end; ++ aiter) {
      rule.rhs_.push_back(aiter->label_);
      
      if (! aiter->terminal())
	tail.push_back(forest.add_node().id_);
    }
    
    edge_type& edge = forest.add_edge(tail.begin(), tail.end());
    edge.rule_ = rule;
    forest.connect_edge(edge.id_, node);
    
    tail_type::const_iterator titer = tail.begin();
    for (tree_type::const_iterator aiter = tree.begin(); aiter != aiter_end; ++ aiter) 
      if (! aiter->terminal()) {
	assign_tree(*titer, *aiter, forest);
	++ titer;
      }
  }
  
  void Forest::assign(const tree_type& tree)
  {
    clear();
    
    goal_ = add_node().id_;
    
    assign_tree(goal_, tree, *this);
    
    topologically_sort(*this);
  }
  
  template <typename Iterator>
  struct rule_parser : boost::spirit::qi::grammar<Iterator, Forest::rule_type(), boost::spirit::standard::space_type>
  {
    typedef Forest::rule_type     rule_type;
    
    typedef rule_type::lhs_type   lhs_type;
    typedef rule_type::rhs_type   rhs_type;
    
    struct push_escaped_func
    {
      template<class, class>
       struct result {
	typedef void type;
      };
      
      void operator()(std::string& result, const uint32_t c) const
      {
	switch (c) {
	case 'b':  result += '\b'; break;
	case 't':  result += '\t'; break;
	case 'n':  result += '\n'; break;
	case 'f':  result += '\f'; break;
	case 'r':  result += '\r'; break;
	case '\"': result += '\"'; break;
	case '\\': result += '\\'; break;
	case '/':  result += '/'; break;
	}
      }
    };
    
    rule_parser() : rule_parser::base_type(rule)
    {
      namespace qi = boost::spirit::qi;
      namespace standard = boost::spirit::standard;
      
      escaped = '\\' >> standard::char_("btnfr\\\"/") [push_escaped(qi::_r1, qi::_1)];
      
      lhs = +(escaped(qi::_val) | (standard::char_ - standard::space - '\"')[qi::_val += qi::_1]);
      
      rhs %= lhs % (+standard::space);
      
      rule %= lhs >> " ||| " >> rhs;
    }
    
    typedef boost::spirit::standard::space_type space_type;
    
    boost::phoenix::function<push_escaped_func> const push_escaped;
    
    boost::spirit::qi::rule<Iterator, void(std::string&)> escaped;
    
    boost::spirit::qi::rule<Iterator, std::string()>           lhs;
    boost::spirit::qi::rule<Iterator, rhs_type(), space_type>  rhs;
    boost::spirit::qi::rule<Iterator, rule_type(), space_type> rule;
  };
  
  template <typename Iterator>
  bool forest_parser(Iterator& iter, Iterator end, Forest& forest)
  {
    typedef Forest::rule_type     rule_type;
    
    typedef Forest::node_type node_type;
    typedef Forest::edge_type edge_type;
    
    typedef Forest::node_set_type node_set_type;
    typedef Forest::edge_set_type edge_set_type;
    
    namespace qi = boost::spirit::qi;
    namespace standard = boost::spirit::standard;
    
    throw std::runtime_error("parsing is not supported!");
    
    return true;
  }
  
  template <typename Iterator>
  struct rule_generator : boost::spirit::karma::grammar<Iterator, Forest::rule_type()>
  {
    typedef Forest::rule_type     rule_type;
    
    typedef rule_type::lhs_type   lhs_type;
    typedef rule_type::rhs_type   rhs_type;
        
    rule_generator() : rule_generator::base_type(rule)
    {
      namespace karma = boost::spirit::karma;
      namespace standard = boost::spirit::standard;
      
      lhs %= *(&standard::char_('\b') << "\\b"
	       | &standard::char_('\t') << "\\t"
	       | &standard::char_('\n') << "\\n"
	       | &standard::char_('\f') << "\\f"
	       | &standard::char_('\r') << "\\r"
	       | &standard::char_('\"') << "\\\""
	       | &standard::char_('\\') << "\\\\"
	       | &standard::char_('/') << "\\/"
	       | standard::char_);
      
      rhs %= -(lhs % ' ');
      
      rule %= lhs << " ||| " << rhs;
    }
    
    boost::spirit::karma::rule<Iterator, lhs_type()>  lhs;
    boost::spirit::karma::rule<Iterator, rhs_type()>  rhs;
    boost::spirit::karma::rule<Iterator, rule_type()> rule;
  };

  template <typename Float>
  struct real_precision10 : boost::spirit::karma::real_policies<Float>
  {
    static unsigned int precision(Float) 
    { 
      return 10;
    }
  };
  
  template <typename Iterator>
  bool forest_generator(Iterator iter, const Forest& forest)
  {
    typedef Forest::node_type node_type;
    typedef Forest::edge_type edge_type;
    
    typedef Forest::node_set_type node_set_type;
    typedef Forest::edge_set_type edge_set_type;
    
    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;
    
    rule_generator<Iterator> rule;
    karma::real_generator<double, real_precision10<double> > double10;
    
    karma::generate(iter, '{');
    
    // First, output rule part
    karma::generate(iter, karma::lit("\"rules\": ["));
    
    bool initial_edge = true;
    edge_set_type::const_iterator eiter_end = forest.edges_.end();
    for (edge_set_type::const_iterator eiter = forest.edges_.begin(); eiter != eiter_end; ++ eiter) {
      if (! initial_edge)
	karma::generate(iter, karma::lit(", "));
      initial_edge = false;
      
      karma::generate(iter, '\"' << rule << '\"', eiter->rule_);
    }
    
    karma::generate(iter, ']');
    
    // Second, output nodes
    karma::generate(iter, karma::lit(", \"nodes\": ["));

    bool initial_node = true;
    node_set_type::const_iterator niter_end = forest.nodes_.end();
    for (node_set_type::const_iterator niter = forest.nodes_.begin(); niter != niter_end; ++ niter) {
      if (! initial_node)
	karma::generate(iter, karma::lit(", "));
      initial_node = false;
      
      karma::generate(iter, '[');

      bool initial_edge = true;
      node_type::edge_set_type::const_iterator eiter_end = niter->edges_.end();
      for (node_type::edge_set_type::const_iterator eiter = niter->edges_.begin(); eiter != eiter_end; ++ eiter) {
	if (! initial_edge)
	  karma::generate(iter, karma::lit(", "));
	initial_edge = false;
	
	const edge_type& edge = forest.edges_[*eiter];
	
	karma::generate(iter, '{');
	
	// tail
	if (! edge.tail_.empty())
	  karma::generate(iter, karma::lit("\"tail\":[") << (karma::uint_generator<Forest::id_type>() % ',') << "],", edge.tail_);
	
	// score
	karma::generate(iter, karma::lit("\"feature\":{") << karma::lit("\"score\":") << double10 << "},", edge.score_);
	
	// rule
	karma::generate(iter, karma::lit("\"rule\":") << karma::uint_generator<Forest::id_type>(), edge.id_ + 1);
	
	karma::generate(iter, '}');
      }
      
      karma::generate(iter, ']');
    }
    
    karma::generate(iter, ']');
    
    // Third, output goal
    if (forest.valid())
      karma::generate(iter, karma::lit(", \"goal\": ") << karma::uint_generator<Forest::id_type>(), forest.goal_);
    
    karma::generate(iter, '}');
    
    return true;
  }

  void Forest::assign(const utils::piece& x)
  {
    utils::piece::const_iterator iter(x.begin());
    utils::piece::const_iterator end(x.end());
    
    const bool result = assign(iter, end);
    
    if (! result || iter != end)
      throw std::runtime_error("forest parsing failed:" + std::string(x.begin(), std::min(x.begin() + 64, x.end())));
  }
  
  bool Forest::assign(std::string::const_iterator& iter, std::string::const_iterator end)
  {
    return forest_parser(iter, end, *this);
  }
  
  bool Forest::assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end)
  {
    return forest_parser(iter, end, *this);
  }
  
  std::ostream& operator<<(std::ostream& os, const Forest& forest)
  {
    forest_generator(std::ostream_iterator<char>(os), forest);
    
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Forest& forest)
  {
    std::string line;
    
    if (utils::getline(is, line))
      forest.assign(line);
    else
      forest.clear();
    
    return is;    
  }

};
