//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#define BOOST_DISABLE_ASSERTS
#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/karma.hpp>

#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iterator>

#include "graphviz.hpp"

namespace trance
{

  struct GraphvizTree
  {
    typedef int id_type;
    typedef Tree tree_type;

    template <typename Iterator>
    void node(Iterator iter, const tree_type& tree)
    {
      id_type id = 0;
      node(iter, tree, id);
    }

    template <typename Iterator>
    void node(Iterator iter, const tree_type& tree, id_type& id)
    {
      namespace karma = boost::spirit::karma;
      namespace standard = boost::spirit::standard;

      karma::generate(iter,
		      " node_" << karma::int_ << " [label=\"" << standard::string << "\"];",
		      id ++, tree.label_);

      tree_type::const_iterator aiter_end = tree.end();
      for (tree_type::const_iterator aiter = tree.begin(); aiter != aiter_end; ++ aiter)
	node(iter, *aiter, id);
    }

    template <typename Iterator>
    void link(Iterator iter, const tree_type& tree)
    {
      id_type id = 0;
      id_type child = 0;
      link(iter, tree, id, child);
    }

    template <typename Iterator>
    void link(Iterator iter, const tree_type& tree, id_type& id, id_type& child)
    {
      namespace karma = boost::spirit::karma;
      namespace standard = boost::spirit::standard;

      const id_type node_id = id ++;

      tree_type::const_iterator aiter_end = tree.end();
      for (tree_type::const_iterator aiter = tree.begin(); aiter != aiter_end; ++ aiter) {
	id_type ant;
	link(iter, *aiter, id, ant);

	karma::generate(iter, " node_" << karma::int_ << " -> node_" << karma::int_ << ';',
			node_id, ant);
      }

      child = node_id;
    }
  };

  std::ostream& Graphviz::operator()(std::ostream& os, const tree_type& tree)
  {
    typedef std::ostream_iterator<char> iterator_type;

    namespace karma = boost::spirit::karma;
    namespace standard = boost::spirit::standard;

    iterator_type iter(os);

    karma::generate(iter, "digraph { rankdir=TB; ordering=in;");

    GraphvizTree graph;

    // draw nodes first...
    graph.node(iter, tree);

    // then, connect
    graph.link(iter, tree);

    karma::generate(iter, "}");

    return os;
  }
};
