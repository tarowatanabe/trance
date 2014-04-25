// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__GRAMMAR__HPP__
#define __RNNP__GRAMMAR__HPP__ 1

#include <vector>

#include <rnnp/symbol.hpp>
#include <rnnp/rule.hpp>

#include <utils/bithack.hpp>
#include <utils/unordered_map.hpp>
#include <utils/alloc_vector.hpp>

#include <boost/filesystem/path.hpp>

namespace rnnp
{
  class Grammar
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    
    typedef Symbol symbol_type;
    typedef Symbol word_type;
    typedef Rule   rule_type;
    
    typedef boost::filesystem::path path_type;
    
  public:
    typedef std::vector<symbol_type, std::allocator<symbol_type> > label_set_type;
    typedef std::vector<rule_type, std::allocator<rule_type> > rule_set_type;
    
    typedef std::pair<symbol_type, symbol_type> symbol_pair_type;
    typedef utils::unordered_map<symbol_pair_type, rule_set_type,
				 utils::hashmurmur3<size_t>, std::equal_to<symbol_pair_type>,
				 std::allocator<std::pair<const symbol_pair_type, rule_set_type> > >::type rule_set_binary_type;
    
    typedef utils::unordered_map<symbol_type, rule_set_type,
				 boost::hash<symbol_type>, std::equal_to<symbol_type>,
				 std::allocator<std::pair<const symbol_type, rule_set_type> > >::type rule_set_unary_type;
    typedef utils::unordered_map<word_type, rule_set_type,
				 boost::hash<word_type>, std::equal_to<word_type>,
				 std::allocator<std::pair<const word_type, rule_set_type> > >::type rule_set_preterminal_type;
    
  public:
    void open(const path_type& path) { read(path); }
    void read(const path_type& path);
    void write(const path_type& path) const;

    void clear()
    {
      goal_ = symbol_type();
      
      binary_.clear();
      unary_.clear();
      preterminal_.clear();
      
      terminal_.clear();
      non_terminal_.clear();
      pos_.clear();
    }

    // given goal_, binary_, unary_ and preterminal_, compute terminal_/non_terminal_/pos_
    void initialize();
    
  public:
    // goal
    symbol_type goal_;
    
    // rule set
    rule_set_binary_type      binary_;
    rule_set_unary_type       unary_;
    rule_set_preterminal_type preterminal_;

    // a set of labels...
    label_set_type terminal_;
    label_set_type non_terminal_;
    label_set_type pos_;
  };
};

#endif
