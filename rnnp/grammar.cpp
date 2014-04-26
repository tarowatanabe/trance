
#include "grammar.hpp"

#include "utils/compact_set.hpp"
#include "utils/unordered_set.hpp"
#include "utils/compress_stream.hpp"
#include "utils/getline.hpp"

#include <boost/algorithm/string/trim.hpp>

namespace rnnp
{
  void Grammar::read(const path_type& path)
  {
    typedef utils::compact_set<symbol_type,
			       utils::unassigned<symbol_type>, utils::unassigned<symbol_type>,
			       boost::hash<symbol_type>, std::equal_to<symbol_type>,
			       std::allocator<symbol_type> > symbol_unique_type;
    
    clear();

    symbol_unique_type terminal;
    symbol_unique_type non_terminal;
    symbol_unique_type pos;
    
    std::string line;
    rule_type rule;
    
    utils::compress_istream is(path, 1024 * 1024);
    
    while (utils::getline(is, line)) {
      boost::algorithm::trim(line);
      
      if (line.empty()) continue;
      
      rule.assign(line);

      if (rule.goal()) {
	if (! goal_.empty())
	  throw std::runtime_error("we do not support multiple goal");
	
	goal_ = rule.lhs_;
	non_terminal.insert(rule.lhs_);
      } else if (rule.unary()) {
	unary_[rule.rhs_[0]].push_back(rule);
	non_terminal.insert(rule.lhs_);
	non_terminal.insert(rule.rhs_.begin(), rule.rhs_.end());
      } else if (rule.binary()) {
	binary_[std::make_pair(rule.rhs_[0], rule.rhs_[1])].push_back(rule);
	non_terminal.insert(rule.lhs_);
	non_terminal.insert(rule.rhs_.begin(), rule.rhs_.end());
      } else if (rule.preterminal()) {
	preterminal_[rule.rhs_[0]].push_back(rule);
	pos.insert(rule.lhs_);
	non_terminal.insert(rule.lhs_);
	terminal.insert(rule.rhs_.front());
      } else
	throw std::runtime_error("invlaid rule: " + rule.string());
    }
    
    if (goal_ == symbol_type())
      throw std::runtime_error("no goal?");
    if (terminal.empty())
      throw std::runtime_error("no terminals?");
    if (non_terminal.empty())
      throw std::runtime_error("no non-terminals?");
    if (pos.empty())
      throw std::runtime_error("no POS?");

    if (terminal.find(symbol_type::UNK) == terminal.end())
      throw std::runtime_error("no fallback preterminal?");
    
    // assign label set
    terminal_.insert(terminal_.end(), terminal.begin(), terminal.end());
    non_terminal_.insert(non_terminal_.end(), non_terminal.begin(), non_terminal.end());
    pos_.insert(pos_.end(), pos.begin(), pos.end());
    
    // check duplicates!
    typedef utils::unordered_set<rule_type,
				 boost::hash<rule_type>, std::equal_to<rule_type>,
				 std::allocator<rule_type> >::type rule_unique_type;

    rule_unique_type uniques;
    
    // binary rules
    rule_set_binary_type::iterator biter_end = binary_.end();
    for (rule_set_binary_type::iterator biter = binary_.begin(); biter != biter_end; ++ biter) {
      uniques.clear();
      uniques.insert(biter->second.begin(), biter->second.end());
      
      biter->second.clear();
      biter->second.insert(biter->second.end(), uniques.begin(), uniques.end());
      rule_set_type(biter->second).swap(biter->second);
    }
    
    // unary rules
    rule_set_unary_type::iterator uiter_end = unary_.end();
    for (rule_set_unary_type::iterator uiter = unary_.begin(); uiter != uiter_end; ++ uiter) {
      uniques.clear();
      uniques.insert(uiter->second.begin(), uiter->second.end());
      
      uiter->second.clear();
      uiter->second.insert(uiter->second.end(), uniques.begin(), uniques.end());
      rule_set_type(uiter->second).swap(uiter->second);
    }

    // preterminal rules
    rule_set_preterminal_type::iterator piter_end = preterminal_.end();
    for (rule_set_preterminal_type::iterator piter = preterminal_.begin(); piter != piter_end; ++ piter) {
      uniques.clear();
      uniques.insert(piter->second.begin(), piter->second.end());
      
      piter->second.clear();
      piter->second.insert(piter->second.end(), uniques.begin(), uniques.end());
      rule_set_type(piter->second).swap(piter->second);
    }
  }

  void Grammar::write(const path_type& path) const
  {
    utils::compress_ostream os(path, 1024 * 1024);

    // goal
    os << rule_type(goal_) << '\n';
    
    // binary rules
    rule_set_binary_type::const_iterator biter_end = binary_.end();
    for (rule_set_binary_type::const_iterator biter = binary_.begin(); biter != biter_end; ++ biter) {
      rule_set_type::const_iterator riter_end = biter->second.end();
      for (rule_set_type::const_iterator riter = biter->second.begin(); riter != riter_end; ++ riter)
	os << *riter << '\n';
    }

    // unary rules
    rule_set_unary_type::const_iterator uiter_end = unary_.end();
    for (rule_set_unary_type::const_iterator uiter = unary_.begin(); uiter != uiter_end; ++ uiter) {
      rule_set_type::const_iterator riter_end = uiter->second.end();
      for (rule_set_type::const_iterator riter = uiter->second.begin(); riter != riter_end; ++ riter)
	os << *riter << '\n';
    } 
    
    // preterminal rules
    rule_set_preterminal_type::const_iterator piter_end = preterminal_.end();
    for (rule_set_preterminal_type::const_iterator piter = preterminal_.begin(); piter != piter_end; ++ piter) {
      rule_set_type::const_iterator riter_end = piter->second.end();
      for (rule_set_type::const_iterator riter = piter->second.begin(); riter != riter_end; ++ riter)
	os << *riter << '\n';
    } 
  }  

  Grammar::size_type Grammar::binary_size() const
  {
    size_type size = 0;
    
    rule_set_binary_type::const_iterator biter_end = binary_.end();
    for (rule_set_binary_type::const_iterator biter = binary_.begin(); biter != biter_end; ++ biter)
      size += biter->second.size();
    
    return size;
  }

  Grammar::size_type Grammar::unary_size() const
  {
    size_type size = 0;
    
    rule_set_unary_type::const_iterator uiter_end = unary_.end();
    for (rule_set_unary_type::const_iterator uiter = unary_.begin(); uiter != uiter_end; ++ uiter)
      size += uiter->second.size();

    return size;
  }

  Grammar::size_type Grammar::preterminal_size() const
  {
    size_type size = 0;
    
    rule_set_preterminal_type::const_iterator piter_end = preterminal_.end();
    for (rule_set_preterminal_type::const_iterator piter = preterminal_.begin(); piter != piter_end; ++ piter)
      size += piter->second.size();

    return size;
  }
  
};
