
#include <iostream>
#include <iterator>

#include "rule.hpp"
#include "grammar.hpp"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << argv[0] << " grammar-file" << std::endl;
    return 1;
  }
  
  rnnp::Grammar grammar(argv[1]);
  
  rnnp::Symbol symbol;
  while (std::cin >> symbol) {
    if (symbol.non_terminal()) {
      const rnnp::Grammar::rule_set_type& rules = grammar.unary(symbol);
      
      if (rules.empty()) 
	std::cout << "no rule..." << std::endl;
      else
	std::copy(rules.begin(), rules.end(), std::ostream_iterator<rnnp::Rule>(std::cout, "\n"));
    } else {
      const rnnp::Grammar::rule_set_type& rules = grammar.preterminal(symbol);
      
      if (rules.empty()) 
	std::cout << "no rule..." << std::endl;
      else
	std::copy(rules.begin(), rules.end(), std::ostream_iterator<rnnp::Rule>(std::cout, "\n"));
    }
  }
  
}
