
#include <iostream>

#include "tree.hpp"
#include "binarize.hpp"
#include "debinarize.hpp"

int main(int argc, char** argv)
{
  trance::Tree tree;
  trance::Tree binarized_right;
  trance::Tree binarized_left;
  trance::Tree debinarized;
  
  while (std::cin >> tree) {
    std::cout << tree << std::endl;
#if 0
    std::cout << "leaf: " << tree.leaf() << std::endl;
    
    trance::binarize_left(tree, binarized_left);
    
    std::cout << binarized_left << std::endl;
    
    trance::debinarize(binarized_left, debinarized);

    if (tree != debinarized)
      std::cerr << "tree is different: " << debinarized << std::endl;

    trance::binarize_right(tree, binarized_right);
    
    std::cout << binarized_right << std::endl;
    
    trance::debinarize(binarized_right, debinarized);

    if (tree != debinarized)
      std::cerr << "tree is different: " << debinarized << std::endl;
#endif
  }
}
