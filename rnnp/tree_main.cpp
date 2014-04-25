
#include <iostream>

#include "tree.hpp"
#include "binarize.hpp"
#include "debinarize.hpp"

int main(int argc, char** argv)
{
  rnnp::Tree tree;
  rnnp::Tree binarized_right;
  rnnp::Tree binarized_left;
  rnnp::Tree debinarized;
  
  while (std::cin >> tree) {
    std::cout << "tree: " << tree << std::endl;
    
    rnnp::binarize_left(tree, binarized_left);
    
    std::cout << "binarized-left: " << binarized_left << std::endl;
    
    rnnp::debinarize(binarized_left, debinarized);

    if (tree != debinarized)
      std::cerr << "tree is different: " << debinarized << std::endl;

    rnnp::binarize_right(tree, binarized_right);
    
    std::cout << "binarized-right: " << binarized_right << std::endl;
    
    rnnp::debinarize(binarized_right, debinarized);

    if (tree != debinarized)
      std::cerr << "tree is different: " << debinarized << std::endl;
  }
}
