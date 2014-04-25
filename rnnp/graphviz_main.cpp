
#include <iostream>

#include "tree.hpp"
#include "graphviz.hpp"

int main(int argc, char** argv)
{
  rnnp::Tree tree;
  
  while (std::cin >> tree) 
    rnnp::graphviz(std::cout, tree) << std::endl;
}
