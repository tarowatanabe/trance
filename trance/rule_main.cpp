
#include <iostream>

#include "rule.hpp"

int main(int argc, char** argv)
{
  trance::Rule rule;
  
  while (std::cin >> rule)
    std::cout << rule << std::endl;
}
