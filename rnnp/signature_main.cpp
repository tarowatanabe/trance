
#include <iostream>
#include <string>

#include "signature.hpp"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << argv[0] << " signature" << std::endl;
    return 1;
  }
  
  rnnp::Signature::signature_ptr_type signature(rnnp::Signature::create(argv[1]));
  
  std::string word;
  
  while (std::cin >> word)
    std::cout << word << " " << signature->operator()(word) << std::endl;
}
