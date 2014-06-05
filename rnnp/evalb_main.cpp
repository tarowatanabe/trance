
#include <iostream>

#include "evalb.hpp"
#include "tree.hpp"

#include "utils/compress_stream.hpp"

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << argv[0] << " [gold file] [test file]" << std::endl;
    return 1;
  }
  
  rnnp::Tree gold;
  rnnp::Tree test;

  rnnp::Evalb       evalb;
  rnnp::EvalbScorer scorer;
  
  utils::compress_istream ig(argv[1]);
  utils::compress_istream it(argv[2]);
  
  for (;;) {
    ig >> gold;
    it >> test;

    if (! ig || ! it) break;
    
    scorer.assign(gold);
    evalb += scorer(test);
  }
  
  if (ig || it)
    throw std::runtime_error("# of trees does not match");

  std::cout << "scor: " << evalb() << " match: " << evalb.match_ << " gold: " << evalb.gold_ << " test: " << evalb.test_<< std::endl;
}
