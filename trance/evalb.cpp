
#include "evalb.hpp"

namespace trance
{
  std::ostream& operator<<(std::ostream& os, const Evalb& evalb)
  {
    os.write((char*) &evalb.match_, sizeof(Evalb::count_type));
    os.write((char*) &evalb.gold_,  sizeof(Evalb::count_type));
    os.write((char*) &evalb.test_,  sizeof(Evalb::count_type));
    return os;
  }
  
  std::istream& operator>>(std::istream& is, Evalb& evalb)
  {
    is.read((char*) &evalb.match_, sizeof(Evalb::count_type));
    is.read((char*) &evalb.gold_,  sizeof(Evalb::count_type));
    is.read((char*) &evalb.test_,  sizeof(Evalb::count_type));
    
    return is;
  }
  
};
