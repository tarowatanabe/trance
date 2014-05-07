
#include "signature.hpp"
#include "option.hpp"

#include "signature/chinese.hpp"
#include "signature/english.hpp"
#include "signature/unicode.hpp"

namespace rnnp
{
  Signature::signature_ptr_type Signature::create(const utils::piece& param)
  {
    typedef rnnp::Option option_type;
    
    option_type option(param);
    
    if (utils::ipiece(option.name()) == "chinese")
      return signature_ptr_type(new signature::Chinese());
    else if (utils::ipiece(option.name()) == "english")
      return signature_ptr_type(new signature::English());
    else if (utils::ipiece(option.name()) == "unicode")
      return signature_ptr_type(new signature::Unicode());
    else if (utils::ipiece(option.name()) == "none"
	     || utils::ipiece(option.name()) == "unk"
	     || utils::ipiece(option.name()) == "default")
      return signature_ptr_type(new Signature());
    else
      throw std::runtime_error("invalid signature? " + param);
  }
  
  std::string Signature::usage()
  {
    static const char* desc = "\
english: English signature\n\
chinese: Chinese signature\n\
unicode: Unicode property signature\n\
none: use <unk> (default)\n\
";
    return desc;
  }
};
