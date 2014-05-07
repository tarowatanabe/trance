// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__SIGNATURE__UNICODE__HPP__
#define __RNNP__SIGNATURE__UNICODE__HPP__ 1

#include <rnnp/signature.hpp>

#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/uscript.h>
#include <unicode/schriter.h>

namespace rnnp
{
  namespace signature
  {
    class Unicode : public rnnp::Signature
    {
    private:
      typedef std::vector<std::string, std::allocator<std::string> > table_type;

      typedef std::vector<bool, std::allocator<bool> > uscript_type;

    public:
      Unicode()
	: script_(USCRIPT_CODE_LIMIT),
	  general_category_(U_CHAR_CATEGORY_COUNT)
      {
	for (int i = 0; i < USCRIPT_CODE_LIMIT; ++ i)
	  script_[i] = uscript_getShortName(static_cast<UScriptCode>(i));
	
	for (int i = 0; i < U_CHAR_CATEGORY_COUNT; ++ i)
	  general_category_[i] = u_getPropertyValueName(UCHAR_GENERAL_CATEGORY, i, U_SHORT_PROPERTY_NAME);
      }
      
    private:
      table_type script_;
      table_type general_category_;
      
    public:
      signature_ptr_type clone() const { return signature_ptr_type(new Unicode()); }
      
      symbol_type operator()(const symbol_type& symbol) const
      {
	const std::string& word = static_cast<const std::string&>(symbol);
	icu::UnicodeString uword = icu::UnicodeString::fromUTF8(icu::StringPiece(word.data(), word.size()));
	
	Unicode& impl = const_cast<Unicode&>(*this);
	
	bool dg = false;
	uint32_t gc = 0;
	uscript_type sc(script_.size(), false);
	
	icu::StringCharacterIterator iter(uword);
	for (iter.setToStart(); iter.hasNext(); /**/) {
	  const UChar32 ch = iter.next32PostInc();

	  dg |= (u_getNumericValue(ch) != U_NO_NUMERIC_VALUE);
	  gc |= u_getIntPropertyValue(ch, UCHAR_GENERAL_CATEGORY_MASK);
	  sc[u_getIntPropertyValue(ch, UCHAR_SCRIPT)] = true;
	}
	
	std::string signature = "<unk";

	for (int i = 1; i < U_CHAR_CATEGORY_COUNT; ++ i)
	  if (gc & U_MASK(i)) {
	    signature += "-";
	    signature += general_category_[i];
	  }
	
	for (int i = 1; i < USCRIPT_CODE_LIMIT; ++ i)
	  if (sc[i]) {
	    signature += "-";
	    signature += script_[i];
	  }
	
	if (dg)
	  signature += "-NUM";
	
	signature += '>';
	
	return signature;
      }
    };
  };
};

#endif
