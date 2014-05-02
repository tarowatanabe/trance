// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__SIGNATURE__ENGLISH__HPP__
#define __RNNP__SIGNATURE__ENGLISH__HPP__ 1

#include <rnnp/signature.hpp>

#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/schriter.h>

namespace rnnp
{
  namespace signature
  {
    class English : public rnnp::Signature
    {
    public:
      signature_ptr_type clone() const { return signature_ptr_type(new English()); }
      
      symbol_type operator()(const symbol_type& symbol) const
      {
	const std::string& word = static_cast<const std::string&>(symbol);
	icu::UnicodeString uword = icu::UnicodeString::fromUTF8(icu::StringPiece(word.data(), word.size()));
	
	std::string signature = "<unk";
	
	// signature for English, taken from Stanford parser's getSignature5
	int num_caps = 0;
	bool has_digit  = false;
	bool has_dash   = false;
	bool has_lower  = false;
	bool has_punct  = false;
	
	size_t length = 0;
	UChar32 ch0 = 0;
	UChar32 ch_1 = 0;
	UChar32 ch_2 = 0;
	
	icu::StringCharacterIterator iter(uword);
	for (iter.setToStart(); iter.hasNext(); ++ length) {
	  const UChar32 ch = iter.next32PostInc();
	  
	  // keep initial char...
	  if (ch0 == 0)
	    ch0 = ch;
	  
	  ch_2 = ch_1;
	  ch_1 = ch;
	  
	  const int32_t gc = u_getIntPropertyValue(ch, UCHAR_GENERAL_CATEGORY_MASK);
	  
	  has_dash   |= ((gc & U_GC_PD_MASK) != 0);
	  has_punct  |= ((gc & U_GC_P_MASK) != 0);
	  
	  has_digit |= (u_getNumericValue(ch) != U_NO_NUMERIC_VALUE);
	  
	  if (u_isUAlphabetic(ch)) {
	    if (u_isULowercase(ch))
	      has_lower = true;
	    else if (u_istitle(ch)) {
	      has_lower = true;
	      ++ num_caps;
	    } else
	      ++ num_caps;
	  }
	}
	
	// transform into lower...
	uword.toLower();
	ch_2 = (ch_2 ? u_tolower(ch_2) : ch_2);
	ch_1 = (ch_1 ? u_tolower(ch_1) : ch_1);
	
	// we do not check loc...
	if (u_isUUppercase(ch0) || u_istitle(ch0))
	  signature += "-caps";
	else if (! u_isUAlphabetic(ch0) && num_caps)
	  signature += "-caps";
	else if (has_lower)
	  signature += "-lc";
      
	if (has_digit)
	  signature += "-num";
	if (has_dash)
	  signature += "-dash";
	if (has_punct)
	  signature += "-punct";
      
	if (length >= 3 && ch_1 == 's') {
	  if (ch_2 != 's' && ch_2 != 'i' && ch_2 != 'u')
	    signature += "-s";
	} else if (length >= 5 && ! has_dash && ! (has_digit && num_caps > 0)) {
	  if (uword.endsWith("ed"))
	    signature += "-ed";
	  else if (uword.endsWith("ing"))
	    signature += "-ing";
	  else if (uword.endsWith("ion"))
	    signature += "-ion";
	  else if (uword.endsWith("er"))
	    signature += "-er";
	  else if (uword.endsWith("est"))
	    signature += "-est";
	  else if (uword.endsWith("ly"))
	    signature += "-ly";
	  else if (uword.endsWith("ity"))
	    signature += "-ity";
	  else if (uword.endsWith("y"))
	    signature += "-y";
	  else if (uword.endsWith("al"))
	    signature += "-al";
	}
      
	signature += '>';
	
	return signature;
      }
    };
  };
};

#endif
