// -*- mode: c++ -*-
// -*- encoding: utf-8 -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__SIGNATURE__CHINESE__HPP__
#define __TRANCE__SIGNATURE__CHINESE__HPP__ 1

#include <trance/signature.hpp>

#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/regex.h>

namespace trance
{
  namespace signature
  {
    class Chinese : public trance::Signature
    {
    public:
      struct Matcher
      {
	Matcher(const char* pattern) : matcher_(0) { initialize(pattern); }
	~Matcher() { std::auto_ptr<icu::RegexMatcher> tmp(matcher_); }
	
	bool operator()(const icu::UnicodeString& x)
	{
	  matcher_->reset(x);
	  
	  UErrorCode status = U_ZERO_ERROR;
	  const bool result = matcher_->matches(status);
	  if (U_FAILURE(status))
	    throw std::runtime_error(std::string("RegexMatcher::matches(): ") + u_errorName(status));
	  
	  return result;
	}
	
      private:
	void initialize(const char* pattern)
	{
	  UErrorCode status = U_ZERO_ERROR;
	  matcher_ = new icu::RegexMatcher(icu::UnicodeString::fromUTF8(pattern), 0, status);
	  if (U_FAILURE(status))
	    throw std::runtime_error(std::string("RegexMatcher: ") + u_errorName(status));
	}
      private:
	icu::RegexMatcher* matcher_;
      };
      
      typedef Matcher matcher_type;
      
    public:
      Chinese()
	: number_match_(".*[[:^Numeric_Type=None:]〇○◯].*"),
	  date_match_(".*[[:^Numeric_Type=None:]〇○◯].*[年月日号]"),
	  ordinal_match_("第.*"),
	  proper_name_match_(".*[··•․‧∙⋅・].*"),
	  punct_match_(".*[[:P:]].*"),
	  symbol_match_(".*[[:S:]].*"),
	  latin_match_(".*[[:Lu:][:Lt:][:Ll:]].*")
      {}
      
    private:
      matcher_type number_match_;
      matcher_type date_match_;
      matcher_type ordinal_match_;
      matcher_type proper_name_match_;
      matcher_type punct_match_;
      matcher_type symbol_match_;
      matcher_type latin_match_;
      
    public:
      signature_ptr_type clone() const { return signature_ptr_type(new Chinese()); }

      symbol_type operator()(const symbol_type& symbol) const
      {
	const std::string& word = static_cast<const std::string&>(symbol);
	icu::UnicodeString uword = icu::UnicodeString::fromUTF8(icu::StringPiece(word.data(), word.size()));
	
	Chinese& impl = const_cast<Chinese&>(*this);
	
	std::string signature = "<unk";
	
	if (impl.date_match_(uword))
	  signature += "-date";
	else if (impl.number_match_(uword)) {
	  signature += "-num";
	  if (impl.ordinal_match_(uword))
	    signature += "-ord";
	}
	
	if (impl.proper_name_match_(uword))
	  signature += "-prop";
	if (impl.punct_match_(uword))
	  signature += "-punct";
	if (impl.symbol_match_(uword))
	  signature += "-symbol";
	if (impl.latin_match_(uword))
	  signature += "-lat";
	
	signature += '>';
	
	return signature;
      }
    };
  };
};

#endif
