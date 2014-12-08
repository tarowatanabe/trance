// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__SIGNATURE__HPP__
#define __TRANCE__SIGNATURE__HPP__ 1

#include <string>

#include <trance/symbol.hpp>

#include <utils/piece.hpp>

#include <boost/shared_ptr.hpp>

namespace trance
{
  class Signature
  {
  public:
    typedef Symbol symbol_type;
    typedef Symbol word_type;

    typedef Signature signature_type;
    typedef boost::shared_ptr<signature_type> signature_ptr_type;
    
  public:
    Signature() {}
    virtual ~Signature() {}
    
  public:
    static signature_ptr_type create(const utils::piece& param);
    static std::string usage();
        
  public:
    virtual
    signature_ptr_type clone() const
    {
      return signature_ptr_type(new Signature());
    }
    
    virtual
    symbol_type operator()(const symbol_type& word) const
    {
      return symbol_type::UNK;
    }
  };
};

#endif
