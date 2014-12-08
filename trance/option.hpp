// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__OPTION__HPP__
#define __TRANCE__OPTION__HPP__ 1

#include <string>
#include <vector>
#include <iostream>

#include <utils/piece.hpp>

namespace trance
{
  struct Option
  {
  public:
    typedef std::string attribute_type;
    typedef std::string key_type;
    typedef std::string data_type;
    typedef std::string mapped_type;
    typedef std::pair<std::string, std::string> value_type;

  private:
    typedef std::vector<value_type, std::allocator<value_type> > value_set_type;      

  public:
    typedef value_set_type::size_type       size_type;
    typedef value_set_type::difference_type difference_type;
    
    typedef value_set_type::const_iterator       iterator;
    typedef value_set_type::const_iterator const_iterator;
      
  public:
    Option(const utils::piece& option)
      : attr_(), values_()  { parse(option); }
    Option() : attr_(), values_() {}
    
    operator attribute_type() const { return attr_; }
    
    const attribute_type& name() const { return attr_; }
    attribute_type& name() { return attr_; }
    
    void push_back(const value_type& x) { values_.push_back(x); }
    
    const_iterator begin() const { return values_.begin(); }
    const_iterator end() const { return values_.end(); }
    
    bool empty() const { return values_.empty(); }
    size_type size() const { return values_.size(); }
    
    const_iterator find(const utils::piece& key) const
    {
      for (const_iterator iter = begin(); iter != end(); ++ iter)
	if (utils::piece(iter->first) == key)
	  return iter;
      return end();
    }

    void erase(const utils::piece& key) 
    {
      while (! values_.empty()) {
	bool found = false;
	for (value_set_type::iterator iter = values_.begin(); iter != values_.end(); ++ iter)
	  if (utils::piece(iter->first) == key) {
	    values_.erase(iter);
	    found = true;
	    break;
	  }
	
	if (! found) break;
      }
    }
    
  public:
    friend
    std::ostream& operator<<(std::ostream& os, const Option& x);
      
  private:
    void parse(const utils::piece& option);
      
  private:
    attribute_type attr_;
    value_set_type values_;
  };
  
};


#endif
