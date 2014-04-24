// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__LOSS__HPP__
#define __RNNP__LOSS__HPP__ 1

#include <stdint.h>

namespace rnnp
{

  struct Loss
  {
    typedef double   value_type;
    typedef uint64_t count_type;

    Loss() : average_(0), count_(0) {}
    Loss(const value_type& x) : average_(x), count_(1) {}
    
    Loss& operator+=(const value_type& x)
    {
      average_ += (x - average_) / (++ count_);
      return *this;
    }
    
    Loss& operator+=(const Loss& x)
    {
      if (! count_) {
	average_ = x.average_;
	count_   = x.count_;
      } else if (x.count_) {
	const count_type total = count_ + x.count_;
	
	average_ = average_ * (value_type(count_) / total) + x.average_ * (value_type(x.count_) / total);
	count_ = total;
      }
      
      return *this;
    }
    
    operator const value_type&() const { return average_; }
    
  private:
    value_type average_;
    count_type count_;
  };
};

#endif
