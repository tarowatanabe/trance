// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__SEMIRING__LOG__HPP__
#define __RNNP__SEMIRING__LOG__HPP__ 1

#include <cmath>
#include <cfloat>
#include <climits>

#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <rnnp/semiring/traits.hpp>

#include <utils/mathop.hpp>

namespace rnnp
{
  namespace semiring
  {

    template <typename Tp>
    class Log
    {
    public:
      typedef Tp weight_type;
      typedef Tp value_type;
      typedef Log<Tp> self_type;
      
    public:
      struct proxy_type
      {
	friend class Log;
	
	proxy_type(const weight_type& x, const bool& s) : value_(x), sign_(s) {}
	
	operator Log() const { return Log(*this); }
	
      private:
	weight_type value_;
	char        sign_;
      };

    public:
      static inline self_type exp(const weight_type& x, const bool& s) { return proxy_type(x, s); }
      static inline self_type pow(const self_type& x,  const weight_type& y) { return proxy_type(x.value_ * y, x.sign_); }
      static inline self_type zero() { return proxy_type(impl::traits_infinity<value_type>::minus(), false); }
      static inline self_type one()  { return proxy_type(0, false); }
      static inline self_type max()  { return proxy_type(impl::traits_infinity<value_type>::plus(), false); }
      static inline self_type min()  { return proxy_type(impl::traits_infinity<value_type>::plus(), true); }
      
    public:
      Log() : value_(impl::traits_infinity<value_type>::minus()), sign_(false) {}
      Log(const weight_type& x) : value_(std::signbit(x) ? std::log(-x) : std::log(x)), sign_(std::signbit(x)) {}
      explicit Log(const proxy_type& x) : value_(x.value_), sign_(x.sign_) {}

      operator Tp() const { return  sign_ ? - std::exp(value_) : std::exp(value_); }
      
    public:
      template <typename T>
      friend
      T log(const Log<T>& x);
      
      Log& operator+=(const Log& x)
      {
	using namespace boost::math::policies;
	typedef policy<domain_error<errno_on_error>,
	  pole_error<errno_on_error>,
	  overflow_error<errno_on_error>,
	  rounding_error<errno_on_error>,
	  evaluation_error<errno_on_error>
	  > policy_type;

	if (x.value_ == impl::traits_infinity<value_type>::minus())
	  return *this;
	else if (value_ == impl::traits_infinity<value_type>::minus()) {
	  *this = x;
	  return *this;
	}

	if (sign_ == x.sign_) {
	  if (x.value_ < value_)
	    value_ = value_ + utils::mathop::log1p(std::exp(x.value_ - value_));
	  else
	    value_ = x.value_ + utils::mathop::log1p(std::exp(value_ - x.value_));
	  
	} else {
	  if (x.value_ == value_)
	    *this = zero();
	  else if (x.value_ < value_) {
	    const Tp exp_value = std::exp(x.value_ - value_);
	    if (exp_value == 1.0)
	      *this = zero();
	    else
	      value_ = value_ + utils::mathop::log1p(- exp_value);
	  } else {
	    const Tp exp_value = std::exp(value_ - x.value_);
	    if (exp_value == 1.0)
	      *this = zero();
	    else {
	      value_ = x.value_ + utils::mathop::log1p(- exp_value);
	      sign_ = ! sign_;
	    }
	  }
	}
	
	return *this;
      }
      
      Log& operator-=(const Log& x)
      {
	return *this += Log(proxy_type(x.value_, ! x.sign_));
      }
      
      Log& operator*=(const Log& x)
      {
	sign_ = (sign_ != x.sign_);
	value_ += x.value_;
	return *this;
      }

      Log& operator/=(const Log& x)
      {
	sign_ = (sign_ != x.sign_);
	value_ -= x.value_;
	return *this;
      }
      
      friend
      bool operator==(const self_type& x, const self_type& y) { return x.value_ == y.value_ && x.sign_ == y.sign_; }
      friend
      bool operator!=(const self_type& x, const self_type& y) { return x.value_ != y.value_ || x.sign_ != y.sign_; }
      friend
      bool operator<(const self_type& x, const self_type& y) { return (x.sign_ > y.sign_) || (x.sign_ && x.value_ > y.value_) || (x.value_ < y.value_); }
      friend
      bool operator>(const self_type& x, const self_type& y) { return y < x; }
      friend
      bool operator<=(const self_type& x, const self_type& y) { return ! (y < x); }
      friend
      bool operator>=(const self_type& x, const self_type& y) { return ! (x < y); }
      
      friend
      std::ostream& operator<<(std::ostream& os, const self_type& x)
      {
	os << (x.sign_ ? '-' : '+') << x.value_;
	return os;
      }
      
      friend
      std::istream& operator>>(std::istream& is, self_type& x)
      {
	char __char;

	is >> __char >> x.value_;

	switch (__char) {
	case '+': x.sign_ = false; break;
	case '-': x.sign_ = true;  break;
	default: 
	  throw std::runtime_error("invlaid sign");
	}
	return is;
      }

      
    private:
      weight_type value_;
      char        sign_;
    };

    template <typename Tp>
    inline
    Tp log(const Log<Tp>& x)
    {
      if (x.sign_)
	throw std::runtime_error("no negative log");
      
      return x.value_;
    }
    
    template <typename Tp>
    inline
    Log<Tp> operator+(const Log<Tp>& x, const Log<Tp>& y)
    {
      Log<Tp> value(x);
      value += y;
      return value;
    }

    template <typename Tp>
    inline
    Log<Tp> operator-(const Log<Tp>& x, const Log<Tp>& y)
    {
      Log<Tp> value(x);
      value -= y;
      return value;
    }

    template <typename Tp>
    inline
    Log<Tp> operator*(const Log<Tp>& x, const Log<Tp>& y)
    {
      Log<Tp> value(x);
      value *= y;
      return value;
    }

    template <typename Tp>
    inline
    Log<Tp> operator/(const Log<Tp>& x, const Log<Tp>& y)
    {
      Log<Tp> value(x);
      value /= y;
      return value;
    }
    
    template <typename Tp>
    struct traits<Log<Tp> >
    {
      static inline Log<Tp> exp(const Tp& x) { return Log<Tp>::exp(x, false); }
      static inline Log<Tp> pow(const Log<Tp>& x, const Tp& y) { return Log<Tp>::pow(x, y); }
      static inline Log<Tp> zero() { return Log<Tp>::zero();  }
      static inline Log<Tp> one()  { return Log<Tp>::one(); }
      static inline Log<Tp> max()  { return Log<Tp>::max(); }
      static inline Log<Tp> min()  { return Log<Tp>::min(); }
    };
  };
};


#endif
