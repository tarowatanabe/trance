// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__SEMIRING__TROPICAL__HPP__
#define __RNNP__SEMIRING__TROPICAL__HPP__ 1

#include <cmath>
#include <cfloat>
#include <climits>

#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <rnnp/semiring/traits.hpp>

namespace rnnp
{
  namespace semiring
  {

    template <typename Tp>
    class Tropical
    {
    public:
      typedef Tp weight_type;
      typedef Tp value_type;
      typedef Tropical<Tp> self_type;

    public:
      struct proxy_type
      {
	friend class Tropical;
	
	proxy_type(const weight_type& x) : value_(x) {}
	
	operator Tropical() const { return Tropical(*this); }
	
      private:
	weight_type value_;
      };
      
    public:
      static inline self_type exp(const Tp& x) { return proxy_type(x); }
      static inline self_type pow(const self_type& x, const Tp& y) { return proxy_type(x.value_ * y); }
      static inline self_type zero() { return proxy_type(impl::traits_infinity<value_type>::minus()); }
      static inline self_type one()  { return proxy_type(0); }
      static inline self_type max()  { return proxy_type(impl::traits_infinity<value_type>::plus()); }
      static inline self_type min()  { return proxy_type(impl::traits_infinity<value_type>::minus()); }
      
    public:
      // any better wayt to make an assignment...?
      Tropical() : value_(impl::traits_infinity<value_type>::minus()) {}
      Tropical(const weight_type& x) : value_(std::log(x)) {}
      explicit Tropical(const proxy_type& x) : value_(x.value_) {}


      operator Tp() const { return std::exp(value_); }
      
    public:
      template <typename T>
      friend
      T log(const Tropical<T>& x);

      Tropical& operator+=(const Tropical& x)
      {
	value_ = std::max(value_, x.value_);
	
	return *this;
      }
      
      Tropical& operator*=(const Tropical& x)
      {
	value_ += x.value_;
	return *this;
      }

      Tropical& operator/=(const Tropical& x)
      {
	value_ -= x.value_;
	return *this;
      }
      
      friend
      bool operator==(const self_type& x, const self_type& y) { return x.value_ == y.value_; }
      friend
      bool operator!=(const self_type& x, const self_type& y) { return x.value_ != y.value_; }
      friend
      bool operator>(const self_type& x, const self_type& y) { return x.value_ > y.value_; }
      friend
      bool operator<(const self_type& x, const self_type& y) { return x.value_ < y.value_; }
      friend
      bool operator>=(const self_type& x, const self_type& y) { return x.value_ >= y.value_; }
      friend
      bool operator<=(const self_type& x, const self_type& y) { return x.value_ <= y.value_; }

      friend
      std::ostream& operator<<(std::ostream& os, const self_type& x)
      {
	os << x.value_;
	return os;
      }
      
      friend
      std::istream& operator>>(std::istream& is, self_type& x)
      {
	is >> x.value_;
	return is;
      }
      
    private:
      weight_type value_;
    };

    template <typename Tp>
    inline
    Tp log(const Tropical<Tp>& x)
    {
      return x.value_;
    }
    
    template <typename Tp>
    inline
    Tropical<Tp> operator+(const Tropical<Tp>& x, const Tropical<Tp>& y)
    {
      Tropical<Tp> value(x);
      value += y;
      return value;
    }

    template <typename Tp>
    inline
    Tropical<Tp> operator*(const Tropical<Tp>& x, const Tropical<Tp>& y)
    {
      Tropical<Tp> value(x);
      value *= y;
      return value;
    }

    template <typename Tp>
    inline
    Tropical<Tp> operator/(const Tropical<Tp>& x, const Tropical<Tp>& y)
    {
      Tropical<Tp> value(x);
      value /= y;
      return value;
    }
    
    template <typename Tp>
    struct traits<Tropical<Tp> >
    {
      static inline Tropical<Tp> exp(const Tp& x) { return Tropical<Tp>::exp(x); }
      static inline Tropical<Tp> pow(const Tropical<Tp>& x, const Tp& y) { return Tropical<Tp>::pow(x, y); }
      static inline Tropical<Tp> zero() { return Tropical<Tp>::zero();  }
      static inline Tropical<Tp> one()  { return Tropical<Tp>::one(); }
      static inline Tropical<Tp> max()  { return Tropical<Tp>::max(); }
      static inline Tropical<Tp> min()  { return Tropical<Tp>::min(); }
    };

  };
};


#endif
