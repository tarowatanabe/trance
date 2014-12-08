// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__FEATURE_VECTOR_LINEAR__HPP__
#define __TRANCE__FEATURE_VECTOR_LINEAR__HPP__ 1

#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include <iterator>
#include <iostream>

#include <trance/feature.hpp>

#include <utils/vector_map.hpp>
#include <utils/hashmurmur3.hpp>

//
// feature vector, for use as "temporary" linear vector for faster access, w/o sorting
//

namespace trance
{
  
  // forward declaration...
  template <typename Tp, typename Alloc >
  class FeatureVector;

  class FeatureVectorCompact;
    
  template <typename Tp, typename Alloc >
  class WeightVector;
  
  template <typename Tp, typename Alloc=std::allocator<Tp> >
  class FeatureVectorLinear
  {
  public:
    typedef trance::Feature feature_type;
    
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

  private:
    typedef std::pair<feature_type, Tp> __value_type;
    typedef typename Alloc::template rebind<__value_type>::other alloc_type;
    typedef utils::vector_map<feature_type, Tp, std::less<feature_type>,  alloc_type> map_type;
    
  public:
    typedef typename map_type::key_type    key_type;
    typedef typename map_type::data_type   data_type;
    typedef typename map_type::mapped_type mapped_type;
    typedef typename map_type::value_type  value_type;
    
    typedef typename map_type::const_iterator  const_iterator;
    typedef typename map_type::iterator        iterator;
     
    typedef typename map_type::const_reference  const_reference;
    typedef typename map_type::reference        reference;
     
  public:
    typedef FeatureVectorLinear<Tp, Alloc> self_type;
     
  public:
    FeatureVectorLinear() : map_() { }
    FeatureVectorLinear(const self_type& x) : map_(x.map_) { }
    template <typename T, typename A>
    FeatureVectorLinear(const FeatureVector<T,A>& x) : map_() { assign(x); }
    FeatureVectorLinear(const FeatureVectorCompact& x) : map_() { assign(x); }
    template <typename T, typename A>
    FeatureVectorLinear(const FeatureVectorLinear<T,A>& x) : map_() { assign(x); }
    template <typename Iterator>
    FeatureVectorLinear(Iterator first, Iterator last) : map_() { map_.insert(first, last); } 
    
    FeatureVectorLinear& operator=(const self_type& x)
    {
      map_ = x.map_;
      return *this;
    }
    
    template <typename T, typename A>
    FeatureVectorLinear& operator=(const FeatureVector<T,A>& x)
    {
      assign(x);
      return *this;
    }
     
    FeatureVectorLinear& operator=(const FeatureVectorCompact& x)
    {
      assign(x);
      return *this;
    }

    template <typename T, typename A>
    FeatureVectorLinear& operator=(const FeatureVectorLinear<T,A>& x)
    {
      assign(x);
      return *this;
    }

  public:
    size_type size() const { return map_.size(); }
    bool empty() const { return map_.empty(); }
     
    void assign(const self_type& x)
    {
      map_ = x.map_;
    }

    template <typename T, typename A>
    void assign(const FeatureVector<T,A>& x)
    {
      map_.clear();
      map_.insert(x.begin(), x.end());
    }

    void assign(const FeatureVectorCompact& x);

    template <typename T, typename A>
    void assign(const FeatureVectorLinear<T,A>& x)
    {
      map_.clear();
      map_.insert(x.begin(), x.end());
    }
    
    template <typename Iterator>
    void assign(Iterator first, Iterator last)
    {
      map_.clear();
      map_.insert(first, last);
    }
     
    Tp operator[](const key_type& x) const
    {
      const_iterator iter = find(x);
      return (iter == end() ? Tp() : iter->second);
    }
     
    Tp& operator[](const key_type& x)
    {
      return map_[x];
    }

    inline const_iterator begin() const { return map_.begin(); }
    inline       iterator begin()       { return map_.begin(); }
     
    inline const_iterator end() const { return map_.end(); }
    inline       iterator end()       { return map_.end(); }

    reference front() { return map_.front(); }
    const_reference front() const { return map_.front(); }
    reference back() { return map_.back(); }
    const_reference back() const { return map_.back(); }
    
    inline const_iterator find(const key_type& x) const { return map_.find(x); }
    inline       iterator find(const key_type& x)       { return map_.find(x); }
    
    inline const_iterator lower_bound(const key_type& x) const { return map_.lower_bound(x); }
    inline       iterator lower_bound(const key_type& x)       { return map_.lower_bound(x); }
    
    inline const_iterator upper_bound(const key_type& x) const { return map_.upper_bound(x); }
    inline       iterator upper_bound(const key_type& x)       { return map_.upper_bound(x); }

    void erase(const key_type& x) { map_.erase(x); }
     
    void swap(FeatureVectorLinear& x) { map_.swap(x.map_); }
     
    void clear() { map_.clear(); }

  private:
    template <typename O, typename T>
    struct __apply_unary : public O
    {
      __apply_unary(const T& x) : const_value(x) {}
      
      template <typename Value>
      void operator()(Value& value) const
      {
	value.second = O::operator()(value.second, const_value);
      }

      T const_value;
    };

  public:
    //operators...
     
    template <typename T>
    FeatureVectorLinear& operator+=(const T& x)
    {
      std::for_each(begin(), end(), __apply_unary<std::plus<Tp>, T>(x));
      return *this;
    }

    template <typename T>
    FeatureVectorLinear& operator-=(const T& x)
    {
      std::for_each(begin(), end(), __apply_unary<std::minus<Tp>, T>(x));
      return *this;
    }
     
    template <typename T>
    FeatureVectorLinear& operator*=(const T& x)
    {
      if (x == T())
	clear();
      else
	std::for_each(begin(), end(), __apply_unary<std::multiplies<Tp>, T>(x));
      return *this;
    }
    
    template <typename T>
    FeatureVectorLinear& operator/=(const T& x)
    {
      std::for_each(begin(), end(), __apply_unary<std::divides<Tp>, T>(x));
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator+=(const FeatureVector<T,A>& x)
    {
      if (x.empty())
	return *this;
      else if (empty()) {
	assign(x);
	return *this;
      } else {
	plus_equal(map_, x.begin(), x.end());
	return *this;
      }
    }

    template <typename T, typename A>
    self_type& operator-=(const FeatureVector<T,A>& x)
    {
      minus_equal(map_, x.begin(), x.end());
      return *this;
    }


    template <typename T, typename A>
    self_type& operator*=(const FeatureVector<T,A>& x)
    {
      if (empty() || x.empty()) {
	clear();
	return *this;
      } else {
	map_type map_new;
	multiply_equal(map_new, map_, x.begin(), x.end());
	map_.swap(map_new);
	return *this;
      }
    }


    template <typename T, typename A>
    self_type& operator+=(const FeatureVectorLinear<T,A>& x)
    {
      if (x.empty())
	return *this;
      else if (empty()) {
	assign(x);
	return *this;
      } else {
	plus_equal_ordered(map_, x.begin(), x.end());
	return *this;
      }
    }
    
    template <typename T, typename A>
    self_type& operator-=(const FeatureVectorLinear<T,A>& x)
    {
      minus_equal_ordered(map_, x.begin(), x.end());
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator*=(const FeatureVectorLinear<T,A>& x)
    {
      if (empty() || x.empty()) {
	clear();
	return *this;
      } else {
	map_type map_new;
	multiply_equal_ordered(map_new, map_.begin(), map_.end(), x.begin(), x.end());
	map_.swap(map_new);
	return *this;
      }
    }
    self_type& operator+=(const FeatureVectorCompact& x);
    self_type& operator-=(const FeatureVectorCompact& x);
    self_type& operator*=(const FeatureVectorCompact& x);

  private:
    template <typename Container, typename Iterator>
    static inline
    void plus_equal(Container& container, Iterator first, Iterator last)
    {
      for (/**/; first != last; ++ first) {
	std::pair<typename Container::iterator, bool> result = container.insert(*first);
	
	if (! result.second) {
	  result.first->second += first->second;
	  
	  if (result.first->second == Tp())
	    container.erase(result.first);
	}
      }
    }

    template <typename Container, typename Iterator>
    static inline
    void plus_equal_ordered(Container& container, Iterator first, Iterator last)
    {
      typename Container::iterator hint = container.begin();

      for (/**/; first != last && hint != container.end(); ++ first) {
	std::pair<typename Container::iterator, bool> result = container.insert(*first);
	
	hint = result.first;
        ++ hint;
	
	if (! result.second) {
	  result.first->second += first->second;
	  
	  if (result.first->second == Tp())
	    container.erase(result.first);
	}
      }
      
      if (first != last)
	container.insert(first, last);
    }
    
    template <typename Container, typename Iterator>
    static inline
    void minus_equal(Container& container, Iterator first, Iterator last)
    {
      for (/**/; first != last; ++ first) {
	std::pair<typename Container::iterator, bool> result = container.insert(std::make_pair(first->first, -Tp(first->second)));
	
	if (! result.second) {
	  result.first->second -= first->second;
	  
	  if (result.first->second == Tp())
	    container.erase(result.first);
	}
      }
    }

    template <typename Container, typename Iterator>
    static inline
    void minus_equal_ordered(Container& container, Iterator first, Iterator last)
    {
      typename Container::iterator hint = container.begin();

      for (/**/; first != last && hint != container.end(); ++ first) {
	std::pair<typename Container::iterator, bool> result = container.insert(std::make_pair(first->first, -Tp(first->second)));
	
	hint = result.first;
        ++ hint;
	
	if (! result.second) {
	  result.first->second -= first->second;
	  
	  if (result.first->second == Tp())
	    container.erase(result.first);
	}
      }
      
      for (/**/; first != last; ++ first)
        container.insert(container.end(), std::make_pair(first->first, -Tp(first->second)));
    }
    
    template <typename Container, typename Original, typename Iterator>
    static inline
    void multiply_equal(Container& container, const Original& orig, Iterator first, Iterator last)
    {
      for (/**/; first != last; ++ first) {
	typename Original::const_iterator iter = orig.find(first->first);
	
	if (iter == orig.end()) continue;
	
	const Tp value(iter->second * first->second);
	
	if (value != Tp())
	  container.insert(std::make_pair(first->first, value));
      }
    }

    template <typename Container, typename Iterator1, typename Iterator2>
    static inline
    void multiply_equal_ordered(Container& container,
				Iterator1 first1, Iterator1 last1,
				Iterator2 first2, Iterator2 last2)
    {
      while (first1 != last1 && first2 != last2) {
	if (first1->first < first2->first)
	  ++ first1;
	else if (first2->first < first1->first)
	  ++ first2;
	else {
	  const Tp value = first1->second * first2->second;
	  
	  if (value != Tp())
	    container.insert(container.end(), std::make_pair(first1->first, value));
	  
	  ++ first1;
	  ++ first2;
	}
      }
    }
    
    
  public:
    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator+(const FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y);
    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator-(const FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y);
    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator*(const FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y);

    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator+(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y);
    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator-(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y);
    template <typename T1, typename A1, typename T2, typename A2>
    friend
    FeatureVectorLinear<T1,A1> operator*(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y);
    
    template <typename T1, typename A1>
    friend
    FeatureVectorLinear<T1,A1> operator+(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y);
    
    template <typename T1, typename A1>
    friend
    FeatureVectorLinear<T1,A1> operator-(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y);
    
    template <typename T1, typename A1>
    friend
    FeatureVectorLinear<T1,A1> operator*(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y);
    
    friend
    size_t hash_value(FeatureVectorLinear const& x) { return utils::hashmurmur3<size_t>()(x.map_.begin(), x.map_.end(), 0); }
    
    friend bool operator==(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ == y.map_; }
    friend bool operator!=(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ != y.map_; }
    friend bool operator<(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ < y.map_; }
    friend bool operator>(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ > y.map_; }
    friend bool operator<=(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ <= y.map_; }
    friend bool operator>=(const FeatureVectorLinear& x, const FeatureVectorLinear& y) { return x.map_ >= y.map_; }
     
  private:
    map_type map_;
  };
  

  template <typename T1, typename A1, typename T2>
  inline
  FeatureVectorLinear<T1,A1> operator+(const FeatureVectorLinear<T1,A1>& x, const T2& y)
  {
    FeatureVectorLinear<T1,A1> features(x);
    features += y;
    return features;
  }

  template <typename T2, typename T1, typename A1>
  inline
  FeatureVectorLinear<T1,A1> operator+(const T2& x, const FeatureVectorLinear<T1,A1>& y)
  {
    FeatureVectorLinear<T1,A1> features(y);
    features += x;
    return features;
  }

  template <typename T1, typename A1, typename T2>
  inline
  FeatureVectorLinear<T1,A1> operator*(const FeatureVectorLinear<T1,A1>& x, const T2& y)
  {
    if (y == T2()) return FeatureVectorLinear<T1,A1>();
    
    FeatureVectorLinear<T1,A1> features(x);
    features *= y;
    return features;
  }

  template <typename T2, typename T1, typename A1>
  inline
  FeatureVectorLinear<T1,A1> operator*(const T2& x, const FeatureVectorLinear<T1,A1>& y)
  {
    if (x == T2()) return FeatureVectorLinear<T1,A1>();
    
    FeatureVectorLinear<T1,A1> features(y);
    features *= x;
    return features;
  }

  template <typename T1, typename A1, typename T2>
  inline
  FeatureVectorLinear<T1,A1> operator-(const FeatureVectorLinear<T1,A1>& x, const T2& y)
  {
    FeatureVectorLinear<T1,A1> features(x);
    features -= y;
    return features;
  }
  
  template <typename T1, typename A1, typename T2>
  inline
  FeatureVectorLinear<T1,A1> operator/(const FeatureVectorLinear<T1,A1>& x, const T2& y)
  {
    FeatureVectorLinear<T1,A1> features(x);
    features /= y;
    return features;
  }

  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator+(FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new += y;
    return x_new;
  }
  
  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator-(FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new -= y;
    return x_new;    
  }
  
  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator*(FeatureVectorLinear<T1,A1>& x, const FeatureVector<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new *= y;
    return x_new;
  }

  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator+(FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new += y;
    return x_new;
  }
  
  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator-(FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new -= y;
    return x_new;    
  }
  
  template <typename T1, typename A1, typename T2, typename A2>
  inline
  FeatureVectorLinear<T1,A1> operator*(FeatureVectorLinear<T1,A1>& x, const FeatureVectorLinear<T2,A2>& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new;
    self_type::multiply_equal_ordered(x_new.map_, x.begin(), x.end(), y.begin(), y.end());
    return x_new;
  }

};

namespace std
{
  template <typename T, typename A>
  inline
  void swap(trance::FeatureVectorLinear<T,A>& x, trance::FeatureVectorLinear<T, A>& y)
  {
    x.swap(y);
  }
};

#include <trance/weight_vector.hpp>
#include <trance/feature_vector.hpp>
#include <trance/feature_vector_compact.hpp>

namespace trance
{
  template <typename T, typename A>
  inline
  void FeatureVectorLinear<T,A>::assign(const FeatureVectorCompact& x)
  {
    map_.clear();
    map_.insert(x.begin(), x.end());
  }

  template <typename T, typename A>
  inline
  FeatureVectorLinear<T,A>& FeatureVectorLinear<T,A>::operator+=(const FeatureVectorCompact& x)
  {
    if (x.empty())
      return *this;
    else if (empty()) {
      assign(x);
      return *this;
    } else {
      plus_equal_ordered(map_, x.begin(), x.end());
      return *this;
    }
  }

  template <typename T, typename A>
  inline
  FeatureVectorLinear<T,A>& FeatureVectorLinear<T,A>::operator-=(const FeatureVectorCompact& x)
  {
    minus_equal_ordered(map_, x.begin(), x.end());
    return *this;
  }
  
  template <typename T, typename A>
  inline
  FeatureVectorLinear<T,A>& FeatureVectorLinear<T,A>::operator*=(const FeatureVectorCompact& x)
  {
    if (empty() || x.empty()) {
      clear();
      return *this;
    } else {
      map_type map_new;
      multiply_equal_ordered(map_new, map_.begin(), map_.end(), x.begin(), x.end());
      map_.swap(map_new);
      return *this;
    } 
  }
  
  template <typename T1, typename A1>
  inline
  FeatureVectorLinear<T1,A1> operator+(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new += y;
    return x_new;
  }
  
  template <typename T1, typename A1>
  inline
  FeatureVectorLinear<T1,A1> operator-(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new(x);
    x_new -= y;
    return x_new;
  }

  template <typename T1, typename A1>
  inline
  FeatureVectorLinear<T1,A1> operator*(const FeatureVectorLinear<T1,A1>& x, const FeatureVectorCompact& y)
  {
    typedef FeatureVectorLinear<T1,A1> self_type;
    
    self_type x_new;
    self_type::multiply_equal_ordered(x_new.map_, x.begin(), x.end(), y.begin(), y.end());
    return x_new;
  }

};

#endif
