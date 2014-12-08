// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__WEIGHT_VECTOR__HPP__
#define __TRANCE__WEIGHT_VECTOR__HPP__ 1

#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <string>

#include <Eigen/Core>

#include <trance/feature.hpp>

#include <utils/bithack.hpp>
#include <utils/aligned_allocator.hpp>

namespace trance
{
  
  // forward declaration...
  template <typename Tp, typename Alloc >
  class FeatureVector;

  class FeatureVectorCompact;

  template <typename Tp, typename Alloc >
  class FeatureVectorLinear;

  template <typename Tp, typename Alloc=utils::aligned_allocator<Tp> >
  class WeightVector
  {
  public:
    typedef trance::Feature feature_type;
    typedef trance::Feature key_type;

  private:
    typedef std::vector<Tp, Alloc> weight_vector_type;
    typedef WeightVector<Tp, Alloc> self_type;
    
  public:
    typedef typename weight_vector_type::size_type       size_type;
    typedef typename weight_vector_type::difference_type difference_type;
    
    typedef typename weight_vector_type::value_type      value_type;
    
    typedef typename weight_vector_type::const_iterator  const_iterator;
    typedef typename weight_vector_type::iterator        iterator;
    typedef typename weight_vector_type::const_reference const_reference;
    typedef typename weight_vector_type::reference       reference;

  public:
    typedef Eigen::Map<Eigen::Matrix<Tp, Eigen::Dynamic, 1> > matrix_type;
    typedef Eigen::Map<Eigen::Array<Tp, Eigen::Dynamic, 1> > array_type;

  public:
    WeightVector() {}
    WeightVector(const WeightVector& x) : values_(x.values_) {}
    
  public:
    size_type size() const { return values_.size(); }
    bool empty() const { return values_.empty(); }
    
    void reserve(size_type x) { values_.reserve(x); }
    void clear() { values_.clear(); }
    
    void swap(WeightVector& x) { values_.swap(x.values_); }
    
    void allocate(const Tp& value)
    {
      values_.reserve(feature_type::allocated());
      values_.resize(feature_type::allocated(), value);
    }

    void allocate()
    {
      values_.reserve(feature_type::allocated());
      values_.resize(feature_type::allocated(), Tp());
    }
    
    Tp operator[](const key_type& x) const
    {
      return (x.id() < values_.size() ? values_[x.id()] : Tp());
    }
    
    Tp& operator[](const key_type& x) 
    {
      if (x.id() >= values_.size())
	values_.resize(x.id() + 1, Tp());
      return values_[x.id()];
    }

    void erase(const key_type& x)
    {
      if (x.id() < values_.size())
	values_[x.id()] = Tp();
    }
        
    const_iterator begin() const { return values_.begin(); }
    iterator begin() { return values_.begin(); }
    const_iterator end() const { return values_.end(); }
    iterator end() { return values_.end(); }

    const_reference front() const { return values_.front(); }
    reference front() { return values_.front(); }
    
    const_reference back() const { return values_.back(); }
    reference back() { return values_.back(); }

    matrix_type matrix() const { return matrix_type(const_cast<Tp*>(&(*values_.begin())), values_.size()); }
    matrix_type matrix()       { return matrix_type(&(*values_.begin()), values_.size()); }

    array_type array() const { return array_type(const_cast<Tp*>(&(*values_.begin())), values_.size()); }
    array_type array()       { return array_type(&(*values_.begin()), values_.size()); }

    Tp norm_l1() const
    {
      Tp ret = 0;
      const_iterator iter_end = end();
      for (const_iterator iter = begin(); iter != iter_end; ++ iter)
	ret += std::fabs(*iter);
      return ret;
    }

    Tp norm_l2() const
    {
      return matrix().squaredNorm();
    }
    
  public:
    // operators...
    template <typename T>
    self_type& operator+=(const T& x)
    {
      array() += x;
      //std::transform(begin(), end(), begin(), std::bind2nd(std::plus<Tp>(), x));
      return *this;
    }
    
    template <typename T>
    self_type& operator-=(const T& x)
    {
      array() -= x;
      //std::transform(begin(), end(), begin(), std::bind2nd(std::minus<Tp>(), x));
      return *this;
    }
    
    template <typename T>
    self_type& operator*=(const T& x)
    {
      array() *= x;
      //std::transform(begin(), end(), begin(), std::bind2nd(std::multiplies<Tp>(), x));
      return *this;
    }
    
    template <typename T>
    self_type& operator/=(const T& x)
    {
      array() /= x;
      //std::transform(begin(), end(), begin(), std::bind2nd(std::divides<Tp>(), x));
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator+=(const WeightVector<T, A>& x)
    {
      if (size() < x.size()) {
	values_.reserve(x.size());
	values_.resize(x.size());
      }

      if (size() == x.size())
	array() += x.array();
      else
	std::transform(begin(), begin() + x.size(), x.begin(), begin(), std::plus<Tp>());
      
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator-=(const WeightVector<T, A>& x)
    {
      if (size() < x.size()) {
	values_.reserve(x.size());
	values_.resize(x.size());
      }
      
      if (size() == x.size())
	array() -= x.array();
      else
	std::transform(begin(), begin() + x.size(), x.begin(), begin(), std::minus<Tp>());
      
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator*=(const WeightVector<T, A>& x)
    {
      if (size() < x.size()) {
	values_.reserve(x.size());
	values_.resize(x.size());
      }
      
      if (size() == x.size())
	array() *= x.array();
      else {
	// transform
	std::transform(begin(), begin() + x.size(), x.begin(), begin(), std::multiplies<Tp>());
	
	//std::transform(begin() + x.size(), end(), begin() + x.size(), std::bind2nd(std::multiplies<Tp>(), Tp()));
	std::fill(begin() + x.size(), end(), Tp());
      }
      
      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator/=(const WeightVector<T, A>& x)
    {
      if (size() < x.size()) {
	values_.reserve(x.size());
	values_.resize(x.size());
      }
      
      if (size() == x.size())
	array() /= x.array();
      else {
	std::transform(begin(), begin() + x.size(), x.begin(), begin(), std::divides<Tp>());
	
	std::transform(begin() + x.size(), end(), begin() + x.size(), std::bind2nd(std::divides<Tp>(), Tp()));
      }
      
      return *this;
    }


    template <typename T, typename A>
    self_type& operator+=(const FeatureVector<T, A>& x)
    {
      typedef typename FeatureVector<T, A>::const_iterator iter_type;
      
      iter_type iter_end = x.end();
      for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
	operator[](iter->first) += iter->second;

      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator-=(const FeatureVector<T, A>& x)
    {
      typedef typename FeatureVector<T, A>::const_iterator iter_type;
      
      iter_type iter_end = x.end();
      for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
	operator[](iter->first) -= iter->second;
      
      return *this;
    }
    

    template <typename T, typename A>
    self_type& operator+=(const FeatureVectorLinear<T, A>& x)
    {
      typedef typename FeatureVectorLinear<T, A>::const_iterator iter_type;
      
      if (! x.empty())
	if (x.back().first.id() >= values_.size()) {
	  values_.reserve(x.back().first.id() + 1);
	  values_.resize(x.back().first.id() + 1);
	}
      
      iter_type iter_end = x.end();
      for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
	operator[](iter->first) += iter->second;

      return *this;
    }
    
    template <typename T, typename A>
    self_type& operator-=(const FeatureVectorLinear<T, A>& x)
    {
      typedef typename FeatureVectorLinear<T, A>::const_iterator iter_type;
      
      if (! x.empty())
	if (x.back().first.id() >= values_.size()) {
	  values_.reserve(x.back().first.id() + 1);
	  values_.resize(x.back().first.id() + 1);
	}

      iter_type iter_end = x.end();
      for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
	operator[](iter->first) -= iter->second;

      return *this;
    }

    self_type& operator+=(const FeatureVectorCompact& x);
    self_type& operator-=(const FeatureVectorCompact& x);

  public:
    //comparison...
    friend
    bool operator==(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ == y.values_;
    }

    friend
    bool operator!=(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ != y.values_;
    }

    friend
    bool operator<(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ < y.values_;
    }

    friend
    bool operator<=(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ <= y.values_;
    }

    friend
    bool operator>(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ > y.values_;
    }

    friend
    bool operator>=(const WeightVector& x, const WeightVector& y)
    {
      return x.values_ >= y.values_;
    }
    
  public:
    
    template <typename T, typename A>
    friend
    std::ostream& operator<<(std::ostream& os, const WeightVector<T, A>& x);
    
    template <typename T, typename A>
    friend
    std::istream& operator>>(std::istream& is, WeightVector<T,A>& x);
    
  private:
    weight_vector_type values_;
  };
    

  template <typename T, typename A>
  inline
  std::ostream& operator<<(std::ostream& os, const WeightVector<T, A>& x)
  {
    typename WeightVector<T,A>::const_iterator iter_begin = x.begin();
    typename WeightVector<T,A>::const_iterator iter_end   = x.end();

    static const trance::Feature __empty;
      
    for (typename WeightVector<T,A>::const_iterator iter = iter_begin; iter != iter_end; ++ iter)
      if (*iter != 0.0) {
	trance::Feature feature(iter - iter_begin);
	if (feature != __empty)
	  os << feature << ' ' << *iter << '\n';
      }
      
    return os;
  }
    
  template <typename T, typename A>
  inline
  std::istream& operator>>(std::istream& is, WeightVector<T,A>& x)
  {
    x.clear();

    std::string feature;
    T value;
    while ((is >> feature) && (is >> value))
      x[trance::Feature(feature).id()] = value;

    return is;
  }

};

namespace std
{
  
  template <typename T, typename A>
  inline
  void swap(trance::WeightVector<T,A>& x, trance::WeightVector<T,A>& y)
  {
    x.swap(y);
  }
  
};

#include <trance/feature_vector.hpp>
#include <trance/feature_vector_compact.hpp>
#include <trance/feature_vector_linear.hpp>

namespace trance
{
  template <typename T, typename A>
  inline
  WeightVector<T,A>& WeightVector<T,A>::operator+=(const FeatureVectorCompact& x)
  {
    typedef typename FeatureVectorCompact::const_iterator iter_type;
    
    iter_type iter_end = x.end();
    for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
      operator[](iter->first) += iter->second;
    
    return *this;
  }

  template <typename T, typename A>
  inline
  WeightVector<T,A>& WeightVector<T,A>::operator-=(const FeatureVectorCompact& x)
  {
    typedef typename FeatureVectorCompact::const_iterator iter_type;
    
    iter_type iter_end = x.end();
    for (iter_type iter = x.begin(); iter != iter_end; ++ iter)
      operator[](iter->first) -= iter->second;
    
    return *this;    
  }
};

#endif
