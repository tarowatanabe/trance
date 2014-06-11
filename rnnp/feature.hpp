// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__FEATURE__HPP__
#define __RNNP__FEATURE__HPP__ 1

#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>

#include <utils/indexed_set.hpp>
#include <utils/rwticket.hpp>
#include <utils/piece.hpp>
#include <utils/chunk_vector.hpp>
#include <utils/traits.hpp>
#include <utils/bithack.hpp>

namespace rnnp
{
  struct FeatureImpl;

  class Feature
  {
  private:
    friend struct FeatureImpl;
    
  public:
    typedef std::string  feature_type;
    typedef utils::piece piece_type;
    typedef uint32_t     id_type;
    
    typedef feature_type::size_type              size_type;
    typedef feature_type::difference_type        difference_type;
    
    typedef feature_type::value_type             value_type;
    typedef feature_type::const_iterator         const_iterator;
    typedef feature_type::const_reverse_iterator const_reverse_iterator;
    typedef feature_type::const_reference        const_reference;
    
  private:
    typedef utils::rwticket ticket_type;
    
  public:
    Feature() : id_(__allocate_empty()) { }
    Feature(const utils::piece& x) : id_(__allocate(x)) { }
    Feature(const feature_type& x) : id_(__allocate(x)) { }
    Feature(const char* x) : id_(__allocate(x)) { }
    Feature(const id_type& x) : id_(x) { }
    template <typename Iterator>
    Feature(Iterator first, Iterator last) : id_(__allocate(piece_type(first, last))) { }
    
    void assign(const piece_type& x) { id_ = __allocate(x); }
    void assign(const feature_type& x) { id_ = __allocate(x); }
    void assign(const char* x) { id_ = __allocate(x); }
    template <typename Iterator>
    void assign(Iterator first, Iterator last) { id_ = __allocate(piece_type(first, last)); }
    
  public:
    void swap(Feature& x) { std::swap(id_, x.id_); }
    
    id_type id() const { return id_; }
    operator const feature_type&() const { return feature(); }
    operator utils::piece() const { return feature(); }
    
    const feature_type& feature() const
    {
      feature_map_type& maps = __feature_maps();
      
      if (id_ >= maps.size()) {
	const size_type size = id_ + 1;
	const size_type power2 = utils::bithack::branch(utils::bithack::is_power2(size),
							size,
							size_type(utils::bithack::next_largest_power2(size)));
	maps.reserve(power2);
	maps.resize(power2, 0);
      }
      if (! maps[id_]) {
	ticket_type::scoped_reader_lock lock(__mutex);
	
	maps[id_] = &(__features()[id_]);
      }
      
      return *maps[id_];
    }
    
    const_iterator begin() const { return feature().begin(); }
    const_iterator end() const { return feature().end(); }
    
    const_reverse_iterator rbegin() const { return feature().rbegin(); }
    const_reverse_iterator rend() const { return feature().rend(); }
    
    const_reference operator[](size_type x) const { return feature()[x]; }
    
    size_type size() const { return feature().size(); }
    bool empty() const { return feature().empty(); }
        
  public:
    // boost hash
    friend
    size_t  hash_value(Feature const& x);
    
    // iostreams
    friend
    std::ostream& operator<<(std::ostream& os, const Feature& x);
    friend
    std::istream& operator>>(std::istream& is, Feature& x);
    
    // comparison...
    friend
    bool operator==(const Feature& x, const Feature& y);
    friend
    bool operator!=(const Feature& x, const Feature& y);
    friend
    bool operator<(const Feature& x, const Feature& y);
    friend
    bool operator>(const Feature& x, const Feature& y);
    friend
    bool operator<=(const Feature& x, const Feature& y);
    friend
    bool operator>=(const Feature& x, const Feature& y);
    
  private:
    typedef utils::indexed_set<piece_type, boost::hash<piece_type>, std::equal_to<piece_type>, std::allocator<piece_type> > feature_index_type;
    typedef utils::chunk_vector<feature_type, 4096 / sizeof(feature_type), std::allocator<feature_type> > feature_set_type;
    typedef std::vector<const feature_type*, std::allocator<const feature_type*> > feature_map_type;
    
  public:
    static bool exists(const piece_type& x)
    {
      ticket_type::scoped_reader_lock lock(__mutex);
      
      const feature_index_type& index = __index();
      
      return index.find(x) != index.end();
    }
    
    static size_t allocated()
    {
      ticket_type::scoped_reader_lock lock(__mutex);
      
      return __features().size();
    }
    
  private:
    static ticket_type    __mutex;
    
    static feature_map_type& __feature_maps();
    
    static feature_set_type& __features()
    {
      static feature_set_type feats;
      return feats;
    }
    
    static feature_index_type& __index()
    {
      static feature_index_type index;
      return index;
    }
    
    static const id_type& __allocate_empty()
    {
      static const id_type id_ = __allocate("");
      return id_;
    }
    
    static id_type __allocate(const piece_type& x)
    {
      ticket_type::scoped_writer_lock lock(__mutex);
      
      feature_index_type& index = __index();
      
      std::pair<feature_index_type::iterator, bool> result = index.insert(x);
      
      if (result.second) {
	feature_set_type& features = __features();
	features.push_back(x);
	const_cast<piece_type&>(*result.first) = features.back();
      }
      
      return result.first - index.begin();
    }
    
  private:
    id_type id_;
  };
  
  inline
  size_t hash_value(Feature const& x)
  {
    return x.id_;
  }
  
  inline
  std::ostream& operator<<(std::ostream& os, const Feature& x)
  {
    os << x.feature();
    return os;
  }
  
  inline
  std::istream& operator>>(std::istream& is, Feature& x)
  {
    std::string feature;
    is >> feature;
    x.assign(feature);
    return is;
  }
  
  inline
   bool operator==(const Feature& x, const Feature& y)
  {
    return x.id_ == y.id_;
  }
  inline
  bool operator!=(const Feature& x, const Feature& y)
  {
    return x.id_ != y.id_;
  }
  inline
  bool operator<(const Feature& x, const Feature& y)
  {
    return x.id_ < y.id_;
  }
  inline
  bool operator>(const Feature& x, const Feature& y)
  {
    return x.id_ > y.id_;
  }
  inline
  bool operator<=(const Feature& x, const Feature& y)
  {
    return x.id_ <= y.id_;
  }
  inline
  bool operator>=(const Feature& x, const Feature& y)
  {
    return x.id_ >= y.id_;
  }

};

namespace std
{
  inline
  void swap(rnnp::Feature& x, rnnp::Feature& y)
  {
    x.swap(y);
  }
};

namespace utils
{
  template <>
  struct traits<rnnp::Feature>
  {
    typedef rnnp::Feature     value_type;
    typedef value_type::id_type id_type;
    
    static inline value_type unassigned() { return value_type(id_type(-1)); }
    static inline value_type deleted() { return value_type(id_type(-2)); }
  };
};

#endif

