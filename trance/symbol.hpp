// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__SYMBOL__HPP__
#define __TRANCE__SYMBOL__HPP__ 1

#include <stdint.h>

#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <utils/indexed_set.hpp>
#include <utils/spinlock.hpp>
#include <utils/rwticket.hpp>
#include <utils/piece.hpp>
#include <utils/chunk_vector.hpp>
#include <utils/traits.hpp>
#include <utils/bithack.hpp>

namespace trance
{
  
  struct SymbolImpl;

  class Symbol
  {
  private:
    friend struct SymbolImpl;

  public:
    typedef std::string  symbol_type;
    typedef utils::piece piece_type;
    typedef uint32_t     id_type;
    
    typedef symbol_type::size_type              size_type;
    typedef symbol_type::difference_type        difference_type;
    
    typedef symbol_type::value_type             value_type;
    typedef symbol_type::const_iterator         const_iterator;
    typedef symbol_type::const_reverse_iterator const_reverse_iterator;
    typedef symbol_type::const_reference        const_reference;

    typedef boost::filesystem::path path_type;
    
  private:
    typedef utils::spinlock mutex_type;
    typedef utils::rwticket ticket_type;

  public:
    // constants
    static const Symbol EMPTY;
    static const Symbol EPSILON;
    static const Symbol UNK;

    static const Symbol AXIOM;
    static const Symbol FINAL;
    static const Symbol IDLE;
    
  public:
    Symbol() : id_(__allocate_empty()) { }
    Symbol(const piece_type& x) : id_(__allocate(x)) { }
    Symbol(const symbol_type& x) : id_(__allocate(x)) { }
    Symbol(const char* x) : id_(__allocate(x)) { }
    Symbol(const id_type& x) : id_(x) {}
    template <typename Iterator>
    Symbol(Iterator first, Iterator last) : id_(__allocate(piece_type(first, last))) { }
    
    void assign(const piece_type& x) { id_ = __allocate(x); }
    void assign(const symbol_type& x) { id_ = __allocate(x); }
    void assign(const char* x) { id_ = __allocate(x); }
    template <typename Iterator>
    void assign(Iterator first, Iterator last) { id_ = __allocate(piece_type(first, last)); }
    
  public:
    void swap(Symbol& x) { std::swap(id_, x.id_); }
    
    id_type id() const { return id_; }
    operator const symbol_type&() const { return symbol(); }
    operator utils::piece() const { return symbol(); }
    
    const symbol_type& symbol() const
    {
      symbol_map_type& maps = __symbol_maps();
      
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
	
	maps[id_] = &(__symbols()[id_]);
      }
      
      return *maps[id_];
    }
    
    const_iterator begin() const { return symbol().begin(); }
    const_iterator end() const { return symbol().end(); }
    
    const_reverse_iterator rbegin() const { return symbol().rbegin(); }
    const_reverse_iterator rend() const { return symbol().rend(); }
    
    const_reference operator[](size_type x) const { return symbol()[x]; }
    
    size_type size() const { return symbol().size(); }
    bool empty() const { return symbol().empty(); }
    
    // non-terminal id
    id_type non_terminal_id() const;
    
    // non_temrinal?
    bool terminal() const { return ! non_terminal(); }
    bool non_terminal() const;
    bool binarized() const;
    
    // non-terminal is [syntactic-label]. This will strip the squared brackets
    piece_type strip() const
    {
      if (non_terminal()) {
	const piece_type stripped(symbol());
	
	return stripped.substr(1, stripped.size() - 2);
      } else
	return symbol();
    }

  public:
    // boost hash
    friend
    size_t  hash_value(Symbol const& x);
    
    // iostreams
    friend
    std::ostream& operator<<(std::ostream& os, const Symbol& x);
    friend
    std::istream& operator>>(std::istream& is, Symbol& x);
    
    // comparison...
    friend
    bool operator==(const Symbol& x, const Symbol& y);
    friend
    bool operator!=(const Symbol& x, const Symbol& y);
    friend
    bool operator<(const Symbol& x, const Symbol& y);
    friend
    bool operator>(const Symbol& x, const Symbol& y);
    friend
    bool operator<=(const Symbol& x, const Symbol& y);
    friend
    bool operator>=(const Symbol& x, const Symbol& y);
    
    
  private:
    typedef utils::indexed_set<piece_type, boost::hash<piece_type>, std::equal_to<piece_type>, std::allocator<piece_type> > symbol_index_type;
    typedef utils::chunk_vector<symbol_type, 4096 / sizeof(symbol_type), std::allocator<symbol_type> > symbol_set_type;
    typedef std::vector<const symbol_type*, std::allocator<const symbol_type*> > symbol_map_type;
    
  public:
    static bool exists(const piece_type& x)
    {
      ticket_type::scoped_reader_lock lock(__mutex);
      
      const symbol_index_type& index = __index();
      
      return index.find(x) != index.end();
    }
    static size_t allocated()
    {
      ticket_type::scoped_reader_lock lock(__mutex);
      
      return __symbols().size();
    }
    
  private:
    static ticket_type    __mutex;
    
    static symbol_map_type& __symbol_maps();
    
    static symbol_set_type& __symbols()
    {
      static symbol_set_type syms;
      return syms;
    }
    
    static symbol_index_type& __index()
    {
      static symbol_index_type index;
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

      symbol_index_type& index = __index();
      
      std::pair<symbol_index_type::iterator, bool> result = index.insert(x);
      
      if (result.second) {
	symbol_set_type& symbols = __symbols();
	symbols.push_back(x);
	const_cast<piece_type&>(*result.first) = symbols.back();
      }
      
      return result.first - index.begin();
    }
    
  private:
    id_type id_;
  };
  
  inline
  size_t hash_value(Symbol const& x)
  {
    return x.id_;
  }
  
  inline
  std::ostream& operator<<(std::ostream& os, const Symbol& x)
  {
    os << x.symbol();
    return os;
  }
  
  inline
  std::istream& operator>>(std::istream& is, Symbol& x)
  {
    std::string symbol;
    is >> symbol;
    x.assign(symbol);
    return is;
  }
  
  inline
   bool operator==(const Symbol& x, const Symbol& y)
  {
    return x.id_ == y.id_;
  }
  inline
  bool operator!=(const Symbol& x, const Symbol& y)
  {
    return x.id_ != y.id_;
  }
  inline
  bool operator<(const Symbol& x, const Symbol& y)
  {
    return x.id_ < y.id_;
  }
  inline
  bool operator>(const Symbol& x, const Symbol& y)
  {
    return x.id_ > y.id_;
  }
  inline
  bool operator<=(const Symbol& x, const Symbol& y)
  {
    return x.id_ <= y.id_;
  }
  inline
  bool operator>=(const Symbol& x, const Symbol& y)
  {
    return x.id_ >= y.id_;
  }
};

namespace std
{
  inline
  void swap(trance::Symbol& x, trance::Symbol& y)
  {
    x.swap(y);
  }
};

namespace utils
{
  template <>
  struct traits<trance::Symbol>
  {
    typedef trance::Symbol      value_type;
    typedef value_type::id_type id_type;
    
    static inline value_type unassigned() { return value_type(id_type(-1)); }
    static inline value_type deleted() { return value_type(id_type(-2)); }
  };
};

#endif

