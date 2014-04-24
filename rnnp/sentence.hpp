// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__SENTENCE__HPP__
#define __RNNP__SENTENCE__HPP__ 1

#include <iostream>
#include <vector>
#include <string>

#include <rnnp/symbol.hpp>

#include <utils/hashmurmur3.hpp>
#include <utils/piece.hpp>

namespace rnnp
{

  class Sentence
  {
  public:
    typedef rnnp::Symbol  symbol_type;
    typedef rnnp::Symbol  word_type;
    
  private:
    typedef std::vector<word_type, std::allocator<word_type> > sent_type;

  public:
    typedef sent_type::size_type              size_type;
    typedef sent_type::difference_type        difference_type;
    typedef sent_type::value_type             value_type;
    
    typedef sent_type::iterator               iterator;
    typedef sent_type::const_iterator         const_iterator;
    typedef sent_type::reverse_iterator       reverse_iterator;
    typedef sent_type::const_reverse_iterator const_reverse_iterator;
    typedef sent_type::reference              reference;
    typedef sent_type::const_reference        const_reference;
    
  public:
    Sentence() : sent_() {}
    Sentence(size_type __n) : sent_(__n) {}
    Sentence(size_type __n, const word_type& __word) : sent_(__n, __word) {}
    template <typename Iterator>
    Sentence(Iterator first, Iterator last) : sent_(first, last) {}
    Sentence(const utils::piece& x) { assign(x); }
    
    void assign(size_type __n, const word_type& __word) { sent_.assign(__n, __word); }
    template <typename Iterator>
    void assign(Iterator first, Iterator last) { sent_.assign(first, last); }
    void assign(const utils::piece& x);
    
    bool assign(std::string::const_iterator& iter, std::string::const_iterator end);
    bool assign(utils::piece::const_iterator& iter, utils::piece::const_iterator end);
    
    // insert/erase...
    iterator insert(iterator pos, const word_type& word) { return sent_.insert(pos, word); }
    void insert(iterator pos, size_type n, const word_type& word) { sent_.insert(pos, n, word); }
    template <typename Iterator>
    void insert(iterator pos, Iterator first, Iterator last) { sent_.insert(pos, first, last); }
    
    iterator erase(iterator pos) { return sent_.erase(pos); }
    iterator erase(iterator first, iterator last) { return sent_.erase(first, last); }
    
    void push_back(const word_type& word) { sent_.push_back(word); }
    void pop_back() { sent_.pop_back(); }
    
    void swap(Sentence& __x) { sent_.swap(__x.sent_); }
    
    void clear() { sent_.clear(); }
    void reserve(size_type __n) { sent_.reserve(__n); }
    void resize(size_type __n) { sent_.resize(__n); }
    void resize(size_type __n, const word_type& __word) { sent_.resize(__n, __word); }

    size_type size() const { return sent_.size(); }
    bool empty() const { return sent_.empty(); }
    
    inline const_iterator begin() const { return sent_.begin(); }
    inline       iterator begin()       { return sent_.begin(); }
    
    inline const_iterator end() const { return sent_.end(); }
    inline       iterator end()       { return sent_.end(); }
    
    inline const_reverse_iterator rbegin() const { return sent_.rbegin(); }
    inline       reverse_iterator rbegin()       { return sent_.rbegin(); }
    
    inline const_reverse_iterator rend() const { return sent_.rend(); }
    inline       reverse_iterator rend()       { return sent_.rend(); }
    
    inline const_reference operator[](size_type pos) const { return sent_[pos]; }
    inline       reference operator[](size_type pos)       { return sent_[pos]; }

    inline const_reference front() const { return sent_.front(); }
    inline       reference front()       { return sent_.front(); }
    
    inline const_reference back() const { return sent_.back(); }
    inline       reference back()       { return sent_.back(); }
    
  public:
    
    friend
    size_t hash_value(Sentence const& x);
    
    friend
    std::ostream& operator<<(std::ostream& os, const Sentence& x);
    friend
    std::istream& operator>>(std::istream& is, Sentence& x);
    
    friend
    bool operator==(const Sentence& x, const Sentence& y);
    friend
    bool operator!=(const Sentence& x, const Sentence& y);
    friend
    bool operator<(const Sentence& x, const Sentence& y);
    friend
    bool operator>(const Sentence& x, const Sentence& y);
    friend
    bool operator<=(const Sentence& x, const Sentence& y);
    friend
    bool operator>=(const Sentence& x, const Sentence& y);
    
  private:
    sent_type sent_;
  };
  
  inline
  size_t hash_value(Sentence const& x) { return utils::hashmurmur3<size_t>()(x.sent_.begin(), x.sent_.end(), 0); }
  
  inline
  bool operator==(const Sentence& x, const Sentence& y) { return x.sent_ == y.sent_; }
  inline
  bool operator!=(const Sentence& x, const Sentence& y) { return x.sent_ != y.sent_; }
  inline
  bool operator<(const Sentence& x, const Sentence& y) { return x.sent_ < y.sent_; }
  inline
  bool operator>(const Sentence& x, const Sentence& y) { return x.sent_ > y.sent_; }
  inline
  bool operator<=(const Sentence& x, const Sentence& y) { return x.sent_ <= y.sent_; }
  inline
  bool operator>=(const Sentence& x, const Sentence& y) { return x.sent_ >= y.sent_; }
  
  
  
};

namespace std
{
  inline
  void swap(rnnp::Sentence& x, rnnp::Sentence& y)
  {
    x.swap(y);
  }
};

#endif
