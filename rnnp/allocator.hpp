// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__ALLOCATOR__HPP__
#define __RNNP__ALLOCATOR__HPP__ 1

#include <memory>

#include <utils/simple_vector.hpp>

namespace rnnp
{

  template <typename Tp>
  class Allocator : public std::allocator<char>
  {
  public:
    typedef Tp         state_type;
    typedef char*      pointer;
    typedef size_t     size_type;
    typedef ptrdiff_t  difference_type;
  
    typedef std::allocator<char> allocator_type;
    typedef utils::simple_vector<pointer, std::allocator<pointer> > state_set_type;
  
  public:
    static const size_type chunk_size  = 1024 * 4;
    static const size_type chunk_mask  = chunk_size - 1;
    
  public:

    Allocator()
      : states_(),
	state_iterator_(0),
	cache_(0),
	state_size_(0),
	state_alloc_size_(0),
	state_chunk_size_(0) {}
  
    Allocator(size_type __state_size)
      : states_(), state_iterator_(0),
	cache_(0),
	state_size_(__state_size),
	state_alloc_size_(0),
	state_chunk_size_(0)
    {
      if (state_size_ != 0) {
	// sizeof(char*) aligned size..
	const size_type pointer_size = sizeof(pointer);
	const size_type pointer_mask = ~(pointer_size - 1);
      
	state_alloc_size_ = (state_size_ + pointer_size - 1) & pointer_mask;
	state_chunk_size_ = state_alloc_size_ * chunk_size;
      }
    }
  
    Allocator(const Allocator& x) 
      : allocator_type(static_cast<const allocator_type&>(x)),
	states_(), state_iterator_(0),
	cache_(0),
	state_size_(x.state_size_),
	state_alloc_size_(x.state_alloc_size_),
	state_chunk_size_(x.state_chunk_size_) {}
  
    Allocator& operator=(const Allocator& x)
    { 
      clear();
      
      static_cast<allocator_type&>(*this) = static_cast<const allocator_type&>(x);
      
      state_size_ = x.state_size_;
      state_alloc_size_ = x.state_alloc_size_;
      state_chunk_size_ = x.state_chunk_size_;

      return *this;
    }
  
    ~Allocator() { clear(); }
  
  public:
    void swap(Allocator& x)
    {
      states_.swap(x.states_);
      std::swap(state_iterator_, x.state_iterator_);
      std::swap(cache_, x.cache_);
      
      std::swap(state_size_,       x.state_size_);
      std::swap(state_alloc_size_, x.state_alloc_size_);
      std::swap(state_chunk_size_, x.state_chunk_size_);
    }

    state_type allocate()
    {
      if (state_size_ == 0) return 0;
    
      if (cache_) {
	pointer state = cache_;
	cache_ = *reinterpret_cast<pointer*>(state);
            
	return state_type(state);
      }
    
      const size_type chunk_pos = state_iterator_ & chunk_mask;
    
      if (chunk_pos == 0)
	states_.push_back(allocator_type::allocate(state_chunk_size_));
    
      ++ state_iterator_;
    
      return state_type(states_.back() + chunk_pos * state_alloc_size_);
    }
  
    void deallocate(const state_type& state)
    {
      if (state.empty() || state_size_ == 0 || states_.empty()) return;
    
      *reinterpret_cast<pointer*>(const_cast<state_type&>(state).buffer_) = cache_;
      cache_ = state.buffer_;
    }
  
    state_type clone(const state_type& state)
    {
      if (state_size_ == 0)
	return state_type();
      
      state_type state_new = allocate();
      
      std::copy(state.buffer_, state.buffer_ + state_alloc_size_, state_new.buffer_);
      
      return state_new;
    }

    void assign(size_type __state_size)
    {
      if (state_size_ != __state_size)
	*this = Allocator(__state_size);
      else
	clear();
    }
  
    void clear()
    {
      typename state_set_type::iterator siter_end = states_.end();
      for (typename state_set_type::iterator siter = states_.begin(); siter != siter_end; ++ siter)
	allocator_type::deallocate(*siter, state_chunk_size_);
      
      states_.clear();
      state_iterator_ = 0;
      cache_ = 0;
    }
  
  private:  
    state_set_type states_;
    size_type state_iterator_;
    pointer   cache_;

    size_type state_size_;
    size_type state_alloc_size_;
    size_type state_chunk_size_;
  };
};

namespace std
{
  template <typename T>
  inline
  void swap(rnnp::Allocator<T>& x, rnnp::Allocator<T>& y)
  {
    x.swap(y);
  }
};

#endif
