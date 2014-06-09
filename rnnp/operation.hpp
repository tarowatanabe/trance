// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__OPERATION__HPP__
#define __RNNP__OPERATION__HPP__ 1

#include <stdint.h>
#include <stddef.h>

#include <iostream>

namespace rnnp
{
  struct Operation
  {
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef uint32_t  index_type;
    
    typedef enum {
      AXIOM,
      SHIFT,
      REDUCE,
      REDUCE_LEFT,
      REDUCE_RIGHT,
      UNARY,
      FINAL,
      IDLE
    } operation_type;
    
    Operation(const operation_type& op, const size_type& closure) : operation_(op | (closure << 8)) {}
    Operation(const operation_type& op) : operation_(op) {}
    Operation(const index_type& op) : operation_(op) {}
    
    operation_type operation() const { return operation_type(operation_ & 0xff); }
    size_type closure() const { return operation_ >> 8; }

    bool axiom() const { return operation() == AXIOM; }
    bool shift() const { return operation() == SHIFT; }
    bool reduce() const { return operation() == REDUCE || operation() == REDUCE_LEFT || operation() == REDUCE_RIGHT; }
    bool unary() const { return operation() == UNARY; }
    bool final() const { return operation() == FINAL; }
    bool idle() const { return operation() == IDLE; }

    bool reduce_left()  const { return operation() == REDUCE_LEFT; }
    bool reduce_right() const { return operation() == REDUCE_RIGHT; }
    
    bool finished() const { return operation() == FINAL || operation() == IDLE; }

    friend
    std::ostream& operator<<(std::ostream& os, const Operation& x);
    
    friend
    bool operator==(const Operation& x, const Operation& y)
    {
      return x.operation_ == y.operation_;
    }

    friend
    bool operator!=(const Operation& x, const Operation& y)
    {
      return x.operation_ != y.operation_;
    }
    
  public:
    index_type operation_;
  };
};


#endif
