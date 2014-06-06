// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __RNNP__LEARN_OPTION__HPP__
#define __RNNP__LEARN_OPTION__HPP__ 1

#include <string>
#include <iostream>

namespace rnnp
{
  struct LearnOption
  {
    typedef enum {
      LEARN_NONE           = 0,
      LEARN_CLASSIFICATION = 1,
      LEARN_EMBEDDING      = 1 << 1,
      LEARN_HIDDEN         = 1 << 2,
      LEARN_MODEL          = (1 << 1) | (1 << 2),
      LEARN_ALL            = 1 | (1 << 1) | (1 << 2),
    } learn_type;

    typedef enum {
      OPTIMIZE_SGD,
      OPTIMIZE_ADAGRAD,
      OPTIMIZE_ADADEC,
      OPTIMIZE_ADADELTA,
    } optimize_type;

    typedef enum {
      MARGIN_DERIVATION,
      MARGIN_EVALB,
      MARGIN_EARLY,
      MARGIN_LATE,
      MARGIN_MAX,
      VIOLATION_EARLY,
      VIOLATION_LATE,
      VIOLATION_MAX,
    } objective_type;

    LearnOption();
    LearnOption(const std::string& param);

    bool learn_classification() const { return learn_ & LEARN_CLASSIFICATION; }
    bool learn_embedding()      const { return learn_ & LEARN_EMBEDDING; }
    bool learn_hidden()         const { return learn_ & LEARN_HIDDEN; }

    bool optimize_sgd()      const { return optimize_ == OPTIMIZE_SGD; }
    bool optimize_adagrad()  const { return optimize_ == OPTIMIZE_ADAGRAD; }
    bool optimize_adadec()   const { return optimize_ == OPTIMIZE_ADADEC; }
    bool optimize_adadelta() const { return optimize_ == OPTIMIZE_ADADELTA; }
    
    bool margin_derivation() const { return objective_ == MARGIN_DERIVATION; }
    bool margin_evalb()      const { return objective_ == MARGIN_EVALB; }
    bool margin_early()      const { return objective_ == MARGIN_EARLY; }
    bool margin_late()       const { return objective_ == MARGIN_LATE; }
    bool margin_max()        const { return objective_ == MARGIN_MAX; }
    bool violation_early()   const { return objective_ == VIOLATION_EARLY; }
    bool violation_late()    const { return objective_ == VIOLATION_LATE; }
    bool violation_max()     const { return objective_ == VIOLATION_MAX; }
    
  public:
    friend
    std::ostream& operator<<(std::ostream& os, const LearnOption& x);

  public:
    int            learn_;
    optimize_type  optimize_;
    objective_type objective_;
    
    int    iteration_;
    int    batch_;
    double lambda_;
    double eta0_;
    bool   shrinking_;
    bool   decay_;

    // other parameters
    double scale_;
  };
};

#endif
