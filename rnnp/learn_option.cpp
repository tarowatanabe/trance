//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <stdexcept>

#include "learn_option.hpp"

#include "option.hpp"

#include "utils/lexical_cast.hpp"
#include "utils/piece.hpp"

namespace rnnp
{
  LearnOption::LearnOption()
    : learn_(LEARN_ALL),
      optimize_(OPTIMIZE_SGD),
      objective_(MARGIN_DERIVATION),
      iteration_(10),
      batch_(4),
      lambda_(0),
      eta0_(0.01),
      shrinking_(false),
      decay_(false),
      scale_(0)
  {
    
  }
  
  LearnOption::LearnOption(const std::string& param)
    : learn_(LEARN_NONE),
      optimize_(OPTIMIZE_SGD),
      objective_(MARGIN_DERIVATION),
      iteration_(10),
      batch_(4),
      lambda_(0),
      eta0_(0.01),
      shrinking_(false),
      decay_(false),
      scale_(0)
  {
    typedef rnnp::Option option_type;
    
    const option_type option(param);
    
    utils::ipiece learn = option.name();
    
    if (learn == "classification" || learn == "class")
      learn_ = LEARN_CLASSIFICATION;
    else if (learn == "embedding" || learn == "embed")
      learn_ = LEARN_EMBEDDING;
    else if (learn == "head")
      learn_ = LEARN_HEAD;
    else if (learn == "hidden")
      learn_ = LEARN_HIDDEN;
    else if (learn == "model")
      learn_ = LEARN_MODEL;
    else if (learn == "all")
      learn_ = LEARN_ALL;
    else
      throw std::runtime_error("invalid learning");

    for (option_type::const_iterator oiter = option.begin(); oiter != option.end(); ++ oiter) {
      if (utils::ipiece(oiter->first) == "learn") {
	if (utils::ipiece(oiter->second) == "classification" || utils::ipiece(oiter->second) == "class")
	  learn_ |= LEARN_CLASSIFICATION;
	else if (utils::ipiece(oiter->second) == "embedding" || utils::ipiece(oiter->second) == "embed")
	  learn_ |= LEARN_EMBEDDING;
	else if (utils::ipiece(oiter->second) == "head")
	  learn_ |= LEARN_HEAD;
	else if (utils::ipiece(oiter->second) == "hidden")
	  learn_ |= LEARN_HIDDEN;
	else if (utils::ipiece(oiter->second) == "model")
	  learn_ |= LEARN_MODEL;
	else
	  throw std::runtime_error("invalid additional learning");
      } else if (utils::ipiece(oiter->first) == "optimize" || utils::ipiece(oiter->first) == "opt") {
	if (utils::ipiece(oiter->second) == "sgd")
	  optimize_ = OPTIMIZE_SGD;
	else if (utils::ipiece(oiter->second) == "adagrad")
	  optimize_ = OPTIMIZE_ADAGRAD;
	else if (utils::ipiece(oiter->second) == "adadec")
	  optimize_ = OPTIMIZE_ADADEC;
	else if (utils::ipiece(oiter->second) == "adadelta")
	  optimize_ = OPTIMIZE_ADADELTA;
	else
	  throw std::runtime_error("unsupported optimization algorithm: " + oiter->second);
      } else if (utils::ipiece(oiter->first) == "margin") {
	if (utils::ipiece(oiter->second) == "derivation")
	  objective_ = MARGIN_DERIVATION;
	else if (utils::ipiece(oiter->second) == "evalb")
	  objective_ = MARGIN_EVALB;
	else if (utils::ipiece(oiter->second) == "early")
	  objective_ = MARGIN_EARLY;
	else if (utils::ipiece(oiter->second) == "late")
	  objective_ = MARGIN_LATE;
	else if (utils::ipiece(oiter->second) == "max")
	  objective_ = MARGIN_MAX;
	else
	  throw std::runtime_error("unsupported margin: " + oiter->second);
      } else if (utils::ipiece(oiter->first) == "violation") {
	if (utils::ipiece(oiter->second) == "early")
	  objective_ = VIOLATION_EARLY;
	else if (utils::ipiece(oiter->second) == "late")
	  objective_ = VIOLATION_LATE;
	else if (utils::ipiece(oiter->second) == "max")
	  objective_ = VIOLATION_MAX;
	else
	  throw std::runtime_error("unsupported violation: " + oiter->second);
      } else if (utils::ipiece(oiter->first) == "iteration" || utils::ipiece(oiter->first) == "iter")
	iteration_ = utils::lexical_cast<int>(oiter->second);
      else if (utils::ipiece(oiter->first) == "batch" || utils::ipiece(oiter->first) == "batch-size")
	batch_ = utils::lexical_cast<int>(oiter->second);
      else if (utils::ipiece(oiter->first) == "lambda")
	lambda_ = utils::lexical_cast<double>(oiter->second);
      else if (utils::ipiece(oiter->first) == "eta0" || utils::ipiece(oiter->first) == "eta")
	eta0_ = utils::lexical_cast<double>(oiter->second);
      else if (utils::ipiece(oiter->first) == "shrinking" || utils::ipiece(oiter->first) == "shrink")
	shrinking_ = utils::lexical_cast<bool>(oiter->second);
      else if (utils::ipiece(oiter->first) == "decay")
	decay_ = utils::lexical_cast<bool>(oiter->second);
      else if (utils::ipiece(oiter->first) == "scale")
	scale_ = utils::lexical_cast<double>(oiter->second);
      else
	throw std::runtime_error("unsupported optimizatin parameters: " + param);
    }
    
    if (iteration_ <= 0)
      throw std::runtime_error("zero or negative iterations?");
    if (batch_ <= 0)
      throw std::runtime_error("zero or negative mini batch size?");
    if (lambda_ < 0.0)
      throw std::runtime_error("negative lambda?");
    if (eta0_ <= 0.0)
      throw std::runtime_error("zero or negative eta0?");

    if (margin_evalb()) {
      if (scale_ == 0)
	scale_ = 2;
      
      if (scale_ <= 0)
	throw std::runtime_error("scaling must be positive");
    }
  }

  std::ostream& operator<<(std::ostream& os, const LearnOption& option)
  {
    typedef rnnp::Option option_type;
    
    option_type opt;

    LearnOption::learn_type learn;

    if (option.learn_ == LearnOption::LEARN_ALL) {
      opt.name() = "all";
      learn = LearnOption::LEARN_ALL;
    } else if (option.learn_ == LearnOption::LEARN_MODEL) {
      opt.name() = "model";
      learn = LearnOption::LEARN_MODEL;
    } else if (option.learn_classification()) {
      opt.name() = "classification";
      learn = LearnOption::LEARN_CLASSIFICATION;
    } else if (option.learn_embedding()) {
      opt.name() = "embedding";
      learn = LearnOption::LEARN_EMBEDDING;
    } else if (option.learn_head()) {
      opt.name() = "head";
      learn = LearnOption::LEARN_HEAD;
    } else if (option.learn_hidden()) {
      opt.name() = "hidden";
      learn = LearnOption::LEARN_HIDDEN;
    }

    if (option.learn_ != LearnOption::LEARN_ALL && option.learn_ != LearnOption::LEARN_MODEL) {
      if (option.learn_classification() && learn != LearnOption::LEARN_CLASSIFICATION)
	opt.push_back(std::make_pair("learn", "classification"));
      if (option.learn_embedding() && learn != LearnOption::LEARN_EMBEDDING)
	opt.push_back(std::make_pair("learn", "embedding"));
      if (option.learn_head() && learn != LearnOption::LEARN_HEAD)
	opt.push_back(std::make_pair("learn", "head"));
      if (option.learn_hidden() && learn != LearnOption::LEARN_HIDDEN)
	opt.push_back(std::make_pair("learn", "hidden"));
    }
    
    if (option.optimize_sgd())
      opt.push_back(std::make_pair("optimize", "sgd"));
    else if (option.optimize_adagrad())
      opt.push_back(std::make_pair("optimize", "adagrad"));
    else if (option.optimize_adadec())
      opt.push_back(std::make_pair("optimize", "adadec"));
    else if (option.optimize_adadelta())
      opt.push_back(std::make_pair("optimize", "adadelta"));
    
    if (option.margin_derivation())
      opt.push_back(std::make_pair("margin", "derivation"));
    else if (option.margin_evalb())
      opt.push_back(std::make_pair("margin", "evalb"));
    else if (option.margin_early())
      opt.push_back(std::make_pair("margin", "early"));
    else if (option.margin_late())
      opt.push_back(std::make_pair("margin", "late"));
    else if (option.margin_max())
      opt.push_back(std::make_pair("margin", "max"));
    else if (option.violation_early())
      opt.push_back(std::make_pair("violation", "early"));
    else if (option.violation_late())
      opt.push_back(std::make_pair("violation", "late"));
    else if (option.violation_max())
      opt.push_back(std::make_pair("violation", "max"));
    
    opt.push_back(std::make_pair("iteration", utils::lexical_cast<std::string>(option.iteration_)));
    
    opt.push_back(std::make_pair("batch", utils::lexical_cast<std::string>(option.batch_)));
    
    if (option.lambda_ > 0)
      opt.push_back(std::make_pair("lambda", utils::lexical_cast<std::string>(option.lambda_)));
    
    if (option.eta0_ > 0)
      opt.push_back(std::make_pair("eta0", utils::lexical_cast<std::string>(option.eta0_)));

    if (option.shrinking_)
      opt.push_back(std::make_pair("shrinking", utils::lexical_cast<std::string>(option.shrinking_)));
    
    if (option.decay_)
      opt.push_back(std::make_pair("decay", utils::lexical_cast<std::string>(option.decay_)));

    if (option.scale_ > 0)
      opt.push_back(std::make_pair("scale", utils::lexical_cast<std::string>(option.scale_)));

    os << opt;
    
    return os;
  }
  
};
