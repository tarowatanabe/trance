// -*- mode: c++ -*-
//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#ifndef __TRANCE__FEATURE__GRAMMAR__HPP__
#define __TRANCE__FEATURE__GRAMMAR__HPP__ 1

#define BOOST_SPIRIT_THREADSAFE
#define PHOENIX_THREADSAFE

#include <boost/spirit/include/qi.hpp>

#include <trance/symbol.hpp>
#include <trance/rule.hpp>
#include <trance/signature.hpp>
#include <trance/option.hpp>
#include <trance/feature_function.hpp>

#include <utils/bithack.hpp>
#include <utils/unordered_map.hpp>
#include <utils/alloc_vector.hpp>
#include <utils/indexed_set.hpp>
#include <utils/chunk_vector.hpp>
#include <utils/mulvector2.hpp>
#include <utils/lexical_cast.hpp>
#include <utils/compress_stream.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/thread.hpp>

namespace trance
{
  namespace feature
  {
    
    struct GrammarImpl
    {
      typedef size_t    size_type;
      typedef ptrdiff_t difference_type;

      typedef trance::Symbol symbol_type;
      typedef trance::Symbol word_type;
      
      typedef trance::Rule      rule_type;
      typedef trance::Signature signature_type;

      typedef trance::FeatureFunction feature_function_type;
      
      typedef feature_function_type::parameter_type parameter_type;
      typedef feature_function_type::feature_type   feature_type;
      
      typedef std::vector<feature_type, std::allocator<feature_type> > name_set_type;
      
      typedef boost::filesystem::path path_type;
      
      struct preterminal_type
      {
	symbol_type lhs_;
	word_type   rhs_;

	preterminal_type() : lhs_(), rhs_() {}
	preterminal_type(const symbol_type& lhs, const word_type& rhs)
	  : lhs_(lhs), rhs_(rhs) {}

	friend
	bool operator==(const preterminal_type& x, const preterminal_type& y)
	{
	  return x.lhs_ == y.lhs_ && x.rhs_ == y.rhs_;
	}
	
	friend
	size_t hash_value(preterminal_type const& x)
	{
	  return utils::hashmurmur3<size_t>()(x.rhs_, x.lhs_.id());
	}
      };
      
      struct unary_type
      {
	symbol_type lhs_;
	symbol_type rhs_;
	
	unary_type() : lhs_(), rhs_() {}
	unary_type(const symbol_type& lhs, const symbol_type& rhs)
	  : lhs_(lhs), rhs_(rhs) {}

	friend
	bool operator==(const unary_type& x, const unary_type& y)
	{
	  return x.lhs_ == y.lhs_ && x.rhs_ == y.rhs_;
	}
	
	friend
	size_t hash_value(unary_type const& x)
	{
	  return utils::hashmurmur3<size_t>()(x.rhs_, x.lhs_.id());
	}
      };
      
      struct binary_type
      {
	symbol_type lhs_;
	symbol_type rhs0_;
	symbol_type rhs1_;
	
	binary_type() : lhs_(), rhs0_(), rhs1_() {}
	binary_type(const symbol_type& lhs, const symbol_type& rhs0, const symbol_type& rhs1)
	  : lhs_(lhs), rhs0_(rhs0), rhs1_(rhs1) {}

	friend
	bool operator==(const binary_type& x, const binary_type& y)
	{
	  return x.lhs_ == y.lhs_ && x.rhs0_ == y.rhs0_ && x.rhs1_ == y.rhs1_;
	}
	
	friend
	size_t hash_value(binary_type const& x)
	{
	  return utils::hashmurmur3<size_t>()(x.lhs_, (size_t(x.rhs0_.id()) << 16) + x.rhs1_.id());
	}
      };
      
      typedef utils::indexed_set<preterminal_type, boost::hash<preterminal_type>, std::equal_to<preterminal_type>,
				 std::allocator<preterminal_type> > preterminal_set_type;
      typedef utils::indexed_set<unary_type, boost::hash<unary_type>, std::equal_to<unary_type>,
				 std::allocator<unary_type> > unary_set_type;
      typedef utils::indexed_set<binary_type, boost::hash<binary_type>, std::equal_to<binary_type>,
				 std::allocator<binary_type> > binary_set_type;
      
      typedef utils::mulvector2<parameter_type, std::allocator<parameter_type> > feature_map_type;
      
      GrammarImpl(const path_type& path, const std::string& name)
      {
	namespace qi = boost::spirit::qi;
	namespace standard = boost::spirit::standard;

	typedef std::vector<parameter_type, std::allocator<parameter_type> > parameter_set_type;
	typedef boost::spirit::istream_iterator iterator_type;
	
	if (path != "-" && ! boost::filesystem::exists(path))
	  throw std::runtime_error("no file? " + path.string());
	
	utils::compress_istream is(path, 1024 * 1024);
	is.unsetf(std::ios::skipws);
	
	iterator_type iter(is);
	iterator_type iter_end;	

	std::string line;
	rule_type rule;

	parameter_set_type parameter;
	size_type parameter_max = 0;

	while (iter != iter_end) {
	  line.clear();
	  parameter.clear();
	  
	  if (! qi::parse(iter, iter_end,
			  *(standard::char_ - qi::eol - ("|||" >> (standard::space | qi::eoi)))
			  >> -(qi::omit["|||" >> +standard::blank] >> -(qi::double_ % (+standard::blank)))
			  >> (qi::eol | qi::eoi),
			  line,
			  parameter))
	    if (iter != iter_end)
	      throw std::runtime_error("parsing failed... " + std::string(iter, iter_end));

	  if (line.empty() || parameter.empty()) continue;
	  
	  rule.assign(line);
	  
	  if (rule.goal()) continue;
	  
	  parameter_max = utils::bithack::max(parameter_max, parameter.size());
	  
	  if (rule.unary()) {
	    if (unary_.insert(unary_type(rule.lhs_, rule.rhs_.front())).second)
	      features_unary_.push_back(parameter.begin(), parameter.end());
	  } else if (rule.binary()) {
	    if (binary_.insert(binary_type(rule.lhs_, rule.rhs_.front(), rule.rhs_.back())).second)
	      features_binary_.push_back(parameter.begin(), parameter.end());
	  } else if (rule.preterminal()) {
	    if (preterminal_.insert(preterminal_type(rule.lhs_, rule.rhs_.front())).second)
	      features_preterminal_.push_back(parameter.begin(), parameter.end());
	  } else
	    throw std::runtime_error("invlaid rule: " + rule.string());
	}
	
	names_.reserve(parameter_max);
	for (size_type id = 0; id != parameter_max; ++ id)
	  names_.push_back(name + ':' + utils::lexical_cast<std::string>(id));

	name_unary_       = name + ":unk-unary";
	name_binary_      = name + ":unk-binary";
	name_preterminal_ = name + ":unk-preterminal";
      }
      
      
    public:
      unary_set_type       unary_;
      binary_set_type      binary_;
      preterminal_set_type preterminal_;
      
      feature_map_type     features_unary_;
      feature_map_type     features_binary_;
      feature_map_type     features_preterminal_;
      
      name_set_type names_;

      feature_type name_unary_;
      feature_type name_binary_;
      feature_type name_preterminal_;
    };
    
    class Grammar : public trance::FeatureFunction
    {
    public:
      typedef trance::Signature signature_type;
      
      typedef signature_type::signature_ptr_type signature_ptr_type;
      
      typedef boost::filesystem::path path_type;

      typedef GrammarImpl impl_type;

    public:
      Grammar(const std::string& name, const impl_type& impl, const signature_ptr_type& signature)
	: FeatureFunction(name, sizeof(symbol_type)),
	  pimpl_(&impl),
	  signature_(signature->clone()) { }
      
    public:
      // cloning
      virtual feature_function_ptr_type clone() const
      {
	std::auto_ptr<Grammar> grammar(new Grammar(*this));
	
	if (grammar->signature_)
	  grammar->signature_ = grammar->signature_->clone();
	
	return feature_function_ptr_type(grammar.release());
      }
      
      
      // feature application for axiom
      virtual void apply(const operation_type& operation,
			 state_type state,
			 feature_vector_type& features) const
      {
	
	*reinterpret_cast<symbol_type*>(state) = symbol_type::EPSILON;
      }
      
      // feature application for shift
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const word_type& head,
			 state_type state,
			 feature_vector_type& features) const
      {
	impl_type::preterminal_set_type::const_iterator iter = pimpl_->preterminal_.find(impl_type::preterminal_type(label, head));
	if (iter == pimpl_->preterminal_.end()) {
	  iter = pimpl_->preterminal_.find(impl_type::preterminal_type(label, signature_->operator()(head)));

	  if (iter == pimpl_->preterminal_.end())
	    iter = pimpl_->preterminal_.find(impl_type::preterminal_type(label, symbol_type::UNK));
	}
	
	if (iter != pimpl_->preterminal_.end()) {
	  const size_type id =  iter - pimpl_->preterminal_.begin();

	  typedef impl_type::feature_map_type::const_reference feature_set_type;
	  
	  impl_type::name_set_type::const_iterator niter = pimpl_->names_.begin();
	  feature_set_type::const_iterator fiter_end = pimpl_->features_preterminal_[id].end();
	  for (feature_set_type::const_iterator fiter = pimpl_->features_preterminal_[id].begin(); fiter != fiter_end; ++ fiter, ++ niter)
	    features[*niter] = *fiter;
	} else
	  features[pimpl_->name_preterminal_] = -1;
	
	*reinterpret_cast<symbol_type*>(state) = label;
      }
      
      // feature application for reduce
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const state_type state_top,
			 const state_type state_next,
			 state_type state,
			 feature_vector_type& features) const
      {
	const symbol_type& rhs0 = *reinterpret_cast<symbol_type*>(state_next);
	const symbol_type& rhs1 = *reinterpret_cast<symbol_type*>(state_top);
	
	impl_type::binary_set_type::const_iterator iter = pimpl_->binary_.find(impl_type::binary_type(label, rhs0, rhs1));
	
	if (iter != pimpl_->binary_.end()) {
	  const size_type id =  iter - pimpl_->binary_.begin();

	  typedef impl_type::feature_map_type::const_reference feature_set_type;
	  
	  impl_type::name_set_type::const_iterator niter = pimpl_->names_.begin();
	  feature_set_type::const_iterator fiter_end = pimpl_->features_binary_[id].end();
	  for (feature_set_type::const_iterator fiter = pimpl_->features_binary_[id].begin(); fiter != fiter_end; ++ fiter, ++ niter)
	    features[*niter] = *fiter;
	} else
	  features[pimpl_->name_binary_] = -1;
	
	*reinterpret_cast<symbol_type*>(state) = label;
      }
      
      // feature application for unary
      virtual void apply(const operation_type& operation,
			 const symbol_type& label,
			 const state_type state_top,
			 state_type state,
			 feature_vector_type& features) const
      {
	const symbol_type& rhs = *reinterpret_cast<symbol_type*>(state_top);
	
	impl_type::unary_set_type::const_iterator iter = pimpl_->unary_.find(impl_type::unary_type(label, rhs));
	
	if (iter != pimpl_->unary_.end()) {
	  const size_type id =  iter - pimpl_->unary_.begin();

	  typedef impl_type::feature_map_type::const_reference feature_set_type;
	  
	  impl_type::name_set_type::const_iterator niter = pimpl_->names_.begin();
	  feature_set_type::const_iterator fiter_end = pimpl_->features_unary_[id].end();
	  for (feature_set_type::const_iterator fiter = pimpl_->features_unary_[id].begin(); fiter != fiter_end; ++ fiter, ++ niter)
	    features[*niter] = *fiter;
	} else
	  features[pimpl_->name_unary_] = -1;
	
	*reinterpret_cast<symbol_type*>(state) = label;
      }
      
      // feature application for final/idle
      virtual void apply(const operation_type& operation,
			 const state_type state_top,
			 state_type state,
			 feature_vector_type& features) const
      {
	if (operation.final())
	  *reinterpret_cast<symbol_type*>(state) = symbol_type::FINAL;
	else if (operation.idle())
	  *reinterpret_cast<symbol_type*>(state) = symbol_type::IDLE;
	else
	  throw std::runtime_error("invalid operation");
      }
      
    private:
      const impl_type* pimpl_;
      signature_ptr_type signature_;
    };
    
    struct FactoryGrammar : public trance::FeatureFunctionFactory
    {
      feature_function_ptr_type create(const std::string& param) const
      {
	typedef trance::Option option_type;
	
	typedef boost::filesystem::path path_type;
	
	typedef boost::mutex            mutex_type;
	typedef mutex_type::scoped_lock lock_type;

	
	typedef GrammarImpl impl_type;
	typedef boost::shared_ptr<impl_type> impl_ptr_type;

	typedef trance::Signature signature_type;
	
	typedef utils::unordered_map<std::string, impl_ptr_type,
				     boost::hash<utils::piece>, std::equal_to<std::string>,
				     std::allocator<std::pair<const std::string, impl_ptr_type> > >::type grammar_map_type;
	
	static mutex_type       mutex;
	static grammar_map_type grammars;
	
	const option_type option(param);
	
	path_type path;
	std::string sig = "none";
	std::string name = "grammar";
	
	if (utils::ipiece(option.name()) != "grammar")
	  throw std::runtime_error("is this really grammar feature function? " + param);
	
	for (option_type::const_iterator oiter = option.begin(); oiter != option.end(); ++ oiter) {
	  if (utils::ipiece(oiter->first) == "file")
	    path = oiter->second;
	  else if (utils::ipiece(oiter->first) == "signature")
	    sig = oiter->second;
	  else if (utils::ipiece(oiter->first) == "name")
	    name = oiter->second;
	  else
	    throw std::runtime_error("invalid options for phrase: " + param);
	}
	
	lock_type lock(mutex);
	
	grammar_map_type::iterator iter = grammars.find(path.string() + ':' + name);
	if (iter == grammars.end())
	  iter = grammars.insert(std::make_pair(path.string() + ':' + name,
						impl_ptr_type(new impl_type(path.string(), name)))).first;
	
	return feature_function_ptr_type(new Grammar(name, *(iter->second.get()), signature_type::create(sig)));
      }
      
      bool supported(const std::string& param) const
      {
	typedef trance::Option option_type;
	
	const option_type option(param);
	
	return utils::ipiece(option.name()) == "grammar";
      }

      std::string usage() const
      {
	static const char* desc = "\
grammar: features from grammar\n\
\tfile=<file name>\n\
\tsignature=<word signature> (default: none)\n\
\tname=feature-name (default: phrase)\n\
";
	return desc;
      }
    };
  }
}
#endif
