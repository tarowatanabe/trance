//
//  Copyright(C) 2014 Taro Watanabe <taro.watanabe@nict.go.jp>
//

#include <iterator>

#define BOOST_SPIRIT_THREADSAFE

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/spirit/include/phoenix.hpp>

#include <boost/xpressive/xpressive.hpp>
#include <boost/functional/hash/hash.hpp>

#include <utils/atomicop.hpp>
#include <utils/lexical_cast.hpp>
#include <utils/config.hpp>
#include <utils/thread_specific_ptr.hpp>
#include <utils/simple_vector.hpp>
#include <utils/array_power2.hpp>

#include "symbol.hpp"

namespace rnnp
{
  struct SymbolImpl
  {
    typedef Symbol::symbol_map_type symbol_map_type;
    typedef Symbol::id_type         id_type;
    typedef Symbol::mutex_type      mutex_type;
    
    typedef utils::indexed_set<id_type, boost::hash<id_type>, std::equal_to<id_type>, std::allocator<id_type> > non_terminal_set_type;

    typedef std::vector<bool, std::allocator<bool> > non_terminal_map_type;
    typedef std::vector<id_type, std::allocator<id_type> > non_terminal_id_map_type;
    typedef utils::simple_vector<id_type, std::allocator<id_type> > id_set_type;

    symbol_map_type              symbol_maps_;
    non_terminal_map_type        non_terminal_maps_;
    non_terminal_id_map_type     non_terminal_id_maps_;
  };
  
  Symbol::ticket_type    Symbol::__mutex;

  static SymbolImpl::mutex_type            __non_terminal_mutex;
  static SymbolImpl::non_terminal_set_type __non_terminal_map;

  namespace symbol_impl
  {
#ifdef HAVE_TLS
    static __thread SymbolImpl*                   impl_tls = 0;
    static utils::thread_specific_ptr<SymbolImpl> impl;
#else
    static utils::thread_specific_ptr<SymbolImpl> impl;
#endif
    
    static SymbolImpl& instance()
    {
#ifdef HAVE_TLS
      if (! impl_tls) {
	impl.reset(new SymbolImpl());
	impl_tls = impl.get();
      }
      
      return *impl_tls;
#else
      if (! impl.get())
	impl.reset(new SymbolImpl());
      
      return *impl;
#endif
    }
  };

  // constants
  const Symbol Symbol::EMPTY   = Symbol("");
  const Symbol Symbol::EPSILON = Symbol("<epsilon>");
  const Symbol Symbol::UNK     = Symbol("<unk>");
  
  Symbol::symbol_map_type& Symbol::__symbol_maps()
  {
    return symbol_impl::instance().symbol_maps_;
  }
    
  bool Symbol::non_terminal() const
  {
    SymbolImpl::non_terminal_map_type& maps =  symbol_impl::instance().non_terminal_maps_;
    
    const size_type scan_pos = (id_ << 1);
    const size_type flag_pos = (id_ << 1) + 1;
    
    if (flag_pos >= maps.size()) {
      const size_type size = flag_pos + 1;
      const size_type power2 = utils::bithack::branch(utils::bithack::is_power2(size),
						      size,
						      size_type(utils::bithack::next_largest_power2(size)));
      
      maps.reserve(power2);
      maps.resize(power2, false);
    }
    
    if (! maps[scan_pos]) {
      namespace qi = boost::spirit::qi;
      namespace standard = boost::spirit::standard;
      
      const symbol_type& word = symbol();
      
      symbol_type::const_iterator iter = word.begin();
      symbol_type::const_iterator iter_end = word.end();
      
      maps[scan_pos] = true;
      maps[flag_pos] = qi::parse(iter, iter_end, '[' >> +(standard::char_ - ',' - ']') >> -(',' >> qi::int_) >> ']') && iter == iter_end;
    }
    
    return maps[flag_pos];
  }
  
  Symbol::id_type Symbol::non_terminal_id() const
  {
    if (! non_terminal()) return id_type(-1);
    
    SymbolImpl::non_terminal_id_map_type& maps = symbol_impl::instance().non_terminal_id_maps_;

    if (id_ >= maps.size()) {
      const size_type size = id_ + 1;
      const size_type power2 = utils::bithack::branch(utils::bithack::is_power2(size),
						      size,
						      size_type(utils::bithack::next_largest_power2(size)));
      maps.reserve(power2);
      maps.resize(power2, id_type(-1));
    }
    
    if (maps[id_] == id_type(-1)) {
      mutex_type::scoped_lock lock(__non_terminal_mutex);
      
      SymbolImpl::non_terminal_set_type::iterator iter = __non_terminal_map.insert(id_).first;
      
      maps[id_] = iter - __non_terminal_map.begin();
    }
    return maps[id_];
  }
  
  bool Symbol::binarized() const
  {
    if (! non_terminal()) return false;
    
    return non_terminal_strip().find('^') != piece_type::npos();
  }

};
