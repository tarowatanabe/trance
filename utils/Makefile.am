
AM_CPPFLAGS = -I$(top_builddir) $(BOOST_CPPFLAGS) $(TCMALLOC_CPPFLAGS) $(PTHREAD_FLAGS) $(JEMALLOC_CPPFLAGS)

# perftools_LDADD = $(JEMALLOC_LDFLAGS) $(TCMALLOC_LDFLAGS) $(PROFILER_LDFLAGS)

LIBUTILS=libutils.la

noinst_LTLIBRARIES = libutils.la

libutils_la_SOURCES = \
	malloc_stats.cpp \
	tempfile.cpp

libutils_la_LDFLAGS = -version-info $(RNNP_LTVERSION)

libutils_la_LIBADD = \
	$(BOOST_THREAD_LDFLAGS) $(BOOST_THREAD_LIBS) \
	$(BOOST_IOSTREAMS_LDFLAGS) $(BOOST_IOSTREAMS_LIBS) \
	$(BOOST_FILESYSTEM_LDFLAGS) $(BOOST_FILESYSTEM_LIBS) \
	$(JEMALLOC_LDFLAGS) \
	$(TCMALLOC_LDFLAGS)

libutilsincludedir = $(pkgincludedir)/utils

libutilsinclude_HEADERS = \
aligned_allocator.hpp \
alloc_vector.hpp \
allocinfo_allocator.hpp \
array_power2.hpp \
atomicop.hpp \
bithack.hpp \
byte_aligned_code.hpp \
c_string_generator.hpp \
c_string_parser.hpp \
chunk_vector.hpp \
compact_func.hpp \
compact_hashtable.hpp \
compact_map.hpp \
compact_set.hpp \
compress_stream.hpp \
config.hpp \
filesystem.hpp \
getline.hpp \
lexical_cast.hpp \
hashmurmur.hpp \
hashmurmur3.hpp \
indexed_hashtable.hpp \
indexed_map.hpp \
indexed_set.hpp \
lockfree_list_queue.hpp \
malloc_stats.hpp \
memory.hpp \
mpi.hpp \
mpi_allocator.hpp \
mpi_device.hpp \
mpi_device_bcast.hpp \
mpi_stream.hpp \
mpi_stream_simple.hpp \
mpi_traits.hpp \
mulvector2.hpp \
piece.hpp \
program_options.hpp \
random_seed.hpp \
repository.hpp \
rwticket.hpp \
simple_vector.hpp \
small_vector.hpp \
space_separator.hpp \
spinlock.hpp \
tempfile.hpp \
thread_specific_ptr.hpp \
traits.hpp \
trie_compact.hpp \
unordered_map.hpp \
unordered_set.hpp \
vector_map.hpp \
vector_set.hpp 

nodist_libutilsinclude_HEADERS = \
config.h

IGNORES = \
config.h \
config.h.in \
stamp-h1

noinst_DATA = .gitignore

.gitignore: Makefile.am
	cat $(top_builddir)/.gitignore > .gitignore
	for bin in $(PROGRAMS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SCRIPTS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(IGNORES); do \
	  echo $$bin >> .gitignore; \
	done
