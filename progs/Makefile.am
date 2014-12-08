AM_CPPFLAGS = -I.. -I../eigen $(BOOST_CPPFLAGS) $(PTHREAD_FLAGS) $(TCMALLOC_CPPFLAGS) $(PROFILER_CPPFLAGS)

perftools_LDADD = $(JEMALLOC_LDFLAGS) $(TCMALLOC_LDFLAGS) $(PROFILER_LDFLAGS)

boost_LDADD = \
	$(BOOST_THREAD_LDFLAGS) $(BOOST_THREAD_LIBS) \
	$(BOOST_PROGRAM_OPTIONS_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIBS) \
	$(BOOST_FILESYSTEM_LDFLAGS) $(BOOST_FILESYSTEM_LIBS) \
	$(BOOST_IOSTREAMS_LDFLAGS) $(BOOST_IOSTREAMS_LIBS) 

LIBTRANCE   = $(top_builddir)/trance/libtrance.la
LIBUTILS = $(top_builddir)/utils/libutils.la
LIBCODEC = $(top_builddir)/codec/libcodec.la

bin_PROGRAMS = \
	trance_grammar \
	trance_graphviz \
	trance_learn \
	trance_learn_mpi \
	trance_parse \
	trance_oracle \
	trance_treebank

trance_grammar_SOURCES = trance_grammar.cpp
trance_grammar_LDADD   = $(LIBTRANCE) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

trance_graphviz_SOURCES = trance_graphviz.cpp
trance_graphviz_LDADD   = $(LIBTRANCE) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

trance_learn_SOURCES = trance_learn.cpp
trance_learn_LDADD   = $(LIBTRANCE) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

trance_parse_SOURCES = trance_parse.cpp
trance_parse_LDADD   = $(LIBTRANCE) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

trance_learn_mpi_SOURCES = trance_learn_mpi.cpp
trance_learn_mpi_CPPFLAGS = $(MPI_CPPFLAGS) $(AM_CPPFLAGS)
trance_learn_mpi_LDADD   = $(MPI_LDFLAGS) $(LIBTRANCE) $(LIBUTILS) $(LIBCODEC) $(boost_LDADD) $(perftools_LDADD)

trance_oracle_SOURCES = trance_oracle.cpp
trance_oracle_LDADD   = $(LIBTRANCE) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

trance_treebank_SOURCES = trance_treebank.cpp
trance_treebank_LDADD   = $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)


noinst_DATA = .gitignore

.gitignore: Makefile.am
	cat $(top_builddir)/.gitignore > .gitignore
	for bin in $(PROGRAMS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SCRIPTS); do \
	  echo $$bin >> .gitignore; \
	done
