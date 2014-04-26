AM_CPPFLAGS = -I.. -I../eigen $(BOOST_CPPFLAGS) $(PTHREAD_FLAGS) $(TCMALLOC_CPPFLAGS) $(PROFILER_CPPFLAGS)

perftools_LDADD = $(JEMALLOC_LDFLAGS) $(TCMALLOC_LDFLAGS) $(PROFILER_LDFLAGS)

boost_LDADD = \
	$(BOOST_THREAD_LDFLAGS) $(BOOST_THREAD_LIBS) \
	$(BOOST_PROGRAM_OPTIONS_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIBS) \
	$(BOOST_FILESYSTEM_LDFLAGS) $(BOOST_FILESYSTEM_LIBS) \
	$(BOOST_IOSTREAMS_LDFLAGS) $(BOOST_IOSTREAMS_LIBS) 

LIBRNNP   = $(top_builddir)/rnnp/librnnp.la
LIBUTILS = $(top_builddir)/utils/libutils.la
LIBCODEC = $(top_builddir)/codec/libcodec.la

bin_PROGRAMS = \
	rnnp_grammar \
	rnnp_parse \
	rnnp_oracle \
	rnnp_treebank

rnnp_grammar_SOURCES = rnnp_grammar.cpp
rnnp_grammar_LDADD   = $(LIBRNNP) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

rnnp_parse_SOURCES = rnnp_parse.cpp
rnnp_parse_LDADD   = $(LIBRNNP) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

rnnp_oracle_SOURCES = rnnp_oracle.cpp
rnnp_oracle_LDADD   = $(LIBRNNP) $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)

rnnp_treebank_SOURCES = rnnp_treebank.cpp
rnnp_treebank_LDADD   = $(LIBUTILS) $(boost_LDADD) $(perftools_LDADD)


noinst_DATA = .gitignore

.gitignore: Makefile.am
	cat $(top_builddir)/.gitignore > .gitignore
	for bin in $(PROGRAMS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SCRIPTS); do \
	  echo $$bin >> .gitignore; \
	done
