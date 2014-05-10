
AM_CPPFLAGS = -I.. -I../eigen $(BOOST_CPPFLAGS) $(ICU_CPPFLAGS) $(PTHREAD_FLAGS)

LIBUTILS=$(top_builddir)/utils/libutils.la

lib_LTLIBRARIES = librnnp.la

librnnpincludedir= $(pkgincludedir)/rnnp

objectivedir = $(pkgincludedir)/rnnp/objective
optimizedir  = $(pkgincludedir)/rnnp/optimize
semiringdir  = $(pkgincludedir)/rnnp/semiring
signaturedir = $(pkgincludedir)/rnnp/signature

dist_objective_HEADERS = \
objective/margin.hpp \
objective/margin_cross.hpp \
objective/margin_derivation.hpp \
objective/margin_evalb.hpp \
objective/margin_early.hpp \
objective/margin_late.hpp \
objective/margin_max.hpp \
objective/violation.hpp \
objective/violation_early.hpp \
objective/violation_late.hpp \
objective/violation_max.hpp 

dist_optimize_HEADERS = \
optimize/adadec.hpp \
optimize/adadelta.hpp \
optimize/adagrad.hpp \
optimize/sgd.hpp

dist_semiring_HEADERS = \
semiring/log.hpp \
semiring/logprob.hpp \
semiring/tropical.hpp \
semiring/traits.hpp

dist_signature_HEADERS = \
signature/chinese.hpp \
signature/english.hpp \
signature/unicode.hpp

librnnpinclude_HEADERS = \
allocator.hpp \
binarize.hpp \
binarize_left.hpp \
binarize_right.hpp \
debinarize.hpp \
derivation.hpp \
evalb.hpp \
forest.hpp \
gradient.hpp \
grammar.hpp \
graphviz.hpp \
learn_option.hpp \
loss.hpp \
model.hpp \
objective.hpp \
operation.hpp \
optimize.hpp \
option.hpp \
oracle.hpp \
parser.hpp \
rule.hpp \
semiring.hpp \
sentence.hpp \
signature.hpp \
span.hpp \
state.hpp \
symbol.hpp \
tree.hpp

librnnp_la_SOURCES = \
forest.cpp \
gradient.cpp \
grammar.cpp \
graphviz.cpp \
learn_option.cpp \
model.cpp \
operation.cpp \
option.cpp \
rule.cpp \
sentence.cpp \
signature.cpp \
span.cpp \
symbol.cpp \
tree.cpp

librnnp_la_LDFLAGS = \
	$(PTHREAD_FLAGS) \
	$(BOOST_THREAD_LDFLAGS) \
	$(BOOST_FILESYSTEM_LDFLAGS) \
	$(BOOST_IOSTREAMS_LDFLAGS) \
	$(ICU_LDFLAGS) \
	-version-info $(RNNP_LTVERSION)

librnnp_la_LIBADD = \
	$(LIBUTILS) \
	$(BOOST_THREAD_LIBS) \
	$(BOOST_FILESYSTEM_LIBS) \
	$(BOOST_IOSTREAMS_LIBS) \
	$(ICU_LDFLAGS)

noinst_PROGRAMS = \
grammar_main \
oracle_main \
rule_main \
signature_main \
tree_main

grammar_main_SOURCES  = grammar_main.cpp 
grammar_main_LDADD    = librnnp.la

oracle_main_SOURCES  = oracle_main.cpp 
oracle_main_LDADD    = librnnp.la $(BOOST_PROGRAM_OPTIONS_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIBS)

rule_main_SOURCES  = rule_main.cpp 
rule_main_LDADD    = librnnp.la

signature_main_SOURCES  = signature_main.cpp 
signature_main_LDADD    = librnnp.la

tree_main_SOURCES  = tree_main.cpp 
tree_main_LDADD    = librnnp.la

noinst_DATA = .gitignore

.gitignore: Makefile.am
	cat $(top_builddir)/.gitignore > .gitignore
	for bin in $(PROGRAMS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SCRIPTS); do \
	  echo $$bin >> .gitignore; \
	done
