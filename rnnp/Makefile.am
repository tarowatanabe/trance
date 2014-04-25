
AM_CPPFLAGS = -I.. -I../eigen $(BOOST_CPPFLAGS) $(PTHREAD_FLAGS)

LIBUTILS=$(top_builddir)/utils/libutils.la

lib_LTLIBRARIES = librnnp.la

librnnpincludedir= $(pkgincludedir)/rnnp

objectivedir = $(pkgincludedir)/rnnp/objective
optimizedir  = $(pkgincludedir)/rnnp/optimize
semiringdir  = $(pkgincludedir)/rnnp/semiring

dist_objective_HEADERS = \
objective/margin.hpp \
objective/margin_derivation.hpp \
objective/margin_early.hpp \
objective/margin_late.hpp \
objective/margin_max.hpp \
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

librnnpinclude_HEADERS = \
allocator.hpp \
binarize.hpp \
binarize_left.hpp \
binarize_right.hpp \
debinarize.hpp \
derivation.hpp \
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
parser.hpp \
rule.hpp \
semiring.hpp \
sentence.hpp \
span.hpp \
state.hpp \
symbol.hpp \
tree.hpp

librnnp_la_SOURCES = \
gradient.cpp \
grammar.cpp \
graphviz.cpp \
learn_option.cpp \
model.cpp \
operation.cpp \
option.cpp \
rule.cpp \
sentence.cpp \
span.cpp \
symbol.cpp \
tree.cpp

librnnp_la_LDFLAGS = \
	$(PTHREAD_FLAGS) \
	$(BOOST_THREAD_LDFLAGS) \
	$(BOOST_FILESYSTEM_LDFLAGS) \
	$(BOOST_IOSTREAMS_LDFLAGS) \
	-version-info $(RNNP_LTVERSION)

librnnp_la_LIBADD = \
	$(LIBUTILS) \
	$(BOOST_THREAD_LIBS) \
	$(BOOST_FILESYSTEM_LIBS) \
	$(BOOST_IOSTREAMS_LIBS) 

noinst_PROGRAMS = \
graphviz_main \
tree_main

graphviz_main_SOURCES  = graphviz_main.cpp 
graphviz_main_LDADD    = librnnp.la

tree_main_SOURCES  = tree_main.cpp 
tree_main_LDADD    = librnnp.la
