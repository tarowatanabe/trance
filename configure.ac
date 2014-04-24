#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([rnnp], [0.0.0], [taro.watanabe@nict.go.jp])

AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR([config])

AM_INIT_AUTOMAKE([subdir-objects])

# 1. Start with version information of ‘0:0:0’ for each libtool library.
# 2. Update the version information only immediately before a public release of your software. More frequent updates are unnecessary, and only guarantee that the current interface number gets larger faster.
# 3. If the library source code has changed at all since the last update, then increment revision (‘c:r:a’ becomes ‘c:r+1:a’).
# 4. If any interfaces have been added, removed, or changed since the last update, increment current, and set revision to 0.
# 5. If any interfaces have been added since the last public release, then increment age.
# 6. If any interfaces have been removed or changed since the last public release, then set age to 0.

RNNP_LTVERSION=0:0:0
AC_SUBST(RNNP_LTVERSION)

RNNP_VERSION=0.0.0
AC_SUBST(RNNP_VERSION)

AC_CONFIG_HEADERS(utils/config.h)

# Checks for programs.
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CC

AM_PROG_CC_C_O

AC_LANG_CPLUSPLUS

AC_SYS_LARGEFILE

AM_PATH_PYTHON(,, [:])

### check -ftemplate-depth=1024
if test -n "$GCC"; then
   CXXFLAGS="$CXXFLAGS -ftemplate-depth=1024"
fi

# Checks for libraries.

AC_ARG_WITH(snappy,
        [AC_HELP_STRING([--with-snappy=DIR], [snappy in DIR])],
        [snappy_dir="${withval}"], [snappy_dir=''])
AC_ARG_ENABLE(snappy,
        [AC_HELP_STRING([--enable-snappy], [enable snappy])],
        [ac_enable_snappy=yes], [ac_enable_snappy=no])

if test "x$snappy_dir" != "x"; then
  SNAPPY_CPPFLAGS="-I${snappy_dir}/include"
  SNAPPY_LDFLAGS="-L${snappy_dir}/lib -lsnappy"
else
  SNAPPY_CPPFLAGS=""
  SNAPPY_LDFLAGS="-lsnappy"
fi

if test "x$ac_enable_snappy" = "xyes"; then
   AC_MSG_CHECKING(for snappy)
   
   save_CXXFLAGS="$CXXFLAGS"
   save_LDFLAGS="$LDFLAGS"
   CXXFLAGS="$CXXFLAGS $SNAPPY_CPPFLAGS"
   LDFLAGS="$LDFLAGS $SNAPPY_LDFLAGS"

   AC_TRY_LINK([], [], [ac_have_libsnappy=yes], [ac_have_libsnappy=no])
   AC_MSG_RESULT($ac_have_libsnappy)
   AC_CHECK_HEADERS([snappy.h])

   if test "x$ac_have_libsnappy" = "xno" -o "x$ac_cv_header_snappy_h" = "xno"; then
     SNAPPY_CPPFLAGS=""
     SNAPPY_LDFLAGS=""
   else
     AC_DEFINE(HAVE_SNAPPY, 1, [Define if you have snappy])
   fi

  CXXFLAGS="$save_CXXFLAGS"
  LDFLAGS="$save_LDFLAGS"
else
  SNAPPY_CPPFLAGS=""
  SNAPPY_LDFLAGS=""
fi
AC_SUBST(SNAPPY_CPPFLAGS)
AC_SUBST(SNAPPY_LDFLAGS)
AM_CONDITIONAL([WITH_SNAPPY], [test "x$ac_have_libsnappy" = "xyes" -a "x$ac_cv_header_snappy_h" = "xyes"])

AC_ARG_WITH(jemalloc,
	[AC_HELP_STRING([--with-jemalloc=DIR], [jemalloc in DIR])],
	[jemalloc_dir="${withval}"], [jemalloc_dir=''])
AC_ARG_ENABLE(jemalloc,
	[AC_HELP_STRING([--enable-jemalloc], [enable jemalloc])],
	[ac_enable_jemalloc=yes], [ac_enable_jemalloc=no])

if test "x$jemalloc_dir" != "x"; then
  JEMALLOC_CPPFLAGS="-I${jemalloc_dir}/include"
  JEMALLOC_LDFLAGS="-L${jemalloc_dir}/lib -ljemalloc"
else
  JEMALLOC_CPPFLAGS=""
  JEMALLOC_LDFLAGS="-ljemalloc"
fi

if test "x$ac_enable_jemalloc" = "xyes"; then
   AC_MSG_CHECKING(for jemalloc)
   save_LDFLAGS="$LDFLAGS"
   LDFLAGS="$LDFLAGS $JEMALLOC_LDFLAGS"
   AC_TRY_LINK([], [], [ac_have_libjemalloc=yes], [ac_have_libjemalloc=no])
   LDFLAGS="$save_LDFLAGS"
   AC_MSG_RESULT($ac_have_libjemalloc)
   if test "x$ac_have_libjemalloc" = "xno"; then
     JEMALLOC_CPPFLAGS=""
     JEMALLOC_LDFLAGS=""
   fi
else
  JEMALLOC_CPPFLAGS=""
  JEMALLOC_LDFLAGS=""
fi
AC_SUBST(JEMALLOC_CPPFLAGS)
AC_SUBST(JEMALLOC_LDFLAGS)


AC_ARG_WITH(tcmalloc,
	[AC_HELP_STRING([--with-tcmalloc=DIR], [tcmalloc in DIR])],
	[tcmalloc_dir="${withval}"], [tcmalloc_dir=''])
AC_ARG_ENABLE(tcmalloc,
	[AC_HELP_STRING([--enable-tcmalloc], [enable tcmalloc])],
	[ac_enable_tcmalloc=yes], [ac_enable_tcmalloc=no])

if test "x$tcmalloc_dir" != "x"; then
  TCMALLOC_CPPFLAGS="-I${tcmalloc_dir}/include"
  TCMALLOC_LDFLAGS="-L${tcmalloc_dir}/lib -ltcmalloc"
else
  TCMALLOC_CPPFLAGS=""
  TCMALLOC_LDFLAGS="-ltcmalloc"
fi

if test "x$ac_enable_tcmalloc" = "xyes"; then
   AC_MSG_CHECKING(for tcmalloc)
   save_LDFLAGS="$LDFLAGS"
   LDFLAGS="$LDFLAGS $TCMALLOC_LDFLAGS"
   AC_TRY_LINK([], [], [ac_have_libtcmalloc=yes], [ac_have_libtcmalloc=no])
   LDFLAGS="$save_LDFLAGS"
   AC_MSG_RESULT($ac_have_libtcmalloc)
   if test "x$ac_have_libtcmalloc" = "xno"; then
     TCMALLOC_CPPFLAGS=""
     TCMALLOC_LDFLAGS=""
   fi
else
  TCMALLOC_CPPFLAGS=""
  TCMALLOC_LDFLAGS=""
fi
AC_SUBST(TCMALLOC_CPPFLAGS)
AC_SUBST(TCMALLOC_LDFLAGS)

## --with-profiler
AC_ARG_WITH(profiler,
	[AC_HELP_STRING([--with-profiler=DIR], [profiler in DIR])],
	[profiler_dir="${withval}"], [profiler_dir=''])
AC_ARG_ENABLE(profiler,
	[AC_HELP_STRING([--enable-profiler], [enable profiling via google's libprofiler])],
	[ac_enable_profiler=yes], [ac_enable_profiler=no])

if test "x$profiler_dir" != "x"; then
  PROFILER_CPPFLAGS="-I${profiler_dir}/include"
  PROFILER_LDFLAGS="-L${profiler_dir}/lib -lprofiler"
else
  PROFILER_CPPFLAGS=""
  PROFILER_LDFLAGS="-lprofiler"
fi

if test "x$ac_enable_profiler" = "xyes"; then
   AC_MSG_CHECKING(for -lprofiler)
   save_LDFLAGS="$LDFLAGS"
   LDFLAGS="$LDFLAGS $PROFILER_LDFLAGS"
   AC_TRY_LINK([],[],[ac_have_libprofiler=yes],[ac_have_libprofiler=no])
   LDFLAGS="$save_LDFLAGS"
   AC_MSG_RESULT($ac_have_libprofiler)
   if test "x$ac_have_libprofiler" = "xno"; then
     PROFILER_CPPFLAGS=""
     PROFILER_LDFLAGS=""
   fi
else
  PROFILER_CPPFLAGS=""
  PROFILER_LDFLAGS=""
fi
AC_SUBST(PROFILER_CPPFLAGS)
AC_SUBST(PROFILER_LDFLAGS)

BOOST_REQUIRE([1.42])
BOOST_PROGRAM_OPTIONS
BOOST_TEST
BOOST_THREADS
BOOST_FILESYSTEM
BOOST_IOSTREAMS

## save pthred-flag, computed by-product of boost-threads
PTHREAD_FLAGS=$boost_cv_pthread_flag
AC_SUBST(PTHREAD_FLAGS)

##
## check for the pthread_spin_lock
##
AC_MSG_CHECKING(POSIX thread spin lock)
orig_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$orig_CXXFLAGS $PTHREAD_FLAGS"
AC_TRY_LINK([#include <pthread.h>],
        [pthread_spinlock_t spin;
        pthread_spin_init(&spin, PTHREAD_PROCESS_SHARED);
        pthread_spin_lock(&spin); pthread_spin_unlock(&spin);
        pthread_spin_destroy(&spin);],
        [pthread_spin=yes], [pthread_spin=no])
AC_MSG_RESULT($pthread_spin)
if test "x$pthread_spin" = "xyes"; then
   AC_DEFINE(HAVE_PTHREAD_SPINLOCK, 1, [Define if you have spinlock])
fi
CXXFLAGS="$orig_CXXFLAGS"

###
###  check for Mac OS X SpinLOck
### 
AC_CHECK_HEADERS([libkern/OSAtomic.h])

AC_CHECKING(OSSpinLock)
AC_TRY_LINK([#include <libkern/OSAtomic.h>],
	    [OSSpinLock spin = OS_SPINLOCK_INIT;
	     OSSpinLockLock(&spin);
	     OSSpinLockUnlock(&spin);],
	    [osspinlock=yes], [osspinlock=no])
AC_MSG_RESULT($osspinlock)
if test "$osspinlock" = "yes"; then
  AC_DEFINE(HAVE_OSSPINLOCK, 1, [Define if you have Mac OS X SpinLock])
fi

### check for MPI
AC_PATH_PROGS(MPI_CXX_COMPILER, [openmpic++ openmpicxx openmpiCC mpic++ mpicxx mpiCC], no)
if test "x$MPI_CXX_COMPILER" = "xno"; then
  AS_ERROR("no mpi...")
fi 

AC_MSG_CHECKING(mpi cppflags)
MPI_CPPFLAGS=`$MPI_CXX_COMPILER -showme | $AWK '{for(i=1;i<=NF;i++){if($i~/^-I/ || $i~/^-D/){printf "%s ", $i;}}}'`
AC_MSG_RESULT($MPI_CPPFLAGS)
AC_SUBST(MPI_CPPFLAGS)

AC_MSG_CHECKING(mpi ldflags)
MPI_LDFLAGS=`$MPI_CXX_COMPILER -showme | $AWK '{for(i=1;i<=NF;i++){if($i~/^-L/ || $i~/^-l/ || $i~/^-Wl/){printf "%s ", $i;}}}'`
AC_MSG_RESULT($MPI_LDFLAGS)
AC_SUBST(MPI_LDFLAGS)

### __thread
AC_MSG_CHECKING([for __thread])
AC_LINK_IFELSE([AC_LANG_PROGRAM([
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)) && ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 1) || (__GNUC__ == 4 && __GNUC_MINOR__ == 1 && __GNUC_PATCHLEVEL__ < 2))
#error gcc has this bug: http://gcc.gnu.org/ml/gcc-bugs/2006-09/msg02275.html
#endif
#if defined(__APPLE__)
#error no __thread in mac os x
#endif
], [static __thread int p = 0])],
               [AC_DEFINE(HAVE_TLS, 1, Define to 1 if compiler supports __thread)
                AC_MSG_RESULT([yes])],
               [AC_MSG_RESULT([no])])


## check malloc stuff...
AC_CHECK_HEADERS([malloc.h malloc/malloc.h sys/resource.h])

if test "x$ac_enable_jemalloc" = "xyes"; then
  save_LDFLAGS="$LDFLAGS"
  save_CXXFLAGS="$CXXFLAGS"
  LDFLAGS="$LDFLAGS $JEMALLOC_LDFLAGS"
  CXXFLAGS="$CXXFLAGS $JEMALLOC_CPPFLAGS"

  AC_CHECK_HEADERS([jemalloc/jemalloc.h])

  AC_MSG_CHECKING(jemalloc_mallctl)

  AC_TRY_LINK([#include <jemalloc/jemalloc.h>],
              [mallctl("stats.allocated", 0, 0, 0, 0);],
	       [have_jemalloc_mallctl=yes], [have_jemalloc_mallctl=no])
  AC_MSG_RESULT($have_jemalloc_mallctl)

  LDFLAGS="$save_LDFLAGS"
  CXXFLAGS="$save_CXXFLAGS"

  if test "$have_jemalloc_mallctl" = "yes"; then
    AC_DEFINE(HAVE_JEMALLOC_MALLCTL, 1, [Define if you have mallctl (in non-Mozilla jemalloc)])
  fi
fi

if test "x$ac_enable_jemalloc" = "xyes"; then
  save_CXXFLAGS="$CXXFLAGS"
  save_CPPFLAGS="$CPPFLAGS"
  CXXFLAGS="$CXXFLAGS $JEMALLOC_CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $JEMALLOC_CPPFLAGS"
  AC_CHECK_HEADERS([jemalloc.h])
  CXXFLAGS="$save_CXXFLAGS"
  CPPFLAGS="$save_CPPFLAGS"

  AC_MSG_CHECKING(jemalloc_stats)
  save_CXXFLAGS="$CXXFLAGS"
  save_LDFLAGS="$LDFLAGS"
  CXXFLAGS="$CXXFLAGS $JEMALLOC_CPPFLAGS"
  LDFLAGS="$LDFLAGS $JEMALLOC_LDFLAGS"
  AC_TRY_LINK([#include <jemalloc.h>],
            [jemalloc_stats_t stat; jemalloc_stats(&stat);],
	    [have_jemalloc_stats=yes], [have_jemalloc_stats=no])
  CXXFLAGS="$save_CXXFLAGS"
  LDFLAGS="$save_LDFLAGS"
  AC_MSG_RESULT($have_jemalloc_stats)
  if test "$have_jemalloc_stats" = "yes"; then
    AC_DEFINE(HAVE_JEMALLOC_STATS, 1, [Define if you have jemalloc_stats])
  fi
fi

if test "x$ac_enable_tcmalloc" = "xyes"; then
   save_CXXFLAGS="$CXXFLAGS"
   save_CPPFLAGS="$CPPFLAGS"
   CXXFLAGS="$CXXFLAGS $TCMALLOC_CPPFLAGS"
   CPPFLAGS="$CPPFLAGS $TCMALLOC_CPPFLAGS"
   AC_CHECK_HEADERS([google/malloc_extension.h])
   AC_CHECK_HEADERS([gperftools/malloc_extension.h])
   CXXFLAGS="$save_CXXFLAGS"
   CPPFLAGS="$save_CPPFLAGS"

   AC_MSG_CHECKING(MallocExtension)
   save_CXXFLAGS="$CXXFLAGS"
   save_LDFLAGS="$LDFLAGS"
   CXXFLAGS="$CXXFLAGS $TCMALLOC_CPPFLAGS"
   LDFLAGS="$LDFLAGS $TCMALLOC_LDFLAGS"

   AC_TRY_LINK([#include <gperftools/malloc_extension.h>],
               [MallocExtension::instance();],
	       	    [have_malloc_extension=yes], [have_malloc_extension=no])
  
   if test "$have_malloc_extension" = "no"; then
     AC_TRY_LINK([#include <google/malloc_extension.h>],
                 [MallocExtension::instance();],
  	       	    [have_malloc_extension=yes], [have_malloc_extension=no])
   fi
   
   CXXFLAGS="$save_CXXFLAGS"
   LDFLAGS="$save_LDFLAGS"
   AC_MSG_RESULT($have_malloc_extension)
   if test "$have_malloc_extension" = "yes"; then
     AC_DEFINE(HAVE_GOOGLE_MALLOC_EXTENSION, 1, [Define if you have malloc_extension])
   fi
fi

AC_MSG_CHECKING(malloc_zone_statistics)
AC_TRY_LINK([#include <malloc/malloc.h>],
            [struct malloc_statistics_t stat;
	     malloc_zone_statistics(NULL, &stat);],
	     [have_malloc_zone_statistics=yes], [have_malloc_zone_statistics=no])
AC_MSG_RESULT($have_malloc_zone_statistics)
if test "$have_malloc_zone_statistics" = "yes"; then
  AC_DEFINE(HAVE_MALLOC_ZONE_STATISTICS, 1, [Define if you have malloc_zone_statistics])
fi

### clock_gettime
AC_MSG_CHECKING([for clock_gettime])

AC_TRY_COMPILE(
[
#include <time.h>
], [
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return 0;
],
[
AC_MSG_RESULT([yes])
AC_DEFINE([HAVE_CLOCK_GETTIME], 1, [Define to 1 if you have the clock_gettime.])
], [
AC_MSG_RESULT([no])
])
AC_CHECK_LIB([rt], [clock_gettime])

### task_info
AC_MSG_CHECKING([for thread_info])
AC_TRY_COMPILE(
[
#include <mach/mach_init.h>
#include <mach/thread_info.h>
#include <mach/thread_act.h>
], [
   kern_return_t error;
   struct thread_basic_info th_info;
   mach_msg_type_number_t th_info_count = THREAD_BASIC_INFO_COUNT;

   error = thread_info(mach_thread_self(), THREAD_BASIC_INFO, (thread_info_t)&th_info, &th_info_count);
   return 0;
],
[
AC_MSG_RESULT([yes])
AC_DEFINE([HAVE_THREAD_INFO], 1, [Define to 1 if you have the thread_info.])
], [
AC_MSG_RESULT([no])
])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h malloc.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# hasn_{map,set}
AC_CHECK_HEADERS([unordered_map unordered_set tr1/unordered_map tr1/unordered_set ext/hash_map ext/hash_set hash_map hash_set])

# cmath, tr1/cmath
AC_CHECK_HEADERS([cmath tr1/cmath])

if test "x$ac_cv_header_cmath" = "xyes"; then
  AC_MSG_CHECKING([for std::log1p])
  AC_TRY_COMPILE([#include <cmath>],
     [ std::log1p(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_STD_LOG1P], 1, [Define to 1 if you have the std::log1p.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for std::lgamma])
  AC_TRY_COMPILE([#include <cmath>],
     [ std::lgamma(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_STD_LGAMMA], 1, [Define to 1 if you have the std::lgamma.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for ::log1p])
  AC_TRY_COMPILE([#include <cmath>],
     [ ::log1p(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_LOG1P], 1, [Define to 1 if you have the ::log1p.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for ::lgamma])
  AC_TRY_COMPILE([#include <cmath>],
     [ ::lgamma(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_LGAMMA], 1, [Define to 1 if you have the ::lgamma.])],
     [AC_MSG_RESULT([no])])
fi

if test "x$ac_cv_header_tr1_cmath" = "xyes"; then
  AC_MSG_CHECKING([for std::log1p])
  AC_TRY_COMPILE([#include <tr1/cmath>],
     [ std::log1p(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_STD_LOG1P], 1, [Define to 1 if you have the std::log1p.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for std::lgamma])
  AC_TRY_COMPILE([#include <tr1/cmath>],
     [ std::lgamma(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_STD_LGAMMA], 1, [Define to 1 if you have the std::lgamma.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for ::log1p])
  AC_TRY_COMPILE([#include <tr1/cmath>],
     [ ::log1p(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_LOG1P], 1, [Define to 1 if you have the ::log1p.])],
     [AC_MSG_RESULT([no])])

  AC_MSG_CHECKING([for ::lgamma])
  AC_TRY_COMPILE([#include <tr1/cmath>],
     [ ::lgamma(1.0); ],
     [AC_MSG_RESULT([yes])
      AC_DEFINE([HAVE_LGAMMA], 1, [Define to 1 if you have the ::lgamma.])],
     [AC_MSG_RESULT([no])])
fi

## SSE related...
AC_CHECK_HEADERS([mm_malloc.h xmmintrin.h nmmintrin.h emmintrin.h tmmintrin.h ammintrin.h bmmintrin.h])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h malloc.h stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_HEADER_MAJOR
AC_FUNC_REALLOC
AC_CHECK_FUNCS([clock_gettime gethostname gettimeofday memmove memset pow select sqrt strerror strncasecmp])

AC_CONFIG_FILES([Makefile
                 codec/Makefile
                 eigen/Makefile
                 progs/Makefile
                 rnnp/Makefile
                 utils/Makefile])
AC_OUTPUT
