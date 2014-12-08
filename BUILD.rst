==========================
How to Build Trance Parser
==========================

Get the cutting-edge source code from `github.com <http://github.com/tarowatanabe/trance>`_:

.. code:: bash

  git clone https://github.com/tarowatanabe/trance.git

Or, grab the stable tar archive from `trance <http://www2.nict.go.jp/univ-com/multi_trans/trance>`_.

Compile
-------

.. code:: bash

   ./autogen.sh (required when you get the code by git clone)
   env CFLAGS="-O3" CXXFLAGS="-O3" ./configure
   make
   make install (optional)

You can set several options. For details see the requirement section.
::

  --enable-snappy         enable snappy
  --enable-jemalloc       enable jemalloc
  --enable-tcmalloc       enable tcmalloc
  --enable-profiler       enable profiling via google's libprofiler
  --enable-static-boost   Prefer the static boost libraries over the shared
                          ones [no]

  --with-snappy=DIR       snappy in DIR
  --with-jemalloc=DIR     jemalloc in DIR
  --with-tcmalloc=DIR     tcmalloc in DIR
  --with-profiler=DIR     profiler in DIR
  --with-boost=DIR        prefix of Boost 1.42 [guess]

In addition to the configuration options, it is better to set ``-O3``
for the ``CFLAGS`` and ``CXXFLAGS`` environment variables for faster
execution, when compiled by **gcc** or **clang**.

Requirements
------------

- Boost library: http://www.boost.org/

  The minimum requirement is boost version 1.42. Prior to this
  version, there were a couple of serious bugs which prevent us from
  running correctly.

- ICU: http://site.icu-project.org/
   
  The `configure` script relies on `icu-config` installed by the ICU
  library. Thus, `icu-config` must be in the executable path.

- Optional:

  + MPI (Open MPI): http://www.open-mpi.org/
    
    We strongly recommend open-mpi since it is regularly tested.
    The MPI libraries are automatically detected by the `configure`
    script by finding either `mpic++`, `mpicxx` or `mpiCC`. Thus,
    those mpi specific compilers should be on the executable path.

   + snappy: http://code.google.com/p/snappy/

   + For better memory management:

     * gperftools: http://code.google.com/p/gperftools/
     * jemalloc: http://www.canonware.com/jemalloc/

     For Linux, you should install one of them for better memory performance
     and to measure how many bytes malloced, since mallinfo is
     "broken" for memory more than 4GB.
     They are configured by `--with-{jemalloc,tcmalloc}` and should be
     enabled using `--enable-{jemalloc,tcmalloc}`
