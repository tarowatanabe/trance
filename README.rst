=============
Trance Parser
=============

Trance parser is an implementation of transition-based neural
constituent parsing proposed by Taro Watanabe, a transition-based
parser with neural networks to score all the derivation histories.

- Model1: no feedback from stack
- Model2: feedback from stack in the shift
- Model3: Model2 + queue context
- Model4: Model2 + feed back from stack in the reduce
- Model5: Model3 + Model4

Compile
-------

For details, see `BUILD.rst`.

.. code:: bash

   ./autogen.sh (required when you get the code by git clone)
   ./configure
   make
   make install (optional)

Parsing
-------



Training
--------

