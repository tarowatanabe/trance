=============
Trance Parser
=============

Trance parser is an implementation of transition-based neural
constituent parsing proposed by Taro Watanabe, a transition-based
parser with neural networks to score all the derivation histories.

Currently, we support following neural networks (For details, see the paper):

- Model1: no feedback from stacks or contexts (`tree` model in the
  paper)
- Model2: feedback from stacks for shift actions
- Model3: Model2 + queue contexts
- Model4: Model2 + feed back from stack for reduce/unary actions
  (`+stack` model in the paper)
- Model5: Model4 + queue contexts (`+queue` model in the paper)

  
Compile
-------

The latest code is available from `github.com <http://github.com/tarowatanabe/trance>`_.

We follow a standard practice of configure/make/make install. For
details, see `BUILD.rst`.

.. code:: bash

   ./autogen.sh (required when you get the code by git clone)
   ./configure
   make
   make install (optional)

Parsing
-------

We provide 2 languages, English (WSJ) and Chinese (CTB), and two
models each by varying the hidden dimension size, 32 and 64. They are
Model5 which performs the best in our settings.

.. code:: bash

   progs/trance_parse \
	  --grammar models/{WSJ,CTB}-grammar.gz \
	  --model models/{WSJ,CTB}-{32,64} \
	  --unary {3,4} \
	  --signature {English,Chinese} \
	  --precompute

where, ``--unary`` specify the number of consequtive unaries,
``--signature`` is used to represent OOVs based on the word's
signature and ``--precompute`` performs word representation
precomputation for faster parsing.

Training
--------

First, we need to compute grammar from a treebank:

.. code:: bash

   progs/trance_grammar \
	  --input [treebank file] \
	  --output [grammar file] \
	  --cutoff 3 \
	  --debug

By default, we use the cutoff threshold to 3 (``--cutoff 3``)
indicating that the words which occur twice or less are mapped to
special token `<unk>`. For English or Chinese, it is better to use
word signature for better mapping OOVs by adding ``--signature
{English,Chinese}`` option. The ``--debug`` option is recommended
since it will output various information, most notable, the maximum
number of unary size, which is used during learning and testing via
``--unary [maximum unary size]`` option.

Second, learn a model:

.. code:: bash

   progs/trance_learn \
	  --input [treebank file] \
	  --test [treebank development file] \
	  --output [model file] \
	  --grammar [grammar file] \
	  --unary   [maximum unary size] \
	  --hidden [hidden dimension size] \
	  --embedding [word embedding dimension size] \
	  --randomize \
	  --learn all:opt=adadec,violation=max,margin-all=true,batch=4,iteration=100,eta=1e-2,gamma=0.9,epsilon=1,lambda=1e-5 \
	  --mix-select \
	  --averaging \
	  --debug

Here, We use ``--input`` option to specify training data and use
``--test`` for development data. The ``--output`` will output a model
with the best evalb score under the development data. By default, we
will train Model5, but you can use different models by
``--model[1-5]`` options. The grammar file is learned by
``trance_grammar`` and if you specified ``--signature`` option, you
have to use the same one. ``--unary`` option should be the same as the
maximum unary size output by the ``trance_grammar`` with ``--debug``
option.

By default, we use the hidden size of 64 and embedding size of 64. You
can precompute word embedding by word2vec or rnnlm, then use it as
initial parameters for word representation by ``--word-embedding
[embedding file]`` option. The format is as follows:
::
   word1 param1 param2 ... param[embedding size]
   word2 param1 param2 ... param[embedding size]
   word3 param1 param2 ... param[embedding size]

The parameter estimation is performed by AdaDec with max-violation
considering expected mistakes (margin-all=true) with hyperparameters
of eta=1e-2, gamma=0.9, epsilon=1, lambda=1e-5. The maximum number of
iterations is set to 100 with mini-batch size of 4. In each iteration,
we select the best model with respect to L1 norm (``--mix-select``)
and performs averaging for model output (``--averaging``). For
details, see ...



