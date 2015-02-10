=============
Trance Parser
=============

Trance parser is an implementation of transition-based neural
constituent parsing proposed by [1]_, a transition-based
parser with neural networks to score all the derivation histories.

Currently, we support following neural networks (For details, see [1]_):

- Model1: no feedback from stacks or contexts (`tree` model)
- Model2: feedback from stacks for shift actions
- Model3: Model2 + queue contexts
- Model4: Model2 + feed back from stack for reduce/unary actions (`+stack` model)
- Model5: Model4 + queue contexts (`+queue` model)

Various training objective:

- {max,early,late}-violation with expected/Viterbi mistakes
- expected evalb
- structured hinge loss

and online optimizer: SGD, AdaGrad, AdaDec and AdaDelta.
  
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

We provide models for 2 languages, English (WSJ) and Chinese
(CTB). They are Model5 which performs the best in our
settings. Following is an example to run our modes, using STDIN/STDOUT
as our input/output (assuming utf-8 encoding of input/output):

.. code:: bash

   progs/trance_parse \
	  --grammar models/{WSJ,CTB}-grammar.gz \
	  --model models/{WSJ,CTB}-model \
	  --unary {3,4} \
	  --signature {English,Chinese} \
	  --precompute \
          --simple

where ``--unary`` specifies the number of consequtive unaries and
uses 3 for WSJ, and 4 for CTB. ``--signature`` is used to represent
OOVs based on the word's signature and ``--precompute`` performs word
representation precomputation for faster parsing. The option
``--simple`` specifies a Penn-treebank style output format.
Input sentences are assumed to be tokenized according to their
standards: For English, it is recommended to use a tokenizer from the
`Stanford Parser <http://nlp.stanford.edu/software/lex-parser.shtml>`_.
For Chinese, the `Stanford Word Segmenter
<http://nlp.stanford.edu/software/segmenter.shtml>`_ is a good choice.

Training
--------

Sample scripts are available in `samples/train-{wsj,ctb}.sh` for
training WSJ and CTB, respectively, using some publicly available
tools.

In brief, first, we need to obtain treebank trees in a normalized
form:

.. code:: bash

   cat [treebank files] | \
   progs/trance_treebank \
	  --output [output normalized treebank]
	  --normalize \
	  --remove-none \
	  --remove-cycle

Here, trees are normalized by adding ROOT label, removing `-NONE-`,
removing X over X unaries and stripping off tags in each label. If you
add ``--leaf`` flag, it will output only leaves, i.e. sentences. The
``--pos`` option can replace each POS tag in trees specified by a
POS-file consisting of a sequence of POSs for each word.

Second, we need to compute grammar from a treebank:

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

Third, learn a model:

.. code:: bash

   progs/trance_learn \
	  --input [treebank file] \
	  --test [treebank development file] \
	  --output [model file] \
	  --grammar [grammar file] \
	  --unary   [maximum unary size] \
	  --hidden [hidden dimension size] \
	  --embedding [word embedding dimension size] \
          --beam 32 \
          --kbest 128 \
	  --randomize \
	  --learn all:opt=adadec,violation=max,margin-all=true,batch=4,iteration=100,eta=1e-2,gamma=0.9,epsilon=1,lambda=1e-5 \
	  --mix-select \
	  --averaging \
	  --debug

Here, we use ``--input`` option to specify training data and use
``--test`` for development data. The ``--output`` will output a model
with the best evalb score under the development data. By default, we
will train Model5, but you can use different models by
``--model[1-5]`` options. The grammar file is learned by
``trance_grammar`` and if you specified ``--signature`` option, you
have to use the same one. ``--unary`` option should be the same as the
maximum unary size output by the ``trance_grammar`` with ``--debug``
option.

By default, we use the hidden size of 64 and embedding size of 64, and
the model parameters are initialized randomly (``--ramdomize``). You
can precompute word embedding by `word2vec <https://code.google.com/p/word2vec/>`_
or `rnnlm <http://rnnlm.org>`_, then use it as initial parameters for
word representation by ``--word-embedding [embedding file]``
option. The format is as follows:
::
   
   word1 param1 param2 ... param[embedding size]
   word2 param1 param2 ... param[embedding size]
   word3 param1 param2 ... param[embedding size]

The parameter estimation is performed by AdaDec with max-violation
considering expected mistakes (``margin-all=true``) with hyperparameters
of eta=1e-2, gamma=0.9, epsilon=1, lambda=1e-5. The maximum number of
iterations is set to 100 with mini-batch size of 4, beam size of 32
and kbest size of 128, i.e., the beam size in the final bin. In each
iteration, we select the best model with respect to L1 norm
(``--mix-select``) and performs averaging for model output
(``--averaging``). For details, see [1]_.

References
----------

.. [1]   Taro Watanabe and Eiichiro Sumita. Transition-based Neural
	 Constituent Parsing. In Proc. of HLT-NAACL 2015.
