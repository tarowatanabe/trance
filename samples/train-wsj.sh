#!/bin/sh

# We use stanford tagger from http://nlp.stanford.edu/software/tagger.shtml

trance=directory-for-trance-parser
corpus=directory-for-pentreebank/parsed/mrg/wsj
tagger=directory-for-stanford-tagger
samples=${trance}/samples

if test ! -e "$trance"; then
  echo "no trance directory" >&2
  exit -1
fi

if test ! -e "$corpus"; then
  echo "no corpus directory" >&2
  exit -1
fi

if test ! -e "$tagger"; then
  echo "no tagger directory" >&2
  exit -1
fi

if test ! -e "$samples"; then
  echo "no samples directory" >&2
  exit -1
fi

## prepare POS: Here, we extract unescaped terminals, \/ into / and \* into * etc.
cat ${corpus}/0[2-9]/* ${corpus}/1[0-9]/* ${corpus}/2[0-1]/* | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --leaf --unescape | \
java -mx4g -cp $tagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model $tagger/models/wsj-0-18-bidirectional-distsim.tagger -textFile /dev/stdin -tokenize false -outputFormat slashTags -tagSeparator _ | \
${samples}/split-pos.py > WSJ-train.pos

## prepare treebanks. The POSs of training data are replaced by those tagged by stanford POS tagger.
cat ${corpus}/0[2-9]/* ${corpus}/1[0-9]/* ${corpus}/2[0-1]/* | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --pos WSJ-train.pos --output WSJ-train.treebank

cat ${corpus}/22/* | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --output WSJ-22.treebank

cat ${corpus}/23/* | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --output WSJ-23.treebank

### grammar... Here we use English signature when falling back to OOV.
${trance}/progs/trance_grammar \
	 --input WSJ-train.treebank \
	 --output WSJ-grammar.gz \
	 --signature English \
	 --cutoff 3 \
	 --split-preterminal \
	 --debug 2> WSJ-grammar.log

## learn...
${trance}/progs/trance_learn \
	 --input WSJ-train.treebank \
	 --test  WSJ-22.treebank \
	 --output WSJ-model \
	 --grammar WSJ-grammar.gz \
	 --signature English \
	 --unary 3 \
	 --model5 \
	 --hidden 64 \
	 --embedding 1024 \
	 --randomize \
	 --kbest 100 \
	 --beam 32 \
	 --learn all:opt=adadec,violation=max,margin-all=true,batch=4,iteration=100,eta=1e-2,gamma=0.9,epsilon=1,lambda=1e-5 \
	 --mix-select \
	 --averaging \
	 --threads 4 \
	 --debug
