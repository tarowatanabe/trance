#!/bin/sh

# We use stanford tagger from http://nlp.stanford.edu/software/tagger.shtml

trance=directory-for-trance-parser
corpus=directory-for-pentreebank/bracketed
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

## prepare POS: Here, we extract terminals only.
cat `./ctb-files.py $corpus 1-270 440-1151` | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --leaf  | \
java -mx4g -cp $tagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model $tagger/models/chinese-distsim.tagger -textFile /dev/stdin -tokenize false -outputFormat slashTags -tagSeparator _ -encoding utf-8 | \
${samples}/split-pos.py > CTB-train.pos

## prepare treebanks. The POSs of training data are replaced by those tagged by stanford POS tagger.
cat `${samples}/ctb-files.py $corpus 1-270 440-1151` | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --pos CTB-train.pos --output CTB-train.treebank

cat `${samples}/ctb-files.py $corpus 301-325` | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --output CTB-dev.treebank

cat `${samples}/ctb-files.py $corpus 271-300` | \
${trance}/progs/trance_treebank --remove-none --remove-cycle --normalize --output CTB-test.treebank

### grammar... Here we use Chinese signature when falling back to OOV.
${trance}/progs/trance_grammar \
	 --input CTB-train.treebank \
	 --output CTB-grammar.gz \
	 --signature Chinese \
	 --cutoff 4 \
	 --split-preterminal \
	 --debug 2> CTB-grammar.log

## learn...
${trance}/progs/trance_learn \
	 --input CTB-train.treebank \
	 --test  CTB-dev.treebank \
	 --output CTB-64 \
	 --grammar CTB-grammar.gz \
	 --signature Chinese \
	 --unary 4 \
	 --model5 \
	 --hidden 64 \
	 --embedding 512 \
	 --randomize \
	 --kbest 100 \
	 --beam 32 \
	 --learn all:opt=adadec,violation=max,margin-all=true,batch=4,iteration=100,eta=1e-2,gamma=0.9,epsilon=1,lambda=1e-5 \
	 --mix-select \
	 --averaging \
	 --threads 4 \
	 --debug
