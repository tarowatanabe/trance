#!/bin/sh

abspath() {
  dir__=$1
  "cd" "$dir__"
  if test "$?" = "0"; then
    /bin/pwd
    "cd" -  &>/dev/null
  fi
}

dirrel=`dirname $0`
trance=`abspath $dirrel`

exec $trance/../progs/trance_parse \
     --grammar $trance/WSJ-grammar.gz \
     --model $trance/WSJ-64 \
     --signature English \
     "$@"
