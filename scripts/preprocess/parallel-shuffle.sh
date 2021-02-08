#!/bin/bash
set -euo pipefail

IN_LEFT=$1
IN_RIGHT=$2
OUT_LEFT=${3:-$IN_LEFT.shuf}
OUT_RIGHT=${4:-$IN_RIGHT.shuf}
# paste $IN_LEFT $IN_RIGHT | shuf | awk -F$'\t' '{ print $1 > "data/dev.shuffle.src" ; print $2 > "data/dev.shuffle.src" }'
paste $IN_LEFT $IN_RIGHT | shuf | awk -F$'\t' '{ print $1 > "'"$OUT_LEFT"'" ; print $2 > "'"$OUT_RIGHT"'" }'
