#!/usr/bin/env bash
set -euo pipefail

MARKER=$1
SRC=$2
TGT=$3

sed -i -s "$(grep -n "^$MARKER$" $SRC | cut -d: -f1 | tr '\n' ' ' | sed 's/ /d;/g')" $SRC $TGT
sed -i -s "$(grep -n "^$MARKER$" $TGT | cut -d: -f1 | tr '\n' ' ' | sed 's/ /d;/g')" $SRC $TGT
