#!/bin/bash
set -euo pipefail

join <(
    jq -r '"\(input_filename) \(.[0].score)"' "$1"/*.??-??.jsonl |
        sed 's#\([^ ]*/\|^\)\([^ /]*\)\.\([a-z][a-z]-[a-z][a-z]\)\.jsonl #\3.\2 #' |
        sort
) <(
    jq -r '"\(input_filename) \(.[0].score)"' "$2"/*.??-??.jsonl |
        sed 's#\([^ ]*/\|^\)\([^ /]*\)\.\([a-z][a-z]-[a-z][a-z]\)\.jsonl #\3.\2 #' |
        sort
) |
    perl -lane '
        $avg += $F[2]-$F[1];
        printf "%s\t%+.1f\n", $_, $F[2]-$F[1];
        END { printf "xx-xx.avg   %+.2f\n", $avg/$.; }' |
    sed 's/\./ /' |
    column -t -n
