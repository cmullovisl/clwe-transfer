#!/bin/bash
set -euo pipefail

##paste <(
##    ls -1d "$dir"
##    ls -1d "$dir"/??-??
#paste <(
#    ls -1d ??-??
#) <(
#    jq '.[1].score' "$1"/??-??/"$scorefile"
#) <(
#    jq '.[1].score' "$2"/??-??/"$scorefile"
#) | perl -lane 'printf "%s\t%+.1f\n", $_, $F[2]-$F[1]'


join <(
    jq -r '"\(input_filename) \(.score)"' "$1"/*.??-??.jsonl |
        sed 's#\([^ ]*/\|^\)\([^ /]*\)\.\([a-z][a-z]-[a-z][a-z]\)\.jsonl #\2.\3 #' |
        sort
) <(
    jq -r '"\(input_filename) \(.score)"' "$2"/*.??-??.jsonl |
        sed 's#\([^ ]*/\|^\)\([^ /]*\)\.\([a-z][a-z]-[a-z][a-z]\)\.jsonl #\2.\3 #' |
        sort
) |
    perl -lane 'printf "%s\t%+.1f\n", $_, $F[2]-$F[1]' |
    column -t
