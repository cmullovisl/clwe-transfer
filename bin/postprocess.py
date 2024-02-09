#!/usr/bin/env python
import sys
import re


langcode = sys.argv[1]

aggressive = False
unclean_count = 0
for i, line in enumerate(sys.stdin):
    ppline = line.rstrip()

    if (re.search("{0}@([^ ]+) {0}@\\1( {0}@\\1)+".format(langcode), ppline)
            or re.search("(({0}@[^ ]+ {0}@[^ ]+)(?: {0}@[^ ]+)*?) \\1( \\1)+".format(langcode), ppline)):
        unclean_count += 1
    aggressive = unclean_count / float(i + 1) > 0.1

    # replace duplicates at end of sentence with "."
    ppline = re.sub("{0}@([^ ]+)( {0}@\\1)+$".format(langcode), "\\1 .", ppline)
    # remove all other duplicates
    if aggressive:
        ppline = re.sub("{0}@([^ ]+)( {0}@\\1)+".format(langcode), "\\1", ppline)
    else:
        ppline = re.sub("{0}@([^ ]+) {0}@\\1( {0}@\\1)+".format(langcode), "\\1", ppline)
    # remove duplicates consisting of two words
    #ppline = re.sub("{0}@([^ ]+) {0}@([^ ]+)( {0}@\\1 {0}@\\2)+".format(langcode), "\\1 \\2", ppline)
    # remove instances of 3+ repititions of n-grams (for n >= 2)
    if aggressive:
        ppline = re.sub("(({0}@[^ ]+ {0}@[^ ]+)(?: {0}@[^ ]+)*?)( \\1)+".format(langcode), "\\1", ppline)
    else:
        ppline = re.sub("(({0}@[^ ]+ {0}@[^ ]+)(?: {0}@[^ ]+)*?) \\1( \\1)+".format(langcode), "\\1", ppline)
    # remove language codes
    ppline = re.sub("{0}@".format(langcode), "", ppline)

    print(ppline)
