#!/usr/bin/env python
import sys
import re

#for line in sys.stdin:
#    # replace duplicates at end of sentence with "."
#    ppline = re.sub("([^ ]+)( \\1)+$", "\\1 .", line.rstrip())
#    # remove all other duplicates
#    ppline = re.sub("([^ ]+)( \\1)+", "\\1", ppline)
#
#    print(ppline)


langcode = sys.argv[1]

for line in sys.stdin:
    ppline = line.rstrip()
    # replace duplicates at end of sentence with "."
    ppline = re.sub("{0}@([^ ]+)( {0}@\\1)+$".format(langcode), "\\1 .", ppline)
    # remove all other duplicates
    #ppline = re.sub("{0}@([^ ]+)( {0}@\\1)+".format(langcode), "\\1", ppline)
    ppline = re.sub("{0}@([^ ]+) {0}@\\1( {0}@\\1)+".format(langcode), "\\1", ppline)
    # remove duplicates consisting of two words
    #ppline = re.sub("{0}@([^ ]+) {0}@([^ ]+)( {0}@\\1 {0}@\\2)+".format(langcode), "\\1 \\2", ppline)
    # remove language codes
    ppline = re.sub("{0}@".format(langcode), "", ppline)

    print(ppline)
