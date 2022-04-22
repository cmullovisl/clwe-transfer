#!/usr/bin/env python
import sys
import re
from collections import Counter

tokens = re.split('[ \n]', sys.stdin.read().rstrip())
words = Counter(tokens)

vocab = sorted(words.items(), key=lambda x: x[1], reverse=True)
print('\n'.join([x[0] for x in vocab]))
