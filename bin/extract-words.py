#!/usr/bin/env python
import sys
from collections import Counter

tokens = sys.stdin.read().rstrip().replace("\n", " ").split(" ")
counts = Counter(tokens).most_common()
print('\n'.join(x for x, _ in counts))
