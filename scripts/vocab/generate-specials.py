#!/usr/bin/env python3
import sys
import io
import numpy as np

emb_size = int(sys.argv[1])
languages = sys.argv[2:]
if not languages:
    print("Warning: no languages specified", file=sys.stderr)

words = ["<blank>", "<unk>", "<s>", "</s>"]
words += ['#{}#'.format(lng) for lng in languages]

vectors = [np.random.normal(scale=emb_size ** -0.5, size=emb_size) for _ in range(len(words))]
lines = []
for w, v in zip(words, vectors):
    vec_str = ' '.join(["{:.4f}".format(x) for x in v])
    lines.append("{} {}".format(w, vec_str))

print("\n".join(lines))
