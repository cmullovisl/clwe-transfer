#!/usr/bin/env python
import sys
import torch

vocab_path = sys.argv[1]
out_dir = sys.argv[2]

fields = torch.load(vocab_path)
vocab = fields['tgt'].base_field.vocab
vectors = vocab.vectors

torch.save(vectors, out_dir + '/embeddings.tgt.pt')
