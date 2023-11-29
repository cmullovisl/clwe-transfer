#!/usr/bin/env python3
import sys
import torch
from util import vec_to_vocab, load_field

N_SPECIALS = 10

vocab_path = sys.argv[1]
embeddings_path = sys.argv[2]
out_path = sys.argv[3]

field = load_field(vocab_path)
tgt_vocab = field['tgt'].base_field.vocab

#field['tgt'].base_field.vocab = vec_to_vocab(embeddings_path, tgt_vocab, N_SPECIALS)
field['tgt'].base_field.vocab = vec_to_vocab(embeddings_path, tgt_vocab)
torch.save(field, out_path)
