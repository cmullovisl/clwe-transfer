#!/usr/bin/env python3
import sys
import torch
from util import vec_to_vocab, load_field

vocab_path = sys.argv[1]
src_embeddings_path = sys.argv[2]
tgt_embeddings_path = sys.argv[3]
out_path = sys.argv[4]

field = load_field(vocab_path)
tgt_vocab = field['tgt'].base_field.vocab
src_vocab = field['src'].base_field.vocab

field['src'].base_field.vocab = vec_to_vocab(src_embeddings_path, tgt_vocab)
field['tgt'].base_field.vocab = vec_to_vocab(tgt_embeddings_path, tgt_vocab)
torch.save(field, out_path)
