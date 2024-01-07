#!/usr/bin/env python3
import sys
import torch
from torch.nn.functional import normalize
from torchtext.vocab import Vocab
from onmt_vocab_utils import (
    load_vec,
    load_field,
    extract_specials,
)

vocab_path = sys.argv[1]
requested_specials = sys.argv[2]
out_file = sys.argv[3]

def add_specials(vocab, new_specials):
    counter = vocab.freqs
    specials = list(extract_specials(vocab))
    new_specials = [s for s in new_specials if s not in specials]
    if len(new_specials) == 0:
        return vocab
    old_vectors = vocab.vectors
    emb_size = old_vectors.size(1)
    # randomly initialize the new special vectors
    new_vectors = normalize(torch.randn(len(new_specials), emb_size))

    # append new vectors in between original specials and non-specials
    vectors = torch.cat([old_vectors[:len(specials)],
                         new_vectors,
                         old_vectors[len(specials):]], dim=0)

    # specials frequencies do not matter since specials are deleted from the
    # the counter anyways (see `torchtext.Vocab.__init__`)
    counter.update({s: 1 for s in new_specials})

    new_vocab = Vocab(counter,
                      specials_first=True,
                      specials=specials + new_specials)

    new_vocab.set_vectors(new_vocab.stoi, vectors, dim=emb_size)
    return new_vocab



new_specials = requested_specials.split()
fields = load_field(vocab_path)
tgt_vocab = fields['tgt'].base_field.vocab
new_vocab = add_specials(tgt_vocab, new_specials)

fields['tgt'].base_field.vocab = new_vocab
#fields['src'].base_field.vocab = new_vocab
torch.save(fields, out_file)
