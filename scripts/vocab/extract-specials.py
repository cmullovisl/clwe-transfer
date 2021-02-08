#!/usr/bin/env python3
import sys
import torch

from util import get_model_embeddings, load_model, filter_vocab, extract_specials
#from torchtext.vocab import Vocab

if len(sys.argv) < 3:
    print(
'''
{} MODEL_PATH OUT_FILE

From a model checkpoint extract the specials from the word embeddings and save
them to a separate field object. Only the field object target vocab is set, the 
source vocab is left unmodified.
'''.format(sys.argv[0])
        )
    exit(0)

model_path = sys.argv[1]
out_file   = sys.argv[2]

model = load_model(model_path)
field = model['vocab']
tgt_vocab = field['tgt'].base_field.vocab
tgt_vectors = get_model_embeddings(model)

#N_SPECIALS = 4
#
#specials = tgt_vocab.itos[:N_SPECIALS]
##new_stoi = {s : tgt_vocab.stoi[s] for s in specials}
#new_stoi = {i : s for i, s in enumerate(specials)}
#
#new_vocab = ...


# filter out all words except for the specials
new_vocab = filter_vocab(tgt_vocab, lambda x: False)
#new_vocab = extract_specials(tgt_vocab)

field['tgt'].base_field.vocab = new_vocab
torch.save(field, out_file)
