import sys
import io
import re
from itertools import takewhile

import numpy as np
import torch
from torch.nn.functional import normalize
from torchtext.vocab import Vocab

def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    return model

def load_field(field_path):
    field = torch.load(field_path)
    return field

def vocab_to_vec(stoi, vectors, out_file):
    new_stoi = {s: i for s, i in stoi.items()}

    if out_file == '-':
        f = sys.stdout
    else:
        f = io.open(out_file, 'w', encoding='utf-8', newline='\n', errors='ignore')
    for s, i in new_stoi.items():
        vec = vectors[i]
        vec_str = ' '.join(['{:.5f}'.format(float(x)) for x in vec])
        f.write("{} {}\n".format(s, vec_str))
    f.close()

def load_vec(in_file):
    words = []
    all_words = set()
    vectors = []

    def _process_line(line):
        word, vector = line.rstrip().split(maxsplit=1)
        if word in all_words:
            return
        words.append(word)
        all_words.add(word)
        vector = np.fromstring(vector, sep=' ', dtype=np.float32)
        vectors.append(vector)

    with io.open(in_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        header = next(f).rstrip()
        if re.match('\d+ \d+$', header):
            vocab_size, emb_size = header.split(' ', 1)
            emb_size, vocab_size = int(emb_size), int(vocab_size)
        else:
            emb_size = len(header.split()) - 1
            _process_line(header)

        for line in f:
            _process_line(line)

    vectors = torch.from_numpy(np.stack(vectors))
    normalize(vectors, p=2, dim=1, out=vectors)
    return words, vectors

def vec_to_vocab(in_file, specials_vocab):
    specials = list(extract_specials(specials_vocab))
    special_indices = [specials_vocab.stoi[s] for s in specials]
    specials_vectors = specials_vocab.vectors[special_indices]

    words, vectors = load_vec(in_file)
    words = specials + words
    vectors = torch.cat((specials_vectors, vectors), dim=0)

    ctr = {s: len(words)+2-i for i, s in enumerate(words)}
    vocab = Vocab(ctr, specials=specials)
    vocab.set_vectors(vocab.stoi, vectors, dim=vectors.size(1))
    return vocab

def get_model_embeddings(model):
    emb_key = '{}.embeddings.make_embedding.emb_luts.0.0.weight'

    assert(model['opt'].share_embeddings)
    assert(model['opt'].share_decoder_embeddings)
    assert(model['model'][emb_key.format('encoder')].equal(
                model['model'][emb_key.format('decoder')]))
    #assert(model['model'][emb_key.format('decoder')].equal(
    #            model['model']['decoder.tgt_out_emb.weight']))

    #key = 'decoder.tgt_out_emb.weight'
    key = emb_key.format('decoder')
    vectors = model['model'][key]
    return vectors

def set_model_embeddings(model, vectors):
    emb_key = '{}.embeddings.make_embedding.emb_luts.0.0.weight'
    model['model'][emb_key.format('decoder')] = vectors
    model['model'][emb_key.format('encoder')] = vectors
    if 'decoder.tgt_out_emb.weight' in model['model']:
        model['model']['decoder.tgt_out_emb.weight'] = vectors
    else:
        model['generator']['0.weight'] = vectors

def counter_to_vocab(ctr, stoi, tgt_vectors, specials):
    """Construct a new vocab from a Counter object and sets the target vectors
    according to the indices in `stoi`.

    :ctr: Counter object with word frequencies
    :stoi: Dictionary mapping words to their indices
    :tgt_vectors: `Tensor` or `List` with `tgt_vectors[stoi[word]]` containing
                  the word vector for `word`
    :specials: list of special words
    :returns: the newly constructed Vocab object
    """
    vocab = Vocab(ctr, specials=specials)
    vocab.set_vectors(stoi, tgt_vectors, dim=tgt_vectors.size(1))
    return vocab

def filter_vocab(vocab, filter_func):
    counter = vocab.freqs

    new_ctr  = {s: freq for s, freq in counter.items() if filter_func(s)}
    specials = extract_specials(vocab)
    new_ctr.update(specials)

    return counter_to_vocab(new_ctr, vocab.stoi, vocab.vectors, list(specials))

def extract_specials(vocab):
    special = '(?![a-z][a-z]@)'
    RE_special = re.compile(special)
    specials = takewhile(RE_special.match, vocab.itos)
    return {s: vocab.freqs[s] for s in specials}

def n_specials(vocab):
    """Count the number of special tokens in the vocabulary, identifying a
    special word as any token lacking a language encoding prefix.
    """
    return len(extract_specials(vocab))

