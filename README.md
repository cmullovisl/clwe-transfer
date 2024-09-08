

# Getting started
## Dependencies
Install [dvc](https://github.com/iterative/dvc) with
```
pip install dvc
```

## Stage 1 - base model pre-training

Preparation:
```
git clone https://github.com/cmullovisl/clwe-transfer
cd clwe-transfer
git submodules update --init --recursive --remote
chmod +x ./fastText/alignment/align.py
ln -s ./fastText/alignment/align.py ./bin/fasttext_align
pip install .
pip install ./seq2seq-con/OpenNMT-py/
```

Define your list of base languages in [dvc.stage1.yaml](./dvc.stage1.yaml) under the `baselanguages` and `baselanguages_list` variables,
then run the pipeline with
```
dvc update -R data
ln -sf dvc.stage1.yaml dvc.yaml
dvc repro
```

## Stage 2 - learning a new language
First define your list of new (so far unseen) languages in [dvc.stage2.yaml](./dvc.stage2.yaml) under the `newlanguages` and `newlanguages_list` variables.

Then link stage 2 with
```
ln -sf dvc.stage2.yaml dvc.yaml
```

### Blind decoding from an unseen language
To "blindly" decode from the new languages run
```
dvc repro evauate_bleu_blindenc
```

To train via backtranslation run
```
dvc repro evauate_bleu_backtranslation_round1
```

## Trying out a pre-trained model
We also offer [word embedding alignments](https://huggingface.co/dz5035/clwe-transfer/tree/main/embeddings/align) and [pre-trained models](https://huggingface.co/dz5035/clwe-transfer/tree/main/models) trained on
English, German, Spanish, French and Italian on the multi-parallel TED data.

In addition to installing the repository and downloading the models/alignments,
you will have to download the Facebook fastText word embeddings for your source
and target language for inference with these models:
```
huggingface-cli download dz5035/clwe-transfer --local-dir . --local-dir-use-symlinks False
download_embeddings "en pt"
# prepare TED test data. Alternatively put your own test data into data/parallel/<dset>.pt-en.{pt,en}
dvc update -R data
prepare_parallel_data test "en pt"
# evaluate
dsets=test evauate_bleu base.softmax-nobias "pt" "en"
```

# Citation
```
@inproceedings{mullov-etal-2024-decoupled,
    title = "Decoupled Vocabulary Learning Enables Zero-Shot Translation from Unseen Languages",
    author = "Mullov, Carlos  and
      Pham, Quan  and
      Waibel, Alexander",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.362",
    pages = "6693--6709",
    abstract = "Multilingual neural machine translation systems learn to map sentences of different languages into a common representation space. Intuitively, with a growing number of seen languages the encoder sentence representation grows more flexible and easily adaptable to new languages. In this work, we test this hypothesis by zero-shot translating from unseen languages. To deal with unknown vocabularies from unknown languages we propose a setup where we decouple learning of vocabulary and syntax, i.e. for each language we learn word representations in a separate step (using cross-lingual word embeddings), and then train to translate while keeping those word representations frozen. We demonstrate that this setup enables zero-shot translation from entirely unseen languages. Zero-shot translating with a model trained on Germanic and Romance languages we achieve scores of 42.6 BLEU for Portuguese-English and 20.7 BLEU for Russian-English on TED domain. We explore how this zero-shot translation capability develops with varying number of languages seen by the encoder. Lastly, we explore the effectiveness of our decoupled learning strategy for unsupervised machine translation. By exploiting our model{'}s zero-shot translation capability for iterative back-translation we attain near parity with a supervised setting.",
}
```
