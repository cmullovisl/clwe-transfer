

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
cp -r scripts/vocab onmt_vocab_utils
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
