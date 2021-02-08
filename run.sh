#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
basedir="$SCRIPT_DIR"

# TODO use getopts
# TODO make configdir relocatable
configdir="$basedir/config"
config="${1:-default}"
source "$configdir/$config"

source "$basedir"/functions/make-embeddings.sh
source "$basedir"/functions/prepare-data.sh
source "$basedir"/functions/build-vocab.sh
source "$basedir"/functions/preprocess.sh
source "$basedir"/functions/train.sh
source "$basedir"/functions/newlang-vocab.sh
source "$basedir"/functions/evaluate.sh

# TODO move all environment variables to config/vars
savedir="${2:-$basedir/saves}"
logdir="$savedir"/logs
embeddingsdir="$savedir"/embeddings
dictdir="$embeddingsdir"/dicts
data_in="$basedir"/data
datadir="$savedir"/data
fasttext="$basedir"/fastText
onmt="$basedir"/onmt
moses="$basedir"/mosesdecoder

stage=base
SAVEDIR="$savedir/saves.$stage"
vocabdir="$SAVEDIR"/vocabs
translationsdir="$SAVEDIR/translations/$model"
evaldir="$savedir/evaldata"

mkdir -p "$logdir"


## Compute Cross-Lingual Word Embeddings
echo "Computing Cross-Linugal Word Embeddings..."
mkdir -p "$embeddingsdir"

download_embeddings "${baselanguages[@]}"
compute_alignments "${clwepivot}" "${baselanguages[@]}"



## Download and build data
echo "Downloading and building data..."
mkdir -p "$data_in"
prepare_data



## Build basesystem vocabulary
echo "Building basesystem vocabulary..."
mkdir -p "$datadir"
generate_specials "$embdim" "${baselanguages[@]}"
# XXX separate .vec file for each language?
build_embeddings "${baselanguages[@]}"
build_basesystem_embeddings "${baselanguages[@]}"



## Preprocess
echo "Concatenating training corpus..."
concat_data "base" "${baselanguages[*]}" "${baselanguages[*]}"

echo "Building PyTorch training shards and vocabulary..."
preprocess "base"




## Train basemodel
echo "Training basesystem..."
train "base" "$model"



## Download and process monolingual new language data and embeddings
#...



## Build new language vocab
echo "Building new language vocabularies..."
mkdir -p "$vocabdir"
build_embeddings "${newlanguages[@]}"

# TODO model specific vocabdir
basemodel="$(get_latest_model "$SAVEDIR/models/$model")"
vocab_from_model "$basemodel" "$vocabdir"



# Evaluate basesystem performance
#evauate_bleu "$stage" "$model" "$iteration" ...
mkdir -p "$evaldir"

echo "Evaluating base language BLEU scores..."
preprocess_evaluation_data "${baselanguages[*]}" "${baselanguages[*]}"
evauate_bleu "$stage" "$basemodel" "${baselanguages[*]}" "${baselanguages[*]}"

echo "Evaluating blind encoding BLEU scores..."
prepare_evaluation_data "${newlanguages[*]}" "${baselanguages[*]}"
preprocess_evaluation_data "${newlanguages[*]}" "${baselanguages[*]}"
evauate_bleu "$stage" "$basemodel" "${newlanguages[*]}" "${baselanguages[*]}"

echo "Evaluating blind decoding BLEU scores..."
prepare_evaluation_data "${baselanguages[*]}" "${newlanguages[*]}"
preprocess_evaluation_data "${baselanguages[*]}" "${newlanguages[*]}"
evauate_bleu "$stage" "$basemodel" "${baselanguages[*]}" "${newlanguages[*]}"

# vocabularies for lnew -> lnew missing
#evauate_bleu "$stage" "$basemodel" "${newlanguages[*]}" "${newlanguages[*]}"
