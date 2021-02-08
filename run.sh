#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TODO use getopts
# TODO make configdir relocatable
readonly configdir="$SCRIPT_DIR/config"
readonly config="${1:-default}"
source "$configdir/$config"

source "$SCRIPT_DIR"/functions/make-embeddings.sh
source "$SCRIPT_DIR"/functions/prepare-data.sh
source "$SCRIPT_DIR"/functions/build-vocab.sh
source "$SCRIPT_DIR"/functions/preprocess.sh
source "$SCRIPT_DIR"/functions/train.sh
source "$SCRIPT_DIR"/functions/newlang-vocab.sh
source "$SCRIPT_DIR"/functions/evaluate.sh

# TODO move all environment variables to config/vars
projectroot="${2:-$SCRIPT_DIR/saves}"
logdir="$projectroot"/logs
embeddingsdir="$projectroot"/embeddings
dictdir="$embeddingsdir"/dicts
datadir="$projectroot"/data
evaldir="$projectroot/evaldata"

data_in="$SCRIPT_DIR"/data
fasttext="$SCRIPT_DIR"/fastText
onmt="$SCRIPT_DIR"/onmt
moses="$SCRIPT_DIR"/mosesdecoder


# Start basemodel stage
mkdir -p "$logdir"
stage=base
savedir="$projectroot/saves.$stage"


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
concat_data "$stage" "${baselanguages[*]}" "${baselanguages[*]}"

echo "Building PyTorch training shards and vocabulary..."
preprocess "$stage"



## Train basemodel
echo "Training basesystem..."
train "$stage" "$model"



## Download and process monolingual new language data and embeddings
#...



## Build new language vocab
echo "Building new language vocabularies..."
vocabdir="$savedir"/vocabs
mkdir -p "$vocabdir"
build_embeddings "${newlanguages[@]}"

# TODO model specific vocabdir
basemodel="$(get_latest_model "$savedir/models/$model")"
vocab_from_model "$basemodel" "$vocabdir"



# Evaluate basesystem performance
#evauate_bleu "$stage" "$model" "$iteration" ...
translationsdir="$savedir/translations/$model"
mkdir -p "$translationsdir"
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
