train() {
    # TODO move some of this into run.sh?
    local stage="$1"
    local model="$2"
    local savedir="$projectroot/saves.$stage"
    local modeldir="$savedir/models/$model"

    mkdir -p "$modeldir"

    # NOTE instead of using `-pre_word_vecs_enc "$savedir/embeddings.tgt.pt"`
    # let onmt load the embedding vectors from tgt_vocab.vectors
    python -u "$onmt"/train.py \
        -data "$savedir/data" \
        -save_model "$modeldir/$model" \
        -gpu_ranks 0 \
        -config "$configdir/$trainconfig.yml" 2>&1 |
            tee "$logdir/training.$stage.$model.log"
}

train_continue() {
    # TODO add `${additional_args}` array to `train()` and call `train` here
    # TODO move some of this into run.sh?
    local stage="$1"
    local model="$2"
    local trainfrom="$3"
    local trainconfig="$4"

    local savedir="$projectroot/saves.$stage"
    local modeldir="$savedir/models/$model"

    mkdir -p "$modeldir"

    # NOTE instead of using `-pre_word_vecs_enc "$savedir/embeddings.tgt.pt"`
    # let onmt load the embedding vectors from tgt_vocab.vectors
    python -u "$onmt"/train.py \
        -data "$savedir/data" \
        -save_model "$modeldir/$model" \
        -train_from "$trainfrom" \
        -new_vocab "$savedir/data.vocab.pt" \
        -modify_opts \
        -gpu_ranks 0 \
        -config "$configdir/$trainconfig.yml" 2>&1 |
            tee "$logdir/training.$stage.$model.log"
}
