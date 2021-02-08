train() {
    # TODO move some of this into run.sh?
    stage="$1"
    model="$2"
    SAVEDIR="$savedir/saves.$stage"
    modeldir="$SAVEDIR/models/$model"

    mkdir -p "$modeldir"

    # NOTE instead of using `-pre_word_vecs_enc "$SAVEDIR/embeddings.tgt.pt"`
    # let onmt load the embedding vectors from tgt_vocab.vectors
    python -u "$onmt"/train.py \
        -data "$SAVEDIR/data" \
        -save_model "$modeldir/$model" \
        -gpu_ranks 0 \
        -config "$configdir/$trainconfig.yml" 2>&1 |
            tee "$logdir/training.$stage.$model.log"

        #-config "$configdir/config-${model}.yml" 2>&1 |
}
