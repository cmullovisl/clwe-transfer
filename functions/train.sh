train() {
    local stage="$1"
    local model="$2"
    local trainconfig="$3"
    shift 3
    local additional_args=("$@")

    local savedir="$projectroot/saves.$stage"
    local modeldir="$savedir/models/$model"

    mkdir -p "$modeldir"

    # `${additional_args[@]+"${additional_args[@]}"}` is an ugly workaround for
    # the expansion of empty arrays failing in older versions of Bash (see
    # https://stackoverflow.com/questions/7577052/bash-empty-array-expansion-with-set-u/61551944#61551944)
    python -u "$onmt"/train.py \
        -data "$savedir/data" \
        -save_model "$modeldir/$model" \
        -gpu_ranks 0 \
        -config "$configdir/$trainconfig.yml" \
        ${additional_args[@]+"${additional_args[@]}"} 2>&1 |
            tee "$logdir/training.$stage.$model.log"
}

train_continue() {
    local stage="$1"
    local model="$2"
    local trainconfig="$3"
    local trainfrom="$4"

    train "$stage" \
        "$model" \
        "$trainconfig" \
        -train_from "$trainfrom" \
        -new_vocab "$savedir/data.vocab.pt"
}
