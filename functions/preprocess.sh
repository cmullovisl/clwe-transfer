shuffle_corpus() {
    local dset="$1"

    echo "shuffling..."
    "$SCRIPT_DIR"/scripts/preprocess/parallel-shuffle.sh \
        "$datadir/$dset.$stage.src" "$datadir/$dset.$stage.tgt" \
        "$datadir/$dset.$stage.shuf.src" "$datadir/$dset.$stage.shuf.tgt"

    mv "$datadir/$dset.$stage.shuf.src" "$datadir/$dset.$stage.src"
    mv "$datadir/$dset.$stage.shuf.tgt" "$datadir/$dset.$stage.tgt"
}

strip_bpe() {
    sed 's/\@\@ //g'
}

preprocess_source_data() {
    local src="$1"
    local tgt="$2"
    strip_bpe | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g"
}

preprocess_target_data() {
    [[ $# = 2 ]] && shift
    local tgt="$1"
    strip_bpe | sed "s/^/${tgt}@/; s/ / ${tgt}@/g"
}

concat_data() {
    local src tgt dset
    local stage="$1"

    rm -f "$datadir/train.$stage.src"
    rm -f "$datadir/train.$stage.tgt"
    rm -f "$datadir/dev.$stage.src"
    rm -f "$datadir/dev.$stage.tgt"

    # explicitly do not quote
    for src in $2; do
        for tgt in $3; do
            [[ $src = "$tgt" ]] && continue
            echo "processing $src-$tgt"
            for dset in train dev; do
                cat "$data_in/$dset.$src-$tgt.$src" | preprocess_source_data "$src" "$tgt" >> "$datadir/$dset.$stage.src"
                cat "$data_in/$dset.$src-$tgt.$tgt" | preprocess_target_data "$src" "$tgt" >> "$datadir/$dset.$stage.tgt"
            done
        done
    done

    shuffle_corpus "train"
}

preprocess() {
    local stage="$1"
    local dset=train

    # TODO move this into run.sh?
    local savedir="$projectroot/saves.$stage"
    mkdir -p "$savedir"

    python -u "$onmt"/preprocess.py \
        -train_src "$datadir/train.$stage.src" \
        -train_tgt "$datadir/train.$stage.tgt" \
        -valid_src "$datadir/dev.$stage.src" \
        -valid_tgt "$datadir/dev.$stage.tgt" \
        -src_vocab "$datadir/vocab.$stage.$dset.txt" \
        -tgt_vocab "$datadir/vocab.$stage.$dset.txt" \
        -save_data "$savedir/data" \
        -tgt_emb "$datadir/embeddings.$stage.$dset.vec" \
        -src_vocab_size 1000000 \
        -tgt_vocab_size 1000000 \
        -src_seq_length 100 \
        -tgt_seq_length 100 |& tee "$logdir/preprocess.log"
}

preprocess_evaluation_data() {
    local dset src tgt
    local sourcelanguages="$1"
    local targetlanguages="$2"

    for dset in dev test; do
        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = "$tgt" ]] && continue

                cat "$data_in/$dset.$src-$tgt.$src" | preprocess_source_data "$src" "$tgt" > "$evaldir/$dset.$src-$tgt.$src"
                cat "$data_in/$dset.$src-$tgt.$tgt" | strip_bpe > "$evaldir/$dset.$src-$tgt.$tgt"

                "$SCRIPT_DIR"/scripts/preprocess/purge-empty-lines.sh "" \
                    "$evaldir/$dset.$src-$tgt.$src" "$evaldir/$dset.$src-$tgt.$tgt"
            done
        done
    done
}
