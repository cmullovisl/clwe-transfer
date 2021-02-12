concat_data() {
    local src tgt dset
    local stage="$1"

    rm -f "$datadir/train.$stage.unshuf.src"
    rm -f "$datadir/train.$stage.unshuf.tgt"
    rm -f "$datadir/train.$stage.src"
    rm -f "$datadir/train.$stage.tgt"
    rm -f "$datadir/dev.$stage.src"
    rm -f "$datadir/dev.$stage.tgt"

    # explicitly do not quote
    for src in $2; do
        for tgt in $3; do
            [[ $src = "$tgt" ]] && continue
            echo "processing $src-$tgt"

            # TODO move source/target preprocessing pipeline to separate functions
            # TODO don't handle shuffling as special case ~> process data sets in for loop
            dset="train"
            cat "$data_in/$dset.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g" >> "$datadir/$dset.$stage.unshuf.src"
            cat "$data_in/$dset.$src-$tgt.$tgt" | sed 's/\@\@ //g' | sed "s/^/${tgt}@/; s/ / ${tgt}@/g" >> "$datadir/$dset.$stage.unshuf.tgt"

            dset="dev"
            cat "$data_in/$dset.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g" >> "$datadir/$dset.$stage.src"
            cat "$data_in/$dset.$src-$tgt.$tgt" | sed 's/\@\@ //g' | sed "s/^/${tgt}@/; s/ / ${tgt}@/g" >> "$datadir/$dset.$stage.tgt"
        done
    done

    echo "shuffling..."
    "$SCRIPT_DIR"/scripts/preprocess/parallel-shuffle.sh \
        "$datadir/train.$stage.unshuf.src" "$datadir/train.$stage.unshuf.tgt" \
        "$datadir/train.$stage.src" "$datadir/train.$stage.tgt"

    rm -f "$datadir/train.$stage.unshuf.src"
    rm -f "$datadir/train.$stage.unshuf.tgt"
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

                cat "$data_in/$dset.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g"  > "$evaldir/$dset.$src-$tgt.$src"
                cat "$data_in/$dset.$src-$tgt.$tgt" | sed 's/\@\@ //g' > "$evaldir/$dset.$src-$tgt.$tgt"

                "$SCRIPT_DIR"/scripts/preprocess/purge-empty-lines.sh "" \
                    "$evaldir/$dset.$src-$tgt.$src" "$evaldir/$dset.$src-$tgt.$tgt"
            done
        done
    done
}
