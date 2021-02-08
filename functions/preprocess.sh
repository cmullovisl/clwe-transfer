concat_data() {
    stage="$1"

    rm -f "$datadir/corpus.$stage.unshuf.src"
    rm -f "$datadir/corpus.$stage.unshuf.tgt"
    rm -f "$datadir/corpus.$stage.src"
    rm -f "$datadir/corpus.$stage.tgt"
    rm -f "$datadir/dev.$stage.src"
    rm -f "$datadir/dev.$stage.tgt"

    # explicitly do not quote
    for src in $2; do
        for tgt in $3; do
            [[ $src = $tgt ]] && continue
            echo "processing $src-$tgt"

            # TODO move source/target preprocessing pipeline to separate functions
            cat "$data_in/train.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g" >> "$datadir/corpus.$stage.unshuf.src"
            cat "$data_in/train.$src-$tgt.$tgt" | sed 's/\@\@ //g' | sed "s/^/${tgt}@/; s/ / ${tgt}@/g" >> "$datadir/corpus.$stage.unshuf.tgt"

            cat "$data_in/dev.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g" >> "$datadir/dev.$stage.src"
            cat "$data_in/dev.$src-$tgt.$tgt" | sed 's/\@\@ //g' | sed "s/^/${tgt}@/; s/ / ${tgt}@/g" >> "$datadir/dev.$stage.tgt"
        done
    done

    echo "shuffling..."
    "$SCRIPT_DIR"/scripts/preprocess/parallel-shuffle.sh \
        "$datadir/corpus.$stage.unshuf.src" "$datadir/corpus.$stage.unshuf.tgt" \
        "$datadir/corpus.$stage.src" "$datadir/corpus.$stage.tgt"

    rm -f "$datadir/corpus.$stage.unshuf.src"
    rm -f "$datadir/corpus.$stage.unshuf.tgt"
}

preprocess() {
    stage="$1"
    dset=train

    #special_embeddings="$datadir/specials.vec"
    #word_embeddings="$datadir/embeddings.$stage.$dset.vec"
    #speciallines=$(wc -l < "$special_embeddings")
    #vocablines=$(wc -l < "$word_embeddings")

    # TODO move this into run.sh?
    savedir="$projectroot/saves.$stage"
    mkdir -p "$savedir"

    python -u "$onmt"/preprocess.py \
        -train_src "$datadir/corpus.$stage.src" \
        -train_tgt "$datadir/corpus.$stage.tgt" \
        -valid_src "$datadir/dev.$stage.src" \
        -valid_tgt "$datadir/dev.$stage.tgt" \
        -src_vocab "$datadir/vocab.$stage.$dset.txt" \
        -tgt_vocab "$datadir/vocab.$stage.$dset.txt" \
        -save_data "$savedir/data" \
        -tgt_emb "$datadir/embeddings.$stage.$dset.vec" \
        -src_vocab_size 1000000 \
        -tgt_vocab_size 1000000 \
        -src_seq_length 100 \
        -tgt_seq_length 100 |& tee $logdir/preprocess.log

        # for use with separate specials and embeddings .vec files
        #-tgt_emb <(echo "$((speciallines + vocablines)) $embdim"; cat "$special_embeddings" "$word_embeddings") \
}

preprocess_evaluation_data() {
    sourcelanguages="$1"
    targetlanguages="$2"

    for dset in dev test; do
        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = $tgt ]] && continue

                cat "$data_in/$dset.$src-$tgt.$src" | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g"  > "$evaldir/$dset.$src-$tgt.$src"
                cat "$data_in/$dset.$src-$tgt.$tgt" | sed 's/\@\@ //g' > "$evaldir/$dset.$src-$tgt.$tgt"

                "$SCRIPT_DIR"/scripts/preprocess/purge-empty-lines.sh "" \
                    "$evaldir/$dset.$src-$tgt.$src" "$evaldir/$dset.$src-$tgt.$tgt"

                #cat $data_in/dev.$src-$tgt.$src | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g"  > $evaldir/dev.$src-$tgt.$src
                #cat $data_in/dev.$src-$tgt.$tgt | sed 's/\@\@ //g' > $evaldir/dev.$src-$tgt.$tgt
                #
                #$SCRIPT_DIR/scripts/preprocess/purge-empty-lines.sh "" \
                #    $evaldir/dev.$src-$tgt.$src $evaldir/dev.$src-$tgt.$tgt
                #
                #cat $data_in/test.$src-$tgt.$src | sed 's/\@\@ //g' | sed "s/^/#${tgt}# /" | sed "s/ / ${src}@/g" > $evaldir/test.$src-$tgt.$src
                #cat $data_in/test.$src-$tgt.$tgt | sed 's/\@\@ //g' > $evaldir/test.$src-$tgt.$tgt
                #
                #$SCRIPT_DIR/scripts/preprocess/purge-empty-lines.sh "" \
                #    $evaldir/test.$src-$tgt.$src $evaldir/test.$src-$tgt.$tgt
            done
        done
    done
}
