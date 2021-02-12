get_latest_model() {
    modeldir="$1"

    # output full model path
    echo -n "$modeldir/"
    ls -t "$modeldir" | head -1
}

extract_specials() {
    # TODO model specific vocabdir
    model="$1"
    vocabdir="$2"
    python "$SCRIPT_DIR"/scripts/vocab/extract-specials.py "$model" "$vocabdir/specials.pt"
}

vocab_from_specials() {
    specials="$1"
    sourcelanguages="$2"
    targetlanguages="$3"

    for dset in train dev test; do
        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = "$tgt" ]] && continue

                python "$SCRIPT_DIR"/scripts/vocab/new_src-tgt_vocab.py \
                    "$specials" \
                    "$datadir/embeddings.$dset.$src.vec" \
                    "$datadir/embeddings.$dset.$tgt.vec" \
                    "$vocabdir/vocab.$dset.$src-$tgt.pt"
            done
        done
    done
}
