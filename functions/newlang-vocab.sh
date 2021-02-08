get_latest_model() {
    modeldir="$1"

    # output full model path
    echo -n "$modeldir/"
    ls -t "$modeldir" | head -1
}

vocab_from_model() {
    model="$1"
    vocabdir="$2"
    # TODO model specific vocabdir
    python "$SCRIPT_DIR"/scripts/vocab/extract-specials.py "$model" "$vocabdir/specials.pt"

    for dset in train dev test; do
        for lbase in "${baselanguages[@]}"; do
            for lnew in "${newlanguages[@]}"; do
                [[ $lbase = $lnew ]] && continue

                python "$SCRIPT_DIR"/scripts/vocab/new_src-tgt_vocab.py \
                    "$vocabdir/specials.pt" \
                    "$datadir/embeddings.$dset.$lbase.vec" \
                    "$datadir/embeddings.$dset.$lnew.vec" \
                    "$vocabdir/vocab.$dset.$lbase-$lnew.pt"

                python "$SCRIPT_DIR"/scripts/vocab/new_src-tgt_vocab.py \
                    "$vocabdir/specials.pt" \
                    "$datadir/embeddings.$dset.$lnew.vec" \
                    "$datadir/embeddings.$dset.$lbase.vec" \
                    "$vocabdir/vocab.$dset.$lnew-$lbase.pt"
            done
        done
    done
}
