backtranslate_translate() {
    local model="$1"
    local src="$2"
    local tgt="$3"
    local dset="$4"

    local btdir="$savedir/backtranslations"

    local GPU=0
    local beamsize=4
    local batchsize=4000

    # TODO for dset in train dev; do
    python -u "$onmt"/translate.py \
        -batch_type tokens \
        -batch_size "$batchsize" \
        -beam_size "$beamsize" \
        -gpu "$GPU" \
        -model "$model" \
        -src "$btdir/$dset.$src-$tgt.$src" \
        -output "$btdir/$dset.$src-$tgt.$tgt.pred" \
        -src_embeddings "$datadir/embeddings.$dset.$src.vec" \
        -tgt_embeddings "$datadir/embeddings.$dset.$tgt.vec"
}

backtranslation_round() {
    local src tgt dset
    local model="$1"
    local sourcelanguages="$2"
    local targetlanguages="$3"

    local btdir="$savedir/backtranslations"

    for dset in train dev; do
        rm -f "$btdir/$dset.$stage.tgt"
        rm -f "$btdir/$dset.$stage.src"

        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = "$tgt" ]] && continue
                [[ -f $btdir/$dset.$src-$tgt.$src ]] || continue

                backtranslate_translate "$model" "$src" "$tgt" "$dset"

                cat "$btdir/$dset.$src-$tgt.$tgt.pred" |
                    python "$SCRIPT_DIR"/scripts/evaluate/postprocess.py "$tgt" |
                    sed "s/^/#${src}# /" >> "$btdir/$dset.$stage.src"

                cat "$btdir/$dset.$src-$tgt.$src" |
                    sed "s/^#${tgt}# //" >> "$btdir/$dset.$stage.tgt"
            done
        done
    done

    # FIXME defined somewhere else
    datadir="$btdir" shuffle_corpus "train"
}
