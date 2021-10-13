backtranslate_translate() {
    local model="$1"
    local src="$2"
    local tgt="$3"
    local dset="$4"

    local btdir="$savedir/backtranslations"

    local gpu="${gpu:-0}"
    local beamsize=4
    local batchsize=4000

    python -u "$onmt"/translate.py \
        -batch_type tokens \
        -batch_size "$batchsize" \
        -beam_size "$beamsize" \
        -gpu "$gpu" \
        -model "$model" \
        -src "$btdir/$dset.$src-$tgt.$src" \
        -output "$btdir/$dset.$src-$tgt.$tgt.pred" \
        -src_embeddings "$datadir/embeddings.$dset.$src.vec" \
        -tgt_embeddings "$datadir/embeddings.$dset.$tgt.vec"
}

backtranslate_parallel() {
    local src tgt dset
    local model="$1"
    local sourcelanguages="$2"
    local targetlanguages="$3"

    local btdir="$savedir/backtranslations"

    local num_gpus=6

    for dset in train dev; do
        rm -f "$btdir/$dset.$stage.tgt"
        rm -f "$btdir/$dset.$stage.src"

        local usegpu=0
        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = "$tgt" ]] && continue
                [[ -f $btdir/$dset.$src-$tgt.$src ]] || continue

                gpu="$usegpu" \
                    backtranslate_translate "$model" "$src" "$tgt" "$dset" &

                usegpu=$((usegpu + 1))
                if [[ $(jobs -p | wc -l) = $num_gpus ]]; then
                    wait
                    usegpu=0
                fi
            done
        done
    done

    wait

    for dset in train dev; do
        for src in $sourcelanguages; do
            for tgt in $targetlanguages; do
                [[ $src = "$tgt" ]] && continue
                [[ -f $btdir/$dset.$src-$tgt.$src ]] || continue

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

backtranslate() {
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

backtranslation_round() {
    local basemodel="$1"
    local backtranslationmodel="$2"
    local sourcelanguages="$3"
    local targetlanguages="$4"

    backtranslate "$backtranslationmodel" "$targetlanguages" "$sourcelanguages"

    build_newlang_vocab "$basemodel" "$sourcelanguages" "$targetlanguages"
    datadir="$savedir/backtranslations" \
        preprocess_reuse_vocab "$stage" "$savedir/data.vocab.pt"

    train_continue "$stage" "$model" "$backtranslationconfig" "$basemodel"

    rm -vf "$savedir"/data.train.[0-9]*.pt
    rm -vf "$savedir"/data.valid.[0-9]*.pt
    rm -vf "$savedir"/data.vocab.pt
}
