translate() {
    model="$1"
    src="$2"
    tgt="$3"
    dset="$4"

    GPU=0
    beamsize=1
    batchsize=4000

    python -u "$onmt"/translate.py \
        -decode_loss cosine \
        -batch_type tokens \
        -batch_size "$batchsize" \
        -beam_size "$beamsize" \
        -gpu "$GPU" \
        -model "$model" \
        -src "$evaldir/$dset.$src-$tgt.$src" \
        -output "$translationsdir/$dset.$src-$tgt.$tgt.pred" \
        -new_vocab "$vocabdir/vocab.$dset.$src-$tgt.pt"
}

calculate_bleu() {
    src="$1"
    tgt="$2"
    dset="$3"

    export LC_CTYPE=en_US.UTF-8
    export LC_ALL=en_US.UTF-8

    sed "s/$tgt@/ /g" "$translationsdir/$dset.$src-$tgt.$tgt.pred" |
        "$moses"/scripts/generic/multi-bleu.perl "$evaldir/$dset.$src-$tgt.$tgt"
}

evauate_bleu() {
    stage="$1"
    model="$2"
    sourcelanguages="$3"
    targetlanguages="$4"

    modelname_regex='^\(.*\)_step_\([0-9]*\)\.pt$'
    modelname="$(basename "$model" | sed "s|${modelname_regex}|\1|")"
    iteration="$(basename "$model" | sed "s|${modelname_regex}|\2|")"
    result="$translationsdir/ted.multi_bleu.$iteration"

    mkdir -p "$translationsdir"

    for src in $sourcelanguages; do
        for tgt in $targetlanguages; do
            [[ $src = $tgt ]] && continue

            echo "$src-$tgt" | tee -a "$result"
            for dset in dev test
            do
                [[ -f $evaldir/$dset.$src-$tgt.$src ]] || continue

                translate "$model" "$src" "$tgt" "$dset"
                calculate_bleu "$src" "$tgt" "$dset" | tee -a "$result"
            done
        done
    done

    cat "$result"
}
