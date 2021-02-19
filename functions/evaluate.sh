translate() {
    local model="$1"
    local src="$2"
    local tgt="$3"
    local dset="$4"

    local GPU=0
    local beamsize=1
    local batchsize=4000

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
    local src="$1"
    local tgt="$2"
    local dset="$3"

    export LC_CTYPE=en_US.UTF-8
    export LC_ALL=en_US.UTF-8

    sed "s/$tgt@/ /g" "$translationsdir/$dset.$src-$tgt.$tgt.pred" |
        "$moses"/scripts/generic/multi-bleu.perl "$evaldir/$dset.$src-$tgt.$tgt"
}

evauate_bleu() {
    local src tgt dset modelname iteration
    local stage="$1"
    local model="$2"
    local sourcelanguages="$3"
    local targetlanguages="$4"

    local modelname_regex='^\(.*\)_step_\([0-9]*\)\.pt$'
    modelname="$(basename "$model" | sed "s|${modelname_regex}|\1|")"
    iteration="$(basename "$model" | sed "s|${modelname_regex}|\2|")"
    local result="$translationsdir/ted.multi_bleu.$stage.$iteration"

    mkdir -p "$translationsdir"
    echo "${modelname}_step_${iteration}.pt" | tee "$result"

    for src in $sourcelanguages; do
        for tgt in $targetlanguages; do
            [[ $src = "$tgt" ]] && continue

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
