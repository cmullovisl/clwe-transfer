translate() {
    #stage="$1"
    #model="$2"
    #src="$3"
    #tgt="$4"
    #dset="$5"

    model="$1"
    src="$2"
    tgt="$3"
    dset="$4"

    #SAVEDIR="saves.$stage"
    evaldir="$savedir/evaldata"
    #modeldir="$SAVEDIR/models/$model"
    #outdir="$SAVEDIR/translations/$model"

    GPU=0
    beamsize=1
    batchsize=4000
    #m="${model}_step_${iteration}.pt"

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
        #-new_vocab "$SAVEDIR/data.vocab.$tgt.pt"
        #-model "$modeldir/$m" \
}

calculate_bleu() {
    #stage="$1"
    #modelname="$2"
    #src="$3"
    #tgt="$4"
    #dset="$5"

    src="$1"
    tgt="$2"
    dset="$3"

    #SAVEDIR="saves.$stage"
    #translationsdir="$SAVEDIR/translations/$modelname"
    evaldir="$savedir/evaldata"

    export LC_CTYPE=en_US.UTF-8
    export LC_ALL=en_US.UTF-8

    sed "s/$tgt@/ /g" "$translationsdir/$dset.$src-$tgt.$tgt.pred" |
        "$moses"/scripts/generic/multi-bleu.perl "$evaldir/$dset.$src-$tgt.$tgt"
}

evauate_bleu() {
    stage="$1"
    model="$2"
    #iteration="$3"
    sourcelanguages="$3"
    targetlanguages="$4"

    modelname_regex='^\(.*\)_step_\([0-9]*\)\.pt$'
    modelname="$(basename "$model" | sed "s|${modelname_regex}|\1|")"
    iteration="$(basename "$model" | sed "s|${modelname_regex}|\2|")"
    result="$translationsdir/ted.multi_bleu.$iteration"

    mkdir -p "$translationsdir"
    #echo "$modelname $iteration" > "$result"

    for src in $sourcelanguages; do
        for tgt in $targetlanguages; do
            [[ $src = $tgt ]] && continue

            echo "$src-$tgt" | tee -a "$result"
            for dset in dev test
            do
                [[ -f $evaldir/$dset.$src-$tgt.$src ]] || continue

                #translate "$stage" "$model" "$src" "$tgt" "$dset"
                #calculate_bleu "$stage" "$modelname" "$src" "$tgt" "$dset" | tee -a "$result"
                translate "$model" "$src" "$tgt" "$dset"
                calculate_bleu "$src" "$tgt" "$dset" | tee -a "$result"
            done
        done
    done

    cat "$result"
}
