map_words() {
    local lng="$1"
    local ftmodel="$embeddingsdir/cc.${lng}.${embdim}.bin"
    local matrix="$embeddingsdir/aligned.${lng}.vec-mat"
    python "$SCRIPT_DIR"/scripts/fasttext/map-vector.py "$ftmodel" "$matrix"
}

post_process() {
    local lng="$1"
    sed "s/^/${lng}@/"
}

embeddings_to_vocab() {
    # assumes absence of duplicate entries
    # TODO handle header line if present
    cut -d' ' -f1
}

generate_specials() {
    local embdim="$1"
    shift
    python "$SCRIPT_DIR"/scripts/vocab/generate-specials.py "$embdim" "$@" > "$datadir/specials.vec"
}

dataset_to_embeddings() {
    # reads data set from stdin
    local lng="$1"
    sed 's/\@\@ //g' |
        "$SCRIPT_DIR"/scripts/vocab/extract-words.py |
        map_words "$lng" |
        post_process "$lng"
}

build_embeddings() {
    local dset lng
    for dset in train dev test; do
        for lng in "$@"; do
            cat "$data_in/$dset.$lng" |
                dataset_to_embeddings "$lng" > "$datadir/embeddings.$dset.$lng.vec"
        done
    done
}

# XXX maybe make this function more generic: take stage, specials and languages
#     as argument
build_basesystem_embeddings() {
    local dset lng out
    local stage=base
    for dset in train dev test; do
        out="$datadir/embeddings.$stage.$dset.vec"
        echo "<linecount> ${embdim}" > "$out"
        cat "$datadir/specials.vec" >> "$out"

        for lng in "$@"; do
            cat "$datadir/embeddings.$dset.$lng.vec" >> "$out"
        done

        sed -i "1s/<linecount>/$(tail -n +2 "$out" | wc -l)/" "$out"
        tail -n +2 "$out" | embeddings_to_vocab > "$datadir/vocab.$stage.$dset.txt"
    done
}

extract_specials() {
    local model="$1"
    local outfile="$2"
    python "$SCRIPT_DIR"/scripts/vocab/extract-specials.py "$model" "$outfile"
}

vocab_from_specials() {
    local dset src tgt
    local specials="$1"
    local sourcelanguages="$2"
    local targetlanguages="$3"

    #for dset in train dev test; do
    for dset in dev test; do
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

build_newlang_vocab() {
    local basemodel="$1"
    local sourcelanguages="$2"
    local targetlanguages="$3"

    local basespecials="$savedir/specials.pt"
    local newspecials="$savedir/specials.pt"
    local newvocab="$savedir/data.vocab.pt"

    extract_specials "$basemodel" "$basespecials"

    local newlanguagecodes=()
    for lng in "${newlanguages[@]}"; do
        newlanguagecodes+=("#${lng}#")
    done

    python "$SCRIPT_DIR"/scripts/vocab/add-specials.py \
        "$basespecials" \
        "$newlanguagecodes" \
        "$newspecials"

    local dset="train"
    local sourceembeddings=()
    for lng in $sourcelanguages; do
        sourceembeddings+=("$datadir/embeddings.$dset.$lng.vec")
    done
    local targetmbeddings=()
    for lng in $targetlanguages; do
        targetembeddings+=("$datadir/embeddings.$dset.$lng.vec")
    done

    python "$SCRIPT_DIR"/scripts/vocab/new_src-tgt_vocab.py \
        "$newspecials" \
        <(cat "${sourceembeddings[@]}") \
        <(cat "${targetembeddings[@]}") \
        "$newvocab"
}
