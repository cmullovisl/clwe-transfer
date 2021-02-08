map_words() {
    lng="$1"
    ftmodel="$embeddingsdir/cc.${lng}.${embdim}.bin"
    matrix="$embeddingsdir/aligned.${lng}.vec-mat"
    python "$SCRIPT_DIR"/scripts/fasttext/map-vector.py "$ftmodel" "$matrix"
}

post_process() {
    lng="$1"
    sed "s/^/${lng}@/"
}

embeddings_to_vocab() {
    # assumes absence of duplicate entries
    # TODO handle header line if present
    cut -d' ' -f1
}

generate_specials() {
    embdim="$1"
    shift
    python "$SCRIPT_DIR"/scripts/vocab/generate-specials.py "$embdim" "$@" > "$datadir/specials.vec"
}

dataset_to_embeddings() {
    # reads data set from stdin
    lng="$1"
    sed 's/\@\@ //g' |
        "$SCRIPT_DIR"/scripts/vocab/extract-words.py |
        map_words "$lng" |
        post_process "$lng"
}

build_embeddings() {
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
    stage=base
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
