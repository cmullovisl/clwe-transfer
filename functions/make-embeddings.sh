download_fasttext_model() {
    local lng
    lng="$1"
    fasttext_url="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"
    model=/cc.en.300.bin.gz

    # TODO limit amount of vocab words downloaded
    curl "${fasttext_url}/cc.${lng}.${embdim}.bin.gz" > "$embeddingsdir/cc.${lng}.${embdim}.bin.gz"
    curl "${fasttext_url}/cc.${lng}.${embdim}.vec.gz" > "$embeddingsdir/cc.${lng}.${embdim}.vec.gz"
    # TODO decompress
}

download_embeddings() {
    for lng in "$@"; do
        [[ -f ${embeddingsdir}/cc.${lng}.${embdim}.bin.gz ]] || download_fasttext_model "$lng"
        # TODO copy dictionaries for language pairs
        [[ -f ${embeddingsdir}/${lng}-*.txt ]] || download_dictionaries "$lng"
    done
}

compute_alignments() {
    pivot="$1"
    shift

    for lng in "$@"; do
        [[ $pivot = $lng ]] && continue

        python -u "$fasttext"/alignment/align.py \
            --src_emb "$embeddingsdir/cc.$lng.300.vec" \
            --tgt_emb "$embeddingsdir/cc.$pivot.300.vec" \
            --dico_train "$dictdir/$lng-$pivot.txt" \
            --dico_test  "$dictdir/$lng-$pivot.0-5000.txt" \
            --output "$embeddingsdir/aligned.$lng.vec" 2>&1 | tee "$logdir/align.$lng.log"

        #[[ $deleteftvec ]] && rm -f "$embeddingsdir/cc.${lng}.${embdim}.vec.gz"
    done

    # XXX
    # "compute" dummy alignment for pivot?
}
