download_fasttext_model() {
    local lng="$1"
    local fasttext_url="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"
    local nvectors=200000

    curl "${fasttext_url}/cc.${lng}.${embdim}.bin.gz" |
        gunzip > "$embeddingsdir/cc.${lng}.${embdim}.bin"

    # Download only the first `$nvectors` word vectors, since 2.000.000 vectors
    # is an excessive amount. Use `|| true` to suppress curl error as to not
    # crash the whole script due to Bash `pipefail` option.
    curl "${fasttext_url}/cc.${lng}.${embdim}.vec.gz" |
        gunzip | sed "1s/^[1-9][0-9]*/${nvectors}/" |
        head -n "$((nvectors+1))" > "$embeddingsdir/cc.${lng}.${embdim}.vec" || true
}

download_embeddings() {
    local lng
    for lng in "$@"; do
        [[ -f ${embeddingsdir}/cc.${lng}.${embdim}.bin ]] || download_fasttext_model "$lng"
    done
}

compute_alignments() {
    # TODO choosable alignment method
    local lng
    local pivot="$1"
    shift

    # TODO copy dictionaries for language pairs
    [[ -e $dictdir ]] || download_dictionaries

    for lng in "$@"; do

        [[ $pivot = "$lng" ]] && continue

        # fasttext saves the aligned embeddings to the output file (`--output`)
        # and the alignment matrix to the output file + "-mat" suffix. Since we
        # only require the matrix, we discard the embeddings by writing to
        # /dev/null (through a symbolic link hack)
        outfile="$embeddingsdir/aligned.$lng.vec"
        ln -s /dev/null "$outfile"

        python -u "$fasttext"/alignment/align.py \
            --src_emb "$embeddingsdir/cc.$lng.$embdim.vec" \
            --tgt_emb "$embeddingsdir/cc.$pivot.$embdim.vec" \
            --dico_train "$dictdir/$lng-$pivot.txt" \
            --dico_test  "$dictdir/$lng-$pivot.0-5000.txt" \
            --output "$outfile" 2>&1 | tee "$logdir/align.$lng.log"
    done

    python "$SCRIPT_DIR"/scripts/fasttext/write-eye.py \
        "$embdim" > "$embeddingsdir/aligned.$pivot.vec-mat"
}
