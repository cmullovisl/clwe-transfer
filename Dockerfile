FROM python:3

COPY seq2seq-con seq2seq-con
RUN pip install --no-cache-dir -U torch seq2seq-con/OpenNMT-py --extra-index-url https://download.pytorch.org/whl/cu113

RUN git clone --depth=1 https://github.com/facebookresearch/fastText && \
        chmod +x fastText/alignment/align.py && \
        ln -s /fastText/alignment/align.py /usr/local/bin/fasttext_align

WORKDIR /src
COPY onmt_vocab_utils onmt_vocab_utils
COPY bin bin
COPY setup.py setup.py
RUN pip install --no-cache-dir .

COPY . .
RUN ln -s dvc.stage1.yaml dvc.yaml
RUN mkdir -p embeddings/models data

ENTRYPOINT ["dvc"]
