#FROM ubuntu:22.04
#ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && apt-get install --no-install-recommends -y python3-dev build-essential ed git ca-certificates cmake automake libtool libboost-regex-dev libpstreams-dev libpthread-stubs0-dev libxml2-dev libcurl4-openssl-dev libssl-dev language-pack-en libyaml-dev
FROM python:3

RUN pip install --no-cache-dir dvc numpy fasttext sacrebleu sacremoses

RUN git clone --depth=1 https://github.com/facebookresearch/fastText && \
        chmod +x fastText/alignment/align.py && \
        ln -s /fastText/alignment/align.py /usr/local/bin/fasttext_align

COPY bin /usr/local/bin

WORKDIR /experiment

COPY .dvc .dvc
COPY .git .git
#COPY data data

#COPY *.yaml ./
#RUN ln -sf dvc.stage1.yaml dvc.yaml
COPY dvc.stage1.yaml dvc.yaml
