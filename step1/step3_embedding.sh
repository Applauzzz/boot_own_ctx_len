#!/usr/bin/env bash
set -euo pipefail

TOKENS_MDS=/home/zhliu/database/mds_wiki_tokens   
EMB_MDS=/home/zhliu/database/mds_wiki_emb_full
MODEL=/home/zhliu/model_base/e5-mistral-7B        
NPROC=7

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29550
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=${NPROC} dist_embed_doc_2.py \
    --tokens-mds "${TOKENS_MDS}" \
    --out-mds    "${EMB_MDS}" \
    --model      "${MODEL}" \
    --batch-size 8 \
    --max-length 4096 \
    --torch-dtype fp16 \
    --size-limit-mb 512 --compression zstd
