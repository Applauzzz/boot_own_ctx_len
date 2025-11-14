MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
rm -r /home/zhliu/database/mds_wiki_tokens
torchrun --nproc_per_node=7 \
  dist_tokenize.py \
  --text-mds /home/zhliu/database/mds_wiki \
  --out-mds  /home/zhliu/database/mds_wiki_tokens \
  --model    /home/zhliu/model_base/e5-mistral-7B \
  --size-per-writer 512
