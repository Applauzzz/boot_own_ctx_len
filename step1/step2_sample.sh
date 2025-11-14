python sample_id_rank0_no_reservoir.py \
  --tokens-mds /home/zhliu/database/mds_wiki_tokens \
  --out-mds    /home/zhliu/database/mds_wiki_tokens_subset \
  --min-len 2000 --max-len 32000 --target 10000 --seed 42 \
  --size-per-writer 512
