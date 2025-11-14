python build_faiss.py \
  --train-mds /home/zhliu/database/mds_wiki_emb_subset \
  --full-mds  /home/zhliu/database/mds_wiki_emb_full \
  --out-dir       /home/zhliu/database/faiss_wiki_full \
  --nlist 200 \
  --nprobe 32 \
  --metric ip
