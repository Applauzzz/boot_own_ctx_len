# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29501
# export OMP_NUM_THREADS=1

# torchrun --nproc_per_node=7 \
rm -rf /home/zhliu/database/mds_train_samples
python step_b2_retrieve_qfs_answer_make_trainset.py \
  --inst-mds /home/zhliu/database/mds_instructions \
  --text-mds /home/zhliu/database/mds_wiki \
  --faiss-index /home/zhliu/database/faiss_wiki_full/faiss.index \
  --out-mds /home/zhliu/database/mds_train_samples \
  --database-path /home/zhliu/database/wiki_db \
  --retriever-model /home/zhliu/model_base/e5-mistral-7B \
  --summarizer-model /home/zhliu/model_base/Qwen2.5-7B-Instruct \
  --answer-model     /home/zhliu/model_base/Qwen2.5-7B-Instruct \
  --topk 12 --per-query-topk 8 --n-queries-limit 6 \
  --chunk-tokens 4096 --summary-budget 4096 \
  --answer-max-new 1536 \
  --retriever-dtype fp16 --summarizer-dtype fp16 --answer-dtype fp16 \
  --per-rank-buffer 16 --compression zstd
